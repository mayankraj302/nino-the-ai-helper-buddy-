"""
store.py — Unified store: Upstash Redis (primary, per-session keys) with local fallback.
"""

import json
import os
import threading
import time
import uuid
from datetime import date, datetime, timezone

# ═══════════════════════════════════════════════════════════════════════
# 1. REDIS INITIALISATION (non-blocking — no startup ping)
# ═══════════════════════════════════════════════════════════════════════
_redis = None


def _init_redis():
    global _redis
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not (url and token):
        return
    try:
        from upstash_redis import Redis
        _redis = Redis(url=url, token=token)
    except Exception as exc:
        print(f"[store] Redis init failed, using in-memory: {exc}")
        _redis = None


_init_redis()

IS_VERCEL = bool(os.environ.get("VERCEL"))
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


def is_redis_enabled() -> bool:
    return _redis is not None


# ═══════════════════════════════════════════════════════════════════════
# 2. RATE LIMITER (IP/visitor-based, per-day)
# ═══════════════════════════════════════════════════════════════════════
_local_usage: dict = {}
_usage_lock = threading.Lock()


def check_rate_limit(identifier: str, limit: int = 15) -> bool:
    today = date.today().isoformat()

    if _redis:
        try:
            key = f"rl:{identifier}:{today}"
            count = _redis.incr(key)
            if count == 1:
                _redis.expire(key, 86400)
            return count <= limit
        except Exception as exc:
            print(f"[rate_limit redis ERROR]: {exc}")

    bucket = (identifier, today)
    with _usage_lock:
        if len(_local_usage) > 2000:
            for k in [k for k in _local_usage if k[1] != today]:
                del _local_usage[k]
        count = _local_usage.get(bucket, 0)
        if count >= limit:
            return False
        _local_usage[bucket] = count + 1
        return True


# ═══════════════════════════════════════════════════════════════════════
# 3. TEST GRADING STORE (server-side answer keys)
# ═══════════════════════════════════════════════════════════════════════
_local_tests: dict = {}
_test_lock = threading.Lock()


def store_test(questions: list, ttl_seconds: int = 7200) -> str:
    test_id = uuid.uuid4().hex[:16]
    payload = json.dumps(questions, ensure_ascii=False)

    if _redis:
        try:
            _redis.setex(f"test:{test_id}", ttl_seconds, payload)
            return test_id
        except Exception as exc:
            print(f"[test_store redis ERROR]: {exc}")

    with _test_lock:
        now = time.time()
        for k in [k for k, v in _local_tests.items() if v["expires_at"] < now]:
            del _local_tests[k]
        _local_tests[test_id] = {"data": payload, "expires_at": now + ttl_seconds}
    return test_id


def retrieve_test(test_id: str) -> list | None:
    if _redis:
        try:
            data = _redis.get(f"test:{test_id}")
            if data:
                return json.loads(data) if isinstance(data, str) else data
        except Exception as exc:
            print(f"[test_retrieve redis ERROR]: {exc}")

    with _test_lock:
        entry = _local_tests.get(test_id)
        if entry and entry["expires_at"] > time.time():
            return json.loads(entry["data"])
    return None


def delete_test(test_id: str):
    if _redis:
        try:
            _redis.delete(f"test:{test_id}")
        except Exception:
            pass
    with _test_lock:
        _local_tests.pop(test_id, None)


# ═══════════════════════════════════════════════════════════════════════
# 4. FEEDBACK RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════
_fb_usage: dict = {}
FEEDBACK_PER_HOUR = 10


def check_feedback_limit(identifier: str) -> bool:
    hour_tag = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    if _redis:
        try:
            key = f"fb:{identifier}:{hour_tag}"
            count = _redis.incr(key)
            if count == 1:
                _redis.expire(key, 3600)
            return count <= FEEDBACK_PER_HOUR
        except Exception:
            pass

    bucket = (identifier, hour_tag)
    if _fb_usage.get(bucket, 0) >= FEEDBACK_PER_HOUR:
        return False
    _fb_usage[bucket] = _fb_usage.get(bucket, 0) + 1
    return True


# ═══════════════════════════════════════════════════════════════════════
# 5. SESSION MANAGER (Per-session Redis keys, synchronous writes)
# ═══════════════════════════════════════════════════════════════════════
class SessionManager:
    """Thread-safe per-session storage. Uses Upstash Redis when configured, else /tmp or local JSON."""

    def __init__(self, filepath: str | None = None):
        default_path = "/tmp/nino_local_memory.json" if IS_VERCEL else "local_memory.json"
        self.filepath = filepath or default_path
        self._lock = threading.Lock()
        self._sessions: dict = {} if _redis else self._load_disk()

    @staticmethod
    def _default(goal: str = "General Study Optimization") -> dict:
        return {
            "pct": 10,
            "goal": goal,
            "history": [],
            "settings": {"mode": "adaptive", "theme": "dark"},
        }

    def _redis_key(self, username: str) -> str:
        return f"nino:sess:{username}"

    def _load_disk(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                print(f"[SessionManager load ERROR]: {exc}")
        return {}

    def _save_disk(self):
        try:
            snapshot = json.dumps(self._sessions, ensure_ascii=False)
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(snapshot)
            os.replace(tmp, self.filepath)
        except Exception:
            pass

    def _read_session(self, username: str) -> dict | None:
        if _redis:
            try:
                raw = _redis.get(self._redis_key(username))
                if raw:
                    return json.loads(raw) if isinstance(raw, str) else raw
                return None
            except Exception as exc:
                print(f"[SessionManager redis read ERROR]: {exc}")
        with self._lock:
            s = self._sessions.get(username)
            return json.loads(json.dumps(s)) if s else None

    def _write_session(self, username: str, data: dict):
        if _redis:
            try:
                _redis.setex(
                    self._redis_key(username),
                    SESSION_TTL_SECONDS,
                    json.dumps(data, ensure_ascii=False),
                )
                return
            except Exception as exc:
                print(f"[SessionManager redis write ERROR]: {exc}")
        with self._lock:
            self._sessions[username] = data
            self._save_disk()

    def get_or_create(self, username: str, default_interest: str = "General Study Optimization") -> dict:
        s = self._read_session(username)
        if s is None:
            s = self._default(default_interest)
            self._write_session(username, s)
        return s

    def update_goal(self, username: str, interest: str) -> dict:
        s = self._read_session(username)
        if s is None:
            s = self._default(interest or "General Study Optimization")
            self._write_session(username, s)
            return s
        # Only set goal if it's still the default placeholder
        if interest and s.get("goal") in ("", "General Study Optimization"):
            s["goal"] = interest[:80]
            self._write_session(username, s)
        return s

    def update_settings(self, username: str, new_settings: dict) -> dict:
        if not isinstance(new_settings, dict):
            new_settings = {}
        s = self._read_session(username) or self._default()
        s.setdefault("settings", {"mode": "adaptive", "theme": "dark"}).update(new_settings)
        self._write_session(username, s)
        return dict(s["settings"])

    def get_chat(self, username: str) -> dict:
        s = self._read_session(username) or {}
        return {
            "history": [{k: v for k, v in m.items() if k != "model_text"} for m in s.get("history", [])],
            "goal": s.get("goal", "General Study Optimization"),
            "pct": s.get("pct", 10),
        }

    def delete_session(self, username: str) -> bool:
        if _redis:
            try:
                deleted = _redis.delete(self._redis_key(username))
                return bool(deleted)
            except Exception as exc:
                print(f"[SessionManager redis delete ERROR]: {exc}")
        with self._lock:
            if username not in self._sessions:
                return False
            del self._sessions[username]
            self._save_disk()
            return True

    def advance_progress(self, username: str, step: int = 12, maximum: int = 100):
        s = self._read_session(username)
        if not s:
            return
        s["pct"] = min(s.get("pct", 10) + step, maximum)
        self._write_session(username, s)

    def get_last_route(self, username: str) -> str | None:
        s = self._read_session(username) or {}
        for m in reversed(s.get("history", [])):
            if m.get("role") in ("model", "assistant") and m.get("route"):
                return m["route"]
        return None

    def append_history(
        self,
        username: str,
        user_content: str,
        ai_content: str,
        test: dict | None = None,
        model_text: str | None = None,
        route: str | None = None,
    ):
        s = self._read_session(username) or self._default()
        h = s.setdefault("history", [])
        h.append({"role": "user", "content": user_content})
        entry: dict = {"role": "model", "content": ai_content}
        if test:
            entry["test"] = test
        if model_text:
            entry["model_text"] = model_text
        if route:
            entry["route"] = route
        h.append(entry)
        s["history"] = h[-40:]
        self._write_session(username, s)
