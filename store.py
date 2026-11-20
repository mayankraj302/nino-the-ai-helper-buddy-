"""
store.py — Storage & Security Architecture for Nino.
Provides:
  - Upstash Redis support with graceful thread-safe in-memory fallbacks
  - Daily rate limiter for visitors/IPs
  - Hourly rate limiter for feedback
  - Secure test question/key store with TTL and server-side secret grading isolation
  - Thread-safe SessionManager with atomic disk persistence and Vercel read-only FS safety

NOTE: LEAF MODULE — must NEVER import app, router, test_engine, or prompts.
Only standard library and optional upstash_redis.
"""

from datetime import date, datetime, timezone
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

# ═══════════════════════════════════════════════════════════════════════
# 1. UPSTASH REDIS INITIALISATION (optional — silent no-op when absent)
# ═══════════════════════════════════════════════════════════════════════
_redis = None


def _init_redis():
    global _redis
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not (url and token):
        return None
    try:
        from upstash_redis import Redis
        client = Redis(url=url, token=token)
        client.ping()
        print("[store] Upstash Redis connected successfully")
        return client
    except Exception as exc:
        print(f"[store] Redis init failed, using in-memory: {exc}")
        return None


_redis = _init_redis()


# ═══════════════════════════════════════════════════════════════════════
# 2. RATE LIMITER (Daily Limit per visitor / IP)
# ═══════════════════════════════════════════════════════════════════════
_local_usage: Dict[tuple, int] = {}       # {(identifier, iso_date): count}
_usage_lock = threading.Lock()


def check_rate_limit(identifier: str, limit: int = 15) -> bool:
    """
    Returns True if request is allowed under the daily rate limit (limit requests/day).
    Increments count on check. Never crashes; falls back to local memory.
    """
    today = date.today().isoformat()

    # Redis path
    if _redis:
        try:
            key = f"rl:{identifier}:{today}"
            count = _redis.incr(key)
            if count == 1:
                _redis.expire(key, 86400)
            return count <= limit
        except Exception as exc:
            print(f"[rate_limit redis ERROR]: {exc}")
            # fall through to in-memory fallback

    # In-memory fallback
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
# 3. TEST GRADING STORE (Server-Side Answer Keys & Solutions)
# ═══════════════════════════════════════════════════════════════════════
_local_tests: Dict[str, dict] = {}        # {test_id: {"data": json_str, "expires_at": float}}
_test_lock = threading.Lock()


def store_test(questions: list, ttl_seconds: int = 7200) -> str:
    """
    Persist full test with answer keys and solutions.
    Returns generated test_id string.
    Never exposes keys to client.
    """
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
        expired = [k for k, v in _local_tests.items() if v["expires_at"] < now]
        for k in expired:
            del _local_tests[k]
        _local_tests[test_id] = {"data": payload, "expires_at": now + ttl_seconds}
    return test_id


def retrieve_test(test_id: str) -> Optional[list]:
    """
    Retrieve test question list by test_id.
    Returns None if missing or expired.
    """
    if _redis:
        try:
            data = _redis.get(f"test:{test_id}")
            if data:
                return json.loads(data)
        except Exception as exc:
            print(f"[test_retrieve redis ERROR]: {exc}")

    with _test_lock:
        entry = _local_tests.get(test_id)
        if entry and entry["expires_at"] > time.time():
            return json.loads(entry["data"])
    return None


def delete_test(test_id: str) -> None:
    """
    Delete test from storage after grading.
    """
    if _redis:
        try:
            _redis.delete(f"test:{test_id}")
        except Exception:
            pass
    with _test_lock:
        _local_tests.pop(test_id, None)


# ═══════════════════════════════════════════════════════════════════════
# 4. FEEDBACK RATE LIMITER (Hourly Limit)
# ═══════════════════════════════════════════════════════════════════════
_fb_usage: Dict[tuple, int] = {}
_fb_lock = threading.Lock()
FEEDBACK_PER_HOUR = 10


def check_feedback_limit(identifier: str) -> bool:
    """
    Returns True if user/visitor is within the hourly feedback limit (10/hr).
    """
    hour_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")

    if _redis:
        try:
            key = f"rl:fb:{identifier}:{hour_bucket}"
            count = _redis.incr(key)
            if count == 1:
                _redis.expire(key, 3600)
            return count <= FEEDBACK_PER_HOUR
        except Exception as exc:
            print(f"[feedback_limit redis ERROR]: {exc}")

    bucket = (identifier, hour_bucket)
    with _fb_lock:
        if len(_fb_usage) > 2000:
            for k in [k for k in _fb_usage if k[1] != hour_bucket]:
                del _fb_usage[k]
        count = _fb_usage.get(bucket, 0)
        if count >= FEEDBACK_PER_HOUR:
            return False
        _fb_usage[bucket] = count + 1
        return True


# ═══════════════════════════════════════════════════════════════════════
# 5. SESSION MANAGER (Thread-Safe, Sticky Routing, Atomic Disk Flush)
# ═══════════════════════════════════════════════════════════════════════
class SessionManager:
    """
    Thread-safe session storage supporting Redis with file + memory fallback.
    Maintains history, goal, user settings, progress percentage, and last route.
    Guarantees atomic disk flush with safety on read-only environments (e.g. Vercel).
    """

    def __init__(self, filepath: str = "local_memory.json"):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._disk_lock = threading.Lock()
        self._save_scheduled = False
        self._sessions: Dict[str, dict] = self._load()

    # ── persistence ──────────────────────────────────────────────────
    def _load(self) -> dict:
        if _redis:
            try:
                raw = _redis.get("nino:sessions")
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                print(f"[SessionManager redis load ERROR]: {exc}")

        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                print(f"[SessionManager load ERROR]: {exc}")
        return {}

    def _flush(self):
        with self._lock:
            self._save_scheduled = False
            snapshot = json.dumps(self._sessions, ensure_ascii=False)

        if _redis:
            try:
                _redis.set("nino:sessions", snapshot)
            except Exception as exc:
                print(f"[SessionManager redis save ERROR]: {exc}")

        with self._disk_lock:
            try:
                tmp = self.filepath + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(snapshot)
                os.replace(tmp, self.filepath)
            except Exception:
                # Silently catch disk write/replace failures for read-only filesystems (Vercel)
                pass

    def _schedule_save(self):
        with self._lock:
            if self._save_scheduled:
                return
            self._save_scheduled = True
        threading.Thread(target=self._flush, daemon=True).start()

    def flush_sync(self):
        """Perform an immediate synchronous flush to Redis and disk."""
        self._flush()

    # ── default session template ─────────────────────────────────────
    @staticmethod
    def _default(goal: str = "General Study Optimization") -> dict:
        return {
            "pct": 10,
            "goal": goal,
            "history": [],
            "settings": {"mode": "adaptive", "theme": "dark"},
        }

    # ── public API ───────────────────────────────────────────────────
    def get_or_create(self, username: str, default_interest: str = "General Study Optimization") -> dict:
        created = False
        with self._lock:
            if username not in self._sessions:
                self._sessions[username] = self._default(default_interest)
                created = True
            s = self._sessions[username]
            out = {
                "pct": s["pct"],
                "goal": s["goal"],
                "history": list(s["history"]),
                "settings": dict(s.get("settings", {})),
            }
        if created:
            self._schedule_save()
        return out

    def update_goal(self, username: str, interest: str) -> dict:
        changed = False
        with self._lock:
            if username not in self._sessions:
                self._sessions[username] = self._default(interest)
                changed = True
            elif interest and interest != self._sessions[username]["goal"]:
                self._sessions[username]["goal"] = interest
                changed = True
            s = self._sessions[username]
            out = {
                "pct": s["pct"],
                "goal": s["goal"],
                "history": list(s["history"]),
                "settings": dict(s.get("settings", {})),
            }
        if changed:
            self._schedule_save()
        return out

    def update_settings(self, username: str, new_settings: dict) -> dict:
        if not isinstance(new_settings, dict):
            new_settings = {}
        with self._lock:
            if username not in self._sessions:
                self._sessions[username] = self._default()
            self._sessions[username].setdefault("settings", {}).update(new_settings)
            out = dict(self._sessions[username]["settings"])
        self._schedule_save()
        return out

    def get_chat(self, username: str) -> dict:
        with self._lock:
            s = self._sessions.get(username, {})
            return {
                "history": [
                    {k: v for k, v in m.items() if k != "model_text"}
                    for m in s.get("history", [])
                ],
                "goal": s.get("goal", "General Study Optimization"),
                "pct": s.get("pct", 10),
            }

    def delete_session(self, username: str) -> bool:
        with self._lock:
            if username not in self._sessions:
                return False
            del self._sessions[username]
        self._schedule_save()
        return True

    def advance_progress(self, username: str, step: int = 12, maximum: int = 100):
        with self._lock:
            if username not in self._sessions:
                self._sessions[username] = self._default()
            current = self._sessions[username].get("pct", 10)
            self._sessions[username]["pct"] = min(current + step, maximum)
        self._schedule_save()

    def get_last_route(self, username: str) -> Optional[str]:
        """
        Return the route_type of the last model message for sticky multi-turn routing.
        """
        with self._lock:
            s = self._sessions.get(username, {})
            for m in reversed(s.get("history", [])):
                if m.get("role") in ("model", "assistant") and m.get("route"):
                    return m["route"]
        return None

    def append_history(
        self,
        username: str,
        user_content: str,
        ai_content: str,
        test: Optional[dict] = None,
        model_text: Optional[str] = None,
        route: Optional[str] = None,
    ):
        with self._lock:
            if username not in self._sessions:
                self._sessions[username] = self._default()
            h = self._sessions[username]["history"]
            h.append({"role": "user", "content": user_content})
            entry: dict = {"role": "model", "content": ai_content}
            if test:
                entry["test"] = test
            if model_text:
                entry["model_text"] = model_text
            if route:
                entry["route"] = route
            h.append(entry)
            self._sessions[username]["history"] = h[-40:]
        self._schedule_save()
