"""
app.py — Nino Production-Grade Multi-Model Backend (v3 — Low-Latency Edition).

KEY SPEED IMPROVEMENTS (100% DROP-IN COMPATIBLE WITH EXISTING FRONTEND):
  1. Zero blocking Redis ping() or full-DB blob download on Vercel cold start.
  2. Per-user Redis keys (`nino:sess:{username}` with 30-day TTL) + fast in-memory LRU cache.
  3. Single batched session write per /guide turn instead of multiple redundant flushes.
  4. Lean conversation history window (6 turns, 1200 chars/turn) to minimize prefill latency.
  5. Fixed stale `progress` return value when `[PROGRESS_UP]` triggers.
  6. Bonus optional `/guide/stream` SSE endpoint for instant token streaming if wired up later.
"""

from collections import OrderedDict
import datetime
import json
import os
import re
import threading
import time
import uuid

import requests
from flask import Flask, Response, jsonify, make_response, render_template, request, stream_with_context
from google.genai import types

from engine import build_test, call_gemini, detect_task, friendly_error, parse_test_request, stream_gemini

# ═══════════════════════════════════════════════════════════════════════
# 1. WSGI SETUP & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
app = Flask(__name__, template_folder='templates')
application = app

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf'}
DAILY_SHARED_LIMIT = 15
FEEDBACK_PER_HOUR = 10
TEST_TTL_SECONDS = 7200          # 2 hours secret answer key TTL
SESSION_TTL_SECONDS = 2_592_000  # 30 days per-user Redis TTL
MAX_CACHED_SESSIONS = 200

# ═══════════════════════════════════════════════════════════════════════
# 2. LAZY UPSTASH REDIS (NO BLOCKING COLD-START PING)
# ═══════════════════════════════════════════════════════════════════════
_redis = None
_redis_initialized = False
_redis_lock = threading.Lock()

def _get_redis():
    global _redis, _redis_initialized
    if _redis_initialized:
        return _redis
    with _redis_lock:
        if _redis_initialized:
            return _redis
        _redis_initialized = True
        url = os.environ.get("UPSTASH_REDIS_REST_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if not (url and token):
            return None
        try:
            from upstash_redis import Redis
            _redis = Redis(url=url, token=token)
        except Exception as exc:
            print(f"[store] Redis init skipped, using in-memory fallback: {exc}")
            _redis = None
        return _redis

# ── Rate Limiter (Daily 15 msgs / visitor, bypassed with BYOK) ────────
_local_usage = {}
_usage_lock = threading.Lock()

def check_rate_limit(identifier: str, limit: int = DAILY_SHARED_LIMIT) -> bool:
    today = datetime.date.today().isoformat()
    r = _get_redis()
    if r:
        try:
            key = f"rl:{identifier}:{today}"
            count = r.incr(key)
            if count == 1:
                r.expire(key, 86400)
            return count <= limit
        except Exception as exc:
            print(f"[rate_limit redis error]: {exc}")

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

# ── Feedback Rate Limiter (Hourly 10 / visitor) ───────────────────────
_fb_usage = {}
_fb_lock = threading.Lock()

def check_feedback_limit(identifier: str) -> bool:
    hour_bucket = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H")
    r = _get_redis()
    if r:
        try:
            key = f"rl:fb:{identifier}:{hour_bucket}"
            count = r.incr(key)
            if count == 1:
                r.expire(key, 3600)
            return count <= FEEDBACK_PER_HOUR
        except Exception as exc:
            print(f"[feedback_limit redis error]: {exc}")

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

# ── Test Secret Answer Key Store (Server-Side Isolation) ──────────────
_local_tests = {}
_test_lock = threading.Lock()

def store_test(questions: list, ttl_seconds: int = TEST_TTL_SECONDS) -> str:
    """Store complete test questions with answer keys and solutions server-side."""
    test_id = uuid.uuid4().hex[:16]
    payload = json.dumps(questions, ensure_ascii=False)
    r = _get_redis()
    if r:
        try:
            r.setex(f"test:{test_id}", ttl_seconds, payload)
            return test_id
        except Exception as exc:
            print(f"[store_test redis error]: {exc}")

    with _test_lock:
        now = time.time()
        expired = [k for k, v in _local_tests.items() if v["expires_at"] < now]
        for k in expired:
            del _local_tests[k]
        _local_tests[test_id] = {"data": payload, "expires_at": now + ttl_seconds}
    return test_id

def retrieve_test(test_id: str) -> list | None:
    """Retrieve test questions with answer keys and solutions by test_id."""
    r = _get_redis()
    if r:
        try:
            data = r.get(f"test:{test_id}")
            if data:
                return json.loads(data)
        except Exception as exc:
            print(f"[retrieve_test redis error]: {exc}")

    with _test_lock:
        entry = _local_tests.get(test_id)
        if entry and entry["expires_at"] > time.time():
            return json.loads(entry["data"])
    return None

def delete_test(test_id: str) -> None:
    """Delete test session from store after grading."""
    r = _get_redis()
    if r:
        try:
            r.delete(f"test:{test_id}")
        except Exception:
            pass
    with _test_lock:
        _local_tests.pop(test_id, None)

# ── Per-User Session Store (LRU RAM Cache + Per-User Redis Key) ───────
class SessionManager:
    def __init__(self, filepath: str = "local_memory.json"):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._disk_lock = threading.Lock()
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._local_loaded = False

    @staticmethod
    def _default(goal: str = "General Study Optimization") -> dict:
        return {
            "pct": 10,
            "goal": goal,
            "history": [],
            "settings": {"mode": "adaptive", "theme": "dark"},
        }

    @staticmethod
    def _redis_key(username: str) -> str:
        return f"nino:sess:{username}"

    def _ensure_local_disk_loaded(self):
        """Only loads local disk JSON when running locally without Redis."""
        if self._local_loaded or _get_redis() is not None:
            return
        self._local_loaded = True
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    for k, v in list(data.items())[-MAX_CACHED_SESSIONS:]:
                        self._cache[k] = v
            except Exception as exc:
                print(f"[SessionManager file load error]: {exc}")

    def _put_cache(self, username: str, session: dict):
        self._cache[username] = session
        self._cache.move_to_end(username)
        while len(self._cache) > MAX_CACHED_SESSIONS:
            self._cache.popitem(last=False)

    def _load_user(self, username: str) -> dict | None:
        with self._lock:
            self._ensure_local_disk_loaded()
            if username in self._cache:
                self._cache.move_to_end(username)
                return self._cache[username]

        r = _get_redis()
        if r:
            try:
                raw = r.get(self._redis_key(username))
                if raw:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    with self._lock:
                        self._put_cache(username, parsed)
                    return parsed
            except Exception as exc:
                print(f"[SessionManager redis get error]: {exc}")
        return None

    def _persist_user(self, username: str, session: dict):
        payload = json.dumps(session, ensure_ascii=False)
        r = _get_redis()
        if r:
            try:
                r.setex(self._redis_key(username), SESSION_TTL_SECONDS, payload)
                return
            except Exception as exc:
                print(f"[SessionManager redis set error]: {exc}")

        # Local dev fallback (silently ignored on read-only Vercel FS)
        with self._lock:
            snapshot = json.dumps(dict(self._cache), ensure_ascii=False)
        with self._disk_lock:
            try:
                tmp = self.filepath + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(snapshot)
                os.replace(tmp, self.filepath)
            except Exception:
                pass

    def _copy(self, s: dict) -> dict:
        return {
            "pct": s.get("pct", 10),
            "goal": s.get("goal", "General Study Optimization"),
            "history": list(s.get("history", [])),
            "settings": dict(s.get("settings", {"mode": "adaptive", "theme": "dark"})),
        }

    def get_or_create(self, username: str, default_interest: str = "General Study Optimization") -> dict:
        s = self._load_user(username)
        if s is None:
            s = self._default(default_interest)
            with self._lock:
                self._put_cache(username, s)
            self._persist_user(username, s)
        return self._copy(s)

    def load_for_turn(self, username: str, interest: str) -> dict:
        """Loads user session and updates goal in memory (persisted once at commit_turn)."""
        s = self._load_user(username)
        with self._lock:
            if s is None:
                s = self._default(interest)
            elif interest and interest != s.get("goal"):
                s["goal"] = interest
            self._put_cache(username, s)
            return self._copy(s)

    def commit_turn(self, username: str, user_content: str, ai_content: str,
                    test: dict | None = None, route: str | None = None,
                    advance_pct: bool = False) -> int:
        """Single atomic update + persist at the end of a turn. Returns updated pct."""
        s = self._load_user(username)
        with self._lock:
            if s is None:
                s = self._default()
            if advance_pct:
                s["pct"] = min(s.get("pct", 10) + 12, 100)
            h = s.setdefault("history", [])
            h.append({"role": "user", "content": user_content})
            entry = {"role": "model", "content": ai_content}
            if test:
                entry["test"] = test
            if route:
                entry["route"] = route
            h.append(entry)
            s["history"] = h[-40:]
            self._put_cache(username, s)
            snapshot = self._copy(s)

        self._persist_user(username, snapshot)
        return snapshot["pct"]

    def update_settings(self, username: str, new_settings: dict) -> dict:
        if not isinstance(new_settings, dict):
            new_settings = {}
        s = self._load_user(username)
        with self._lock:
            if s is None:
                s = self._default()
            s.setdefault("settings", {}).update(new_settings)
            self._put_cache(username, s)
            snapshot = self._copy(s)
        self._persist_user(username, snapshot)
        return snapshot["settings"]

    def get_chat(self, username: str) -> dict:
        s = self._load_user(username) or {}
        return {
            "history": list(s.get("history", [])),
            "goal": s.get("goal", "General Study Optimization"),
            "pct": s.get("pct", 10),
        }

    def delete_session(self, username: str) -> bool:
        existed = False
        with self._lock:
            if username in self._cache:
                del self._cache[username]
                existed = True
        r = _get_redis()
        if r:
            try:
                deleted = r.delete(self._redis_key(username))
                return bool(deleted) or existed
            except Exception:
                pass
        return existed

session_store = SessionManager()

# ═══════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_visitor_id() -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    ip = forwarded.split(',')[0].strip() if forwarded else request.remote_addr
    return request.cookies.get("nino_vid") or ip or "unknown_visitor"

def skey(name: str) -> str:
    return f"{get_visitor_id()}::{name}"

def get_api_key() -> tuple[str | None, bool]:
    """
    Returns (key, is_owner_key).
    If student provides personal BYOK key in X-User-Key, is_owner_key is False.
    """
    user_key = (request.headers.get("X-User-Key") or "").strip()
    if user_key:
        return user_key, False
    return os.environ.get("GEMINI_API_KEY"), True

def build_model_history(history: list, max_turns: int = 6) -> list:
    """Builds compact Gemini-compliant alternating history to minimize prefill latency."""
    recent = history[-max_turns:]
    out = []
    for m in recent:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 1200:
            content = content[:1200] + "..."
        role = "model" if m.get("role") in ("model", "assistant") else "user"
        out.append({"role": role, "content": content})
    # Ensure history doesn't start with a stray model message
    while out and out[0]["role"] == "model":
        out.pop(0)
    return out

def _build_contents(history: list, user_prompt: str, file_bytes=None, mime_type=None) -> list:
    history_msgs = build_model_history(history)
    contents = [
        types.Content(
            role=m["role"],
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in history_msgs
    ]
    curr_parts = []
    if file_bytes and mime_type:
        curr_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
    curr_parts.append(types.Part.from_text(text=user_prompt))
    contents.append(types.Content(role="user", parts=curr_parts))
    return contents

def _parse_guide_request():
    if request.content_type and 'multipart/form-data' in request.content_type:
        raw_name = request.form.get('name', '').strip()
        interest = request.form.get('interest', 'General Study Optimization').strip()
        followup = request.form.get('followup', '').strip()
        deep_solve = request.form.get('deep_solve', 'false').lower() == 'true'

        file = request.files.get('file')
        file_bytes = None
        mime_type = None
        if file and file.filename != '' and allowed_file(file.filename):
            file_bytes = file.read()
            mime_type = file.content_type or 'image/jpeg'
            if not followup:
                followup = "Please solve this JEE doubt step by step."
    else:
        data = request.get_json(silent=True) or {}
        raw_name = (data.get('name') or '').strip()
        interest = (data.get('interest') or 'General Study Optimization').strip()
        followup = (data.get('followup') or '').strip()
        deep_solve = bool(data.get('deep_solve', False))
        file_bytes = None
        mime_type = None

    return raw_name, interest, followup, deep_solve, file_bytes, mime_type

def run_bg(fn, *args):
    try:
        threading.Thread(target=fn, args=args, daemon=True).start()
    except Exception as e:
        print(f"[run_bg error]: {e}")

def _post_webhook(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=4)
    except Exception as e:
        print(f"[Discord webhook error]: {e}")

def send_chat_log_to_discord(name: str, goal: str, pct: int, user_msg: str, ai_reply: str, route: str):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    short_reply = ai_reply[:600] + "..." if len(ai_reply) > 600 else ai_reply
    content = (
        f"🚀 **Nino Turn Log** [{route}]\n"
        f"User: `{name[:30]}` | Track: {goal[:60]} | Progress: {pct}%\n"
        f"**User:** {user_msg[:300]}\n"
        f"**Nino:** {short_reply}"
    )[:1900]
    run_bg(_post_webhook, url, {"content": content, "allowed_mentions": {"parse": []}})

# ═══════════════════════════════════════════════════════════════════════
# 4. ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════
@app.route('/')
def home():
    """Serves index.html with secure HttpOnly visitor cookie."""
    resp = make_response(render_template('index.html'))
    if not request.cookies.get("nino_vid"):
        resp.set_cookie(
            "nino_vid",
            uuid.uuid4().hex,
            max_age=60 * 60 * 24 * 365,
            samesite="Lax",
            secure=True,
            httponly=True,
        )
    return resp

@app.route('/api/health', methods=['GET'])
def health():
    """Liveness probe."""
    return jsonify({
        "status": "ok",
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }), 200

@app.route('/api/profile', methods=['GET'])
def get_profile():
    """Returns user goal & progress percentage."""
    display = request.args.get('name', 'Anonymous')
    session = session_store.get_or_create(skey(display))
    return jsonify({
        "name": display,
        "goal": session.get("goal", "General Study Optimization"),
        "progress": session.get("pct", 10)
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Retrieves or updates user settings."""
    if request.method == 'POST':
        body = request.get_json(silent=True) or {}
        name = skey(body.get('name', 'Anonymous'))
        updated = session_store.update_settings(name, body.get('settings', {}))
        return jsonify({"status": "success", "settings": updated})

    name = skey(request.args.get('name', 'Anonymous'))
    session = session_store.get_or_create(name)
    return jsonify({"settings": session.get('settings', {"mode": "adaptive", "theme": "dark"})})

@app.route('/api/chat/<chat_id>', methods=['GET'])
def get_chat(chat_id: str):
    """Retrieves session chat history."""
    return jsonify(session_store.get_chat(skey(chat_id)))

@app.route('/api/chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id: str):
    """Clears conversation history."""
    deleted = session_store.delete_session(skey(chat_id))
    return jsonify({"status": "deleted" if deleted else "not_found"}), 200

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Logs user feedback to Discord webhook subject to hourly rate limit."""
    vid = get_visitor_id()
    if not check_feedback_limit(vid):
        return jsonify({"status": "error", "error": "Rate limit exceeded"}), 429

    body = request.get_json(silent=True) or {}
    msg = str(body.get('message', '')).strip()[:1000]
    rating = max(0, min(5, int(body.get('rating', 0))))
    url = os.environ.get("DISCORD_FEEDBACK_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        stars = "★" * rating + "☆" * (5 - rating)
        payload = {
            "content": f"📝 **Feedback [{body.get('category', 'other')} - {stars}]**\nVisitor: `{vid}`\n{msg}"
        }
        run_bg(_post_webhook, url, payload)
    return jsonify({"status": "ok"}), 200

@app.route('/api/test/submit', methods=['POST'])
def submit_test():
    """Secure server-side test grading."""
    body = request.get_json(silent=True) or {}
    test_id = body.get('test_id')
    user_answers = body.get('answers') or {}

    if not test_id:
        return jsonify({"error": "Missing test_id"}), 400

    secret_questions = retrieve_test(test_id)
    if not secret_questions:
        return jsonify({"error": "Test session expired or not found. Please retake the test."}), 404

    total = len(secret_questions)
    correct_count = 0
    mistake_counts = {"concept": 0, "formula": 0, "silly": 0}
    graded_questions = []

    for idx, q in enumerate(secret_questions, 1):
        q_id = str(q.get("id", idx))
        selected_raw = user_answers.get(q_id)
        selected = int(selected_raw) if selected_raw is not None else None
        is_correct = (selected is not None and selected == q.get("correct"))
        mt = q.get("mistakeType", "Concept")

        if is_correct:
            correct_count += 1
        elif selected is not None:
            mt_lower = mt.lower()
            if mt_lower == "concept":
                mistake_counts["concept"] += 1
            elif mt_lower == "silly":
                mistake_counts["silly"] += 1
            else:
                mistake_counts["formula"] += 1

        graded_questions.append({
            "id": q.get("id", idx),
            "question": q["question"],
            "options": q["options"],
            "selected": selected,
            "correct": q.get("correct"),
            "isCorrect": is_correct,
            "mistakeType": mt,
            "solution": q.get("solution", ""),
        })

    delete_test(test_id)
    accuracy_pct = round((correct_count / total * 100)) if total else 0

    return jsonify({
        "status": "graded",
        "total": total,
        "score": correct_count,
        "accuracy": accuracy_pct,
        "mistakeBreakdown": mistake_counts,
        "questions": graded_questions,
    }), 200

@app.route('/guide', methods=['POST'])
def guide():
    """Main message dispatcher — 100% compatible JSON response contract."""
    try:
        raw_name, interest, followup, deep_solve, file_bytes, mime_type = _parse_guide_request()
        name = skey(raw_name if raw_name else "Anonymous")
        current_state = session_store.load_for_turn(name, interest)
        user_prompt = followup if followup else f"Guide me for {current_state['goal']}"

        api_key, is_owner_key = get_api_key()
        vid = get_visitor_id()
        if is_owner_key and not check_rate_limit(vid, DAILY_SHARED_LIMIT):
            return jsonify({
                "response": "You've used today's free messages. Add your personal Gemini key in Settings to keep going without limits.",
                "progress": current_state["pct"],
                "isTestTrigger": False,
                "questions": [],
            }), 429

        task = detect_task(user_prompt, bool(file_bytes), deep_solve)
        t0 = time.perf_counter()

        # ── Route: TEST GENERATION ────────────────────────────────────
        if task == 'test':
            count, subject, hint = parse_test_request(user_prompt)
            title, secret_questions, err = build_test(count, subject, hint, api_key)
            print(f"[nino-v3] task=test count={len(secret_questions)}/{count} elapsed={time.perf_counter()-t0:.2f}s")

            if not secret_questions:
                return jsonify({
                    "response": friendly_error(err) if err else "Test generation was interrupted. Please try asking again.",
                    "progress": current_state["pct"],
                    "isTestTrigger": False,
                    "questions": [],
                })

            test_id = store_test(secret_questions, ttl_seconds=TEST_TTL_SECONDS)
            public_questions = [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "options": q["options"],
                    "mistakeType": q.get("mistakeType", "Concept"),
                }
                for q in secret_questions
            ]

            friendly_msg = f"Here is your {title} test set ({len(public_questions)} questions). Solve with precision before submitting."
            pct = session_store.commit_turn(
                name, user_prompt, friendly_msg,
                test={"title": title, "questions": public_questions},
                route="test",
            )
            send_chat_log_to_discord(name, current_state["goal"], pct, user_prompt, friendly_msg, "test")

            return jsonify({
                "chatResponse": friendly_msg,
                "response": friendly_msg,
                "isTestTrigger": True,
                "testTitle": title,
                "test_id": test_id,
                "questions": public_questions,
                "progress": pct,
            })

        # ── Route: SOLVER / CHAT ──────────────────────────────────────
        contents = _build_contents(current_state["history"], user_prompt, file_bytes, mime_type)
        raw_response, err = call_gemini(task, contents, api_key)
        print(f"[nino-v3] task={task} elapsed={time.perf_counter()-t0:.2f}s")

        if not raw_response:
            return jsonify({
                "response": friendly_error(err),
                "progress": current_state["pct"],
                "isTestTrigger": False,
                "questions": [],
            })

        # Smart fallback if chat model emits [TRIGGER_TEST: ...]
        test_trigger_match = re.search(r'\[TRIGGER_TEST:\s*([^\]]+)\]', raw_response)
        if test_trigger_match:
            test_topic = test_trigger_match.group(1).strip()
            clean_reply = raw_response.replace(test_trigger_match.group(0), "").strip()
            count, subject, hint = parse_test_request(test_topic or user_prompt)
            title, secret_questions, _ = build_test(count, subject, hint, api_key)

            if secret_questions:
                test_id = store_test(secret_questions, ttl_seconds=TEST_TTL_SECONDS)
                public_questions = [
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "options": q["options"],
                        "mistakeType": q.get("mistakeType", "Concept"),
                    }
                    for q in secret_questions
                ]
                final_chat_msg = clean_reply or f"Here is your {title} test set ({len(public_questions)} questions)."
                pct = session_store.commit_turn(
                    name, user_prompt, final_chat_msg,
                    test={"title": title, "questions": public_questions},
                    route="test",
                )
                send_chat_log_to_discord(name, current_state["goal"], pct, user_prompt, final_chat_msg, "test_fallback")

                return jsonify({
                    "chatResponse": final_chat_msg,
                    "response": final_chat_msg,
                    "isTestTrigger": True,
                    "testTitle": title,
                    "test_id": test_id,
                    "questions": public_questions,
                    "progress": pct,
                })

        advance_pct = "[PROGRESS_UP]" in raw_response
        clean_response = raw_response.replace("[PROGRESS_UP]", "").strip() if advance_pct else raw_response

        pct = session_store.commit_turn(
            name, user_prompt, clean_response,
            route=task, advance_pct=advance_pct,
        )
        send_chat_log_to_discord(name, current_state["goal"], pct, user_prompt, clean_response, task)

        return jsonify({
            "response": clean_response,
            "progress": pct,
            "isTestTrigger": False,
            "testTitle": "",
            "questions": [],
        })

    except Exception as e:
        print(f"[/guide Critical ERROR]: {e}")
        return jsonify({"response": "Nino encountered an unexpected system error. Please try again."}), 500

@app.route('/guide/stream', methods=['POST'])
def guide_stream():
    """
    Optional Server-Sent Events (SSE) endpoint for instant token-by-token responses
    (~0.8-1.5s Time-To-First-Token on chat). Wire up in index.html whenever ready.
    """
    raw_name, interest, followup, deep_solve, file_bytes, mime_type = _parse_guide_request()
    name = skey(raw_name if raw_name else "Anonymous")
    current_state = session_store.load_for_turn(name, interest)
    user_prompt = followup if followup else f"Guide me for {current_state['goal']}"

    api_key, is_owner_key = get_api_key()
    vid = get_visitor_id()
    if is_owner_key and not check_rate_limit(vid, DAILY_SHARED_LIMIT):
        return jsonify({"error": "Daily free limit reached."}), 429

    task = detect_task(user_prompt, bool(file_bytes), deep_solve)
    if task == 'test':
        # Delegate tests to standard JSON payload wrapped as a single SSE event
        def _test_sse():
            count, subject, hint = parse_test_request(user_prompt)
            title, secret_questions, err = build_test(count, subject, hint, api_key)
            if not secret_questions:
                yield f"data: {json.dumps({'type': 'error', 'error': friendly_error(err)})}\n\n"
                return
            test_id = store_test(secret_questions, ttl_seconds=TEST_TTL_SECONDS)
            public_questions = [
                {"id": q["id"], "question": q["question"], "options": q["options"], "mistakeType": q.get("mistakeType", "Concept")}
                for q in secret_questions
            ]
            msg = f"Here is your {title} test set ({len(public_questions)} questions). Solve with precision before submitting."
            pct = session_store.commit_turn(name, user_prompt, msg, test={"title": title, "questions": public_questions}, route="test")
            yield f"data: {json.dumps({'type': 'test', 'response': msg, 'testTitle': title, 'test_id': test_id, 'questions': public_questions, 'progress': pct})}\n\n"
        return Response(stream_with_context(_test_sse()), mimetype="text/event-stream")

    contents = _build_contents(current_state["history"], user_prompt, file_bytes, mime_type)

    def _generate():
        chunks = []
        try:
            for piece in stream_gemini(task, contents, api_key):
                chunks.append(piece)
                yield f"data: {json.dumps({'type': 'chunk', 'text': piece})}\n\n"
            full_text = "".join(chunks)
            advance_pct = "[PROGRESS_UP]" in full_text
            clean = full_text.replace("[PROGRESS_UP]", "").strip() if advance_pct else full_text
            pct = session_store.commit_turn(name, user_prompt, clean, route=task, advance_pct=advance_pct)
            send_chat_log_to_discord(name, current_state["goal"], pct, user_prompt, clean, task)
            yield f"data: {json.dumps({'type': 'done', 'progress': pct})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': friendly_error(str(exc))})}\n\n"

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
