"""
app.py — Nino Production-Grade Multi-Model Backend (v3).
WSGI Application, In-Memory & Redis Store, Server-Side Test Grading, and Clean 1-Turn Dispatch.

CRITICAL ARCHITECTURE:
  1. CLEAN DEPENDENCIES:
     - Imports: json, os, re, threading, time, datetime, uuid, requests
     - Flask: Flask, jsonify, make_response, render_template, request
     - google.genai: types
     - Local: prompts (CHAT_PROMPT, SOLVER_PROMPT, TEST_PROMPT),
              engine (detect_task, call_gemini, build_test, parse_test_request)
  2. WSGI SETUP:
     app = Flask(__name__, template_folder='templates')
     application = app
  3. IN-MEMORY & UPSTASH REDIS STORE:
     - Sessions: thread-safe dictionary with background disk persistence (and Redis if configured)
     - Rate limiting: 15 free messages/day on shared key, bypassed if BYOK in X-User-Key
     - Server-side test grading:
       Stores secret answer keys (TTL 2 hours), returns public questions without 'correct' or 'solution'.
       POST /api/test/submit grades answers, counts Concept/Formula/Silly mistakes, attaches solutions.
  4. SMART FALLBACK:
     - If chat model returns [TRIGGER_TEST: ...], seamlessly triggers test generation!
"""

import datetime
import json
import os
import re
import threading
import time
import uuid

import requests
from flask import Flask, jsonify, make_response, render_template, request
from google.genai import types

from prompts import CHAT_PROMPT, SOLVER_PROMPT, TEST_PROMPT
from engine import detect_task, call_gemini, build_test, parse_test_request

# ═══════════════════════════════════════════════════════════════════════
# 1. WSGI SETUP & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
app = Flask(__name__, template_folder='templates')
application = app

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf'}
DAILY_SHARED_LIMIT = 15
FEEDBACK_PER_HOUR = 10
TEST_TTL_SECONDS = 7200  # 2 hours secret answer key TTL

# ═══════════════════════════════════════════════════════════════════════
# 2. UPSTASH REDIS & IN-MEMORY STORE
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
        print(f"[store] Redis init failed, using in-memory fallback: {exc}")
        return None

_redis = _init_redis()

# ── Rate Limiter (Daily 15 msgs / visitor, bypassed with BYOK) ────────
_local_usage = {}  # {(identifier, iso_date): count}
_usage_lock = threading.Lock()

def check_rate_limit(identifier: str, limit: int = DAILY_SHARED_LIMIT) -> bool:
    today = datetime.date.today().isoformat()
    if _redis:
        try:
            key = f"rl:{identifier}:{today}"
            count = _redis.incr(key)
            if count == 1:
                _redis.expire(key, 86400)
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
    if _redis:
        try:
            key = f"rl:fb:{identifier}:{hour_bucket}"
            count = _redis.incr(key)
            if count == 1:
                _redis.expire(key, 3600)
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
_local_tests = {}  # {test_id: {"data": json_str, "expires_at": float}}
_test_lock = threading.Lock()

def store_test(questions: list, ttl_seconds: int = TEST_TTL_SECONDS) -> str:
    """Store complete test questions with answer keys and solutions server-side."""
    test_id = uuid.uuid4().hex[:16]
    payload = json.dumps(questions, ensure_ascii=False)
    if _redis:
        try:
            _redis.setex(f"test:{test_id}", ttl_seconds, payload)
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
    if _redis:
        try:
            data = _redis.get(f"test:{test_id}")
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
    if _redis:
        try:
            _redis.delete(f"test:{test_id}")
        except Exception:
            pass
    with _test_lock:
        _local_tests.pop(test_id, None)

# ── Thread-Safe Session Store with Disk Persistence ───────────────────
class SessionManager:
    def __init__(self, filepath: str = "local_memory.json"):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._disk_lock = threading.Lock()
        self._save_scheduled = False
        self._sessions = self._load()

    def _load(self) -> dict:
        if _redis:
            try:
                raw = _redis.get("nino:sessions")
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                print(f"[SessionManager redis load error]: {exc}")

        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                print(f"[SessionManager file load error]: {exc}")
        return {}

    def _flush(self):
        with self._lock:
            self._save_scheduled = False
            snapshot = json.dumps(self._sessions, ensure_ascii=False)

        if _redis:
            try:
                _redis.set("nino:sessions", snapshot)
            except Exception as exc:
                print(f"[SessionManager redis save error]: {exc}")

        with self._disk_lock:
            try:
                tmp = self.filepath + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(snapshot)
                os.replace(tmp, self.filepath)
            except Exception:
                # Silently catch read-only filesystem errors on Vercel
                pass

    def _schedule_save(self):
        with self._lock:
            if self._save_scheduled:
                return
            self._save_scheduled = True
        threading.Thread(target=self._flush, daemon=True).start()

    @staticmethod
    def _default(goal: str = "General Study Optimization") -> dict:
        return {
            "pct": 10,
            "goal": goal,
            "history": [],
            "settings": {"mode": "adaptive", "theme": "dark"}
        }

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
                "settings": dict(s.get("settings", {}))
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
                "settings": dict(s.get("settings", {}))
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
                "history": list(s.get("history", [])),
                "goal": s.get("goal", "General Study Optimization"),
                "pct": s.get("pct", 10)
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

    def append_history(self, username: str, user_content: str, ai_content: str, test: dict | None = None, route: str | None = None):
        with self._lock:
            if username not in self._sessions:
                self._sessions[username] = self._default()
            h = self._sessions[username]["history"]
            h.append({"role": "user", "content": user_content})
            entry = {"role": "model", "content": ai_content}
            if test:
                entry["test"] = test
            if route:
                entry["route"] = route
            h.append(entry)
            self._sessions[username]["history"] = h[-40:]
        self._schedule_save()

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

def build_model_history(history: list, max_turns: int = 8) -> list:
    recent = history[-max_turns:]
    out = []
    for m in recent:
        content = m.get("content", "")
        if len(content) > 2000:
            content = content[:2000] + "..."
        role = "model" if m.get("role") in ("model", "assistant") else "user"
        out.append({"role": role, "content": content})
    return out

def run_bg(fn, *args):
    try:
        t = threading.Thread(target=fn, args=args, daemon=True)
        t.start()
    except Exception as e:
        print(f"[run_bg error]: {e}")

def _post_webhook(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=6)
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
    fb = {
        "visitor": vid,
        "category": body.get("category", "other"),
        "rating": rating,
        "message": msg,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    url = os.environ.get("DISCORD_FEEDBACK_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        stars = "★" * rating + "☆" * (5 - rating)
        payload = {
            "content": f"📝 **Feedback [{fb['category']} - {stars}]**\nVisitor: `{vid}`\n{msg}"
        }
        run_bg(_post_webhook, url, payload)
    return jsonify({"status": "ok"}), 200

@app.route('/api/test/submit', methods=['POST'])
def submit_test():
    """
    Secure server-side test grading:
    Receives: { test_id: string, answers: { q_id: selected_index } }
    Retrieves secret key, grades answers, counts Concept/Formula/Silly mistakes,
    attaches solutions, deletes test session, and returns final score & breakdown.
    """
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
            "solution": q.get("solution", "")
        })

    delete_test(test_id)
    accuracy_pct = round((correct_count / total * 100)) if total else 0

    return jsonify({
        "status": "graded",
        "total": total,
        "score": correct_count,
        "accuracy": accuracy_pct,
        "mistakeBreakdown": mistake_counts,
        "questions": graded_questions
    }), 200

@app.route('/guide', methods=['POST'])
def guide():
    """
    Main message dispatcher:
    1. Handles multipart/form-data (photo upload) & JSON.
    2. Checks rate limits (15/day free shared quota, bypassed by BYOK).
    3. Calls detect_task(user_prompt, has_file, deep_solve).
    4. If task == 'test': calls build_test(), stores secret key, returns public questions + test_id.
    5. If task in ('solver', 'chat'): formats history, calls call_gemini(task, ...), logs to Discord in background, returns clean response.
    6. If chat model returns [TRIGGER_TEST: ...], seamlessly triggers test generation as a smart fallback!
    """
    try:
        # 1. Parse Multipart or JSON body
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

        name = skey(raw_name if raw_name else "Anonymous")
        current_state = session_store.update_goal(name, interest)
        user_prompt = followup if followup else f"Guide me for {current_state['goal']}"

        # 2. Key & Rate Limit check
        api_key, is_owner_key = get_api_key()
        vid = get_visitor_id()
        if is_owner_key and not check_rate_limit(vid, DAILY_SHARED_LIMIT):
            return jsonify({
                "response": "You've used today's free messages. Add your personal Gemini key in Settings to keep going without limits.",
                "progress": current_state["pct"],
                "isTestTrigger": False,
                "questions": []
            }), 429

        # 3. Detect task
        task = detect_task(user_prompt, bool(file_bytes), deep_solve)
        print(f"[nino-v3 dispatch] prompt='{user_prompt[:50]}' -> task='{task}'")

        # ── Route: TEST GENERATION ────────────────────────────────────
        if task == 'test':
            count, subject, hint = parse_test_request(user_prompt)
            title, secret_questions, err = build_test(count, subject, hint, api_key)
            if not secret_questions:
                err_msg = err or "Test generation was interrupted. Please try asking again."
                return jsonify({
                    "response": f"Nino couldn't generate the test right now: {err_msg}",
                    "progress": current_state["pct"],
                    "isTestTrigger": False,
                    "questions": []
                })

            test_id = store_test(secret_questions, ttl_seconds=TEST_TTL_SECONDS)
            public_questions = [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "options": q["options"],
                    "mistakeType": q.get("mistakeType", "Concept")
                }
                for q in secret_questions
            ]

            friendly_msg = f"Here is your {title} test set ({len(public_questions)} questions). Solve with precision before submitting."
            session_store.append_history(
                name, user_prompt, friendly_msg,
                test={"title": title, "questions": public_questions},
                route="test"
            )
            send_chat_log_to_discord(name, current_state["goal"], current_state["pct"], user_prompt, friendly_msg, "test")

            return jsonify({
                "chatResponse": friendly_msg,
                "response": friendly_msg,
                "isTestTrigger": True,
                "testTitle": title,
                "test_id": test_id,
                "questions": public_questions,
                "progress": current_state["pct"]
            })

        # ── Route: SOLVER / CHAT ──────────────────────────────────────
        history_msgs = build_model_history(current_state["history"])
        contents = [
            types.Content(
                role="model" if m.get("role") in ("assistant", "model") else "user",
                parts=[types.Part.from_text(text=m.get("content", ""))]
            )
            for m in history_msgs
        ]

        curr_parts = []
        if file_bytes and mime_type:
            curr_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
        curr_parts.append(types.Part.from_text(text=user_prompt))
        contents.append(types.Content(role="user", parts=curr_parts))

        raw_response, err = call_gemini(task, contents, api_key)
        if not raw_response:
            return jsonify({
                "response": f"Nino encountered an error: {err or 'Could not generate a response'}. Please try again.",
                "progress": current_state["pct"],
                "isTestTrigger": False,
                "questions": []
            })

        # Smart fallback: If chat model returns [TRIGGER_TEST: ...]
        test_trigger_match = re.search(r'\[TRIGGER_TEST:\s*([^\]]+)\]', raw_response)
        if test_trigger_match:
            test_topic = test_trigger_match.group(1).strip()
            clean_reply = raw_response.replace(test_trigger_match.group(0), "").strip()
            count, subject, hint = parse_test_request(test_topic or user_prompt)
            title, secret_questions, test_err = build_test(count, subject, hint, api_key)

            if secret_questions:
                test_id = store_test(secret_questions, ttl_seconds=TEST_TTL_SECONDS)
                public_questions = [
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "options": q["options"],
                        "mistakeType": q.get("mistakeType", "Concept")
                    }
                    for q in secret_questions
                ]
                final_chat_msg = clean_reply or f"Here is your {title} test set ({len(public_questions)} questions)."
                session_store.append_history(
                    name, user_prompt, final_chat_msg,
                    test={"title": title, "questions": public_questions},
                    route="test"
                )
                send_chat_log_to_discord(name, current_state["goal"], current_state["pct"], user_prompt, final_chat_msg, "test_fallback")

                return jsonify({
                    "chatResponse": final_chat_msg,
                    "response": final_chat_msg,
                    "isTestTrigger": True,
                    "testTitle": title,
                    "test_id": test_id,
                    "questions": public_questions,
                    "progress": current_state["pct"]
                })

        clean_response = raw_response
        if "[PROGRESS_UP]" in clean_response:
            session_store.advance_progress(name)
            clean_response = clean_response.replace("[PROGRESS_UP]", "").strip()

        session_store.append_history(name, user_prompt, clean_response, route=task)
        send_chat_log_to_discord(name, current_state["goal"], current_state["pct"], user_prompt, clean_response, task)

        return jsonify({
            "response": clean_response,
            "progress": current_state["pct"],
            "isTestTrigger": False,
            "testTitle": "",
            "questions": []
        })

    except Exception as e:
        print(f"[/guide Critical ERROR]: {e}")
        return jsonify({"response": "Nino encountered an unexpected system error. Please try again."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
