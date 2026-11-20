"""
app.py — Nino Production-Grade Multi-Model Backend.
"""

import mimetypes
import os
import re
import time
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, make_response, render_template, request
from google.genai import types

from prompts import (
    CONCEPT_CHAT_PROMPT,
    DOUBT_SOLVE_PROMPT,
    JEE_ADVANCED_PROMPT,
)
from router import (
    classify_intent,
    friendly_error,
    generate,
)
from store import (
    SessionManager,
    check_feedback_limit,
    check_rate_limit,
    delete_test,
    is_redis_enabled,
    retrieve_test,
    store_test,
)
from test_engine import (
    build_test_parallel,
    parse_test_request,
)

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4 MB (under Vercel's 4.5 MB body cap)
application = app

# ═══════════════════════════════════════════════════════════════════════
# 1. HELPERS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf'}
DAILY_SHARED_LIMIT = 15

PERSONA_MODIFIERS = {
    "strict": "\n\nMENTOR STYLE OVERRIDE: Be strict, direct, and rank-focused. Push accountability hard; zero fluff.",
    "supportive": "\n\nMENTOR STYLE OVERRIDE: Be extra warm, calm, and burnout-aware while keeping math strictly accurate.",
    "adaptive": "",
}

session_store = SessionManager()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_visitor_id():
    forwarded = request.headers.get("X-Forwarded-For")
    ip = forwarded.split(',')[0].strip() if forwarded else request.remote_addr
    return request.cookies.get("nino_vid") or ip or "unknown_visitor"


def skey(name):
    return f"{get_visitor_id()}::{name}"


def get_api_key():
    user_key = (request.headers.get("X-User-Key") or "").strip()
    if user_key:
        return user_key, False
    return os.environ.get("GEMINI_API_KEY"), True


def build_model_history(history, max_turns=8):
    recent = history[-max_turns:]
    out = []
    for m in recent:
        content = m.get("content", "")
        if len(content) > 2000:
            content = content[:2000] + "..."
        out.append({"role": m["role"], "content": content})
    return out


def _post_webhook(url, payload):
    try:
        requests.post(url, json=payload, timeout=2)
    except Exception as e:
        print(f"[Discord webhook ERROR]: {e}")


def send_chat_log_to_discord(name, goal, pct, user_msg, ai_reply, route):
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
    _post_webhook(url, {"content": content, "allowed_mentions": {"parse": []}})


@app.errorhandler(413)
def too_large(_e):
    return jsonify({
        "response": "That file is too large (max 4 MB). Compress or crop the image and try again.",
        "isTestTrigger": False,
        "questions": [],
    }), 413


# ═══════════════════════════════════════════════════════════════════════
# 2. ROUTES: STATIC & UTILITIES
# ═══════════════════════════════════════════════════════════════════════
@app.route('/')
def home():
    resp = make_response(render_template('index.html'))
    if not request.cookies.get("nino_vid"):
        resp.set_cookie(
            "nino_vid",
            os.urandom(12).hex(),
            max_age=60 * 60 * 24 * 365,
            samesite="Lax",
            secure=request.is_secure,
            httponly=True,
        )
    return resp


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "redis": is_redis_enabled(),
        "time": datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route('/api/profile', methods=['GET'])
def get_profile():
    display = request.args.get('name', 'Anonymous')
    session = session_store.get_or_create(skey(display))
    return jsonify({
        "name": display,
        "goal": session["goal"],
        "progress": session["pct"],
    })


@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        body = request.get_json(silent=True) or {}
        name = skey(body.get('name', 'Anonymous'))
        updated = session_store.update_settings(name, body.get('settings', {}))
        return jsonify({"status": "success", "settings": updated})

    name = skey(request.args.get('name', 'Anonymous'))
    session = session_store.get_or_create(name)
    return jsonify({"settings": session.get('settings', {"mode": "adaptive", "theme": "dark"})})


@app.route('/api/chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    return jsonify(session_store.get_chat(skey(chat_id)))


@app.route('/api/chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    deleted = session_store.delete_session(skey(chat_id))
    return jsonify({"status": "deleted" if deleted else "not_found"}), 200


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    vid = get_visitor_id()
    if not check_feedback_limit(vid):
        return jsonify({"status": "error", "error": "Rate limit exceeded"}), 429

    body = request.get_json(silent=True) or {}
    msg = str(body.get('message', '')).strip()[:1000]
    try:
        rating = max(0, min(5, int(body.get('rating', 0))))
    except (TypeError, ValueError):
        rating = 0

    url = os.environ.get("DISCORD_FEEDBACK_WEBHOOK_URL") or os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        stars = "★" * rating + "☆" * (5 - rating)
        _post_webhook(url, {"content": f"📝 Feedback [{body.get('category', 'other')} - {stars}]:\n{msg}"})
    return jsonify({"status": "ok"}), 200


# ═══════════════════════════════════════════════════════════════════════
# 3. ROUTE: SERVER-SIDE TEST GRADING
# ═══════════════════════════════════════════════════════════════════════
@app.route('/api/test/submit', methods=['POST'])
def submit_test():
    body = request.get_json(silent=True) or {}
    test_id = body.get('test_id')
    user_answers = body.get('answers') or {}

    if not test_id:
        return jsonify({"error": "Missing test_id"}), 400

    secret_questions = retrieve_test(test_id)
    if not secret_questions:
        return jsonify({"error": "Test session expired or not found. Please generate a fresh test."}), 404

    total = len(secret_questions)
    correct_count = 0
    mistake_counts = {"concept": 0, "formula": 0, "silly": 0}
    graded_questions = []

    for idx, q in enumerate(secret_questions, 1):
        q_id = str(q.get("id", idx))
        raw_sel = user_answers.get(q_id)
        try:
            selected = int(raw_sel) if raw_sel is not None else None
        except (TypeError, ValueError):
            selected = None

        is_correct = (selected is not None and selected == q["correct"])
        mt = q.get("mistakeType", "Concept")

        if is_correct:
            correct_count += 1
        elif selected is not None:
            mt_low = mt.lower()
            if mt_low == "concept":
                mistake_counts["concept"] += 1
            elif mt_low == "silly":
                mistake_counts["silly"] += 1
            else:
                mistake_counts["formula"] += 1

        graded_questions.append({
            "id": q.get("id", idx),
            "question": q["question"],
            "options": q["options"],
            "selected": selected,
            "correct": q["correct"],
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


# ═══════════════════════════════════════════════════════════════════════
# 4. ROUTE: MAIN ADVISORY & SOLVER (/guide)
# ═══════════════════════════════════════════════════════════════════════
TEST_TRIGGER_RE = re.compile(
    r"\b(mock\s*tests?|test\s+me|take\s+my\s+test|quiz(?:\s+me)?|practice\s+(?:test|questions?|set|paper)|"
    r"(?:give|make|create|generate|send|prepare|set|show)\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?(?:\d+\s+)?(?:[a-z]+\s+){0,4}"
    r"(?:tests?|questions?|ques|mcqs?|problems))\b",
    re.I,
)


@app.route('/guide', methods=['POST'])
def guide():
    t0 = time.perf_counter()
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            raw_name = request.form.get('name', '').strip()
            interest = request.form.get('interest', 'General Study Optimization').strip()
            followup = request.form.get('followup', '').strip()
            deep_solve = request.form.get('deep_solve', 'false').lower() == 'true'

            file = request.files.get('file')
            file_bytes = None
            mime_type = None
            if file and file.filename:
                if not allowed_file(file.filename):
                    return jsonify({
                        "response": "Unsupported file type. Please upload a PNG, JPG, WEBP, or PDF.",
                        "isTestTrigger": False,
                        "questions": [],
                    }), 400
                file_bytes = file.read()
                mime_type = file.content_type
                if not mime_type or mime_type == "application/octet-stream":
                    mime_type = mimetypes.guess_type(file.filename)[0] or "image/jpeg"
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

        api_key, is_owner_key = get_api_key()
        vid = get_visitor_id()
        if is_owner_key and not check_rate_limit(vid, DAILY_SHARED_LIMIT):
            return jsonify({
                "response": "You've used today's free messages. Add your personal Gemini key in Settings to keep going without limits.",
                "progress": current_state["pct"],
                "isTestTrigger": False,
                "questions": [],
            }), 429

        is_test_intent = bool(not file_bytes and TEST_TRIGGER_RE.search(user_prompt))
        previous_route = session_store.get_last_route(name)

        route_type = classify_intent(
            text=user_prompt,
            has_image=bool(file_bytes),
            is_test_endpoint=is_test_intent,
            deep_solve=deep_solve,
            previous_route=previous_route,
            api_key=api_key,
        )
        print(f"[intent] '{user_prompt[:50]}' -> {route_type}")

        # ── Route: TEST GENERATE ──────────────────────────────────────
        if route_type == "test_generate":
            count, subject, hint = parse_test_request(user_prompt)
            title, secret_questions, err = build_test_parallel(count, subject, hint, api_key, run_verifier=False)
            if not secret_questions:
                return jsonify({
                    "response": friendly_error(err or "Test generation was interrupted. Please ask again."),
                    "progress": current_state["pct"],
                    "isTestTrigger": False,
                    "questions": [],
                })

            test_id = store_test(secret_questions, ttl_seconds=7200)
            public_questions = [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "options": q["options"],
                    "mistakeType": q["mistakeType"],
                }
                for q in secret_questions
            ]

            friendly_msg = f"Here is your {title} test set ({len(public_questions)} questions). Solve with precision before submitting."
            session_store.append_history(
                name,
                user_prompt,
                friendly_msg,
                test={"title": title, "test_id": test_id, "questions": public_questions},
                route="test_generate",
            )
            send_chat_log_to_discord(name, current_state["goal"], current_state["pct"], user_prompt, friendly_msg, "test_generate")

            return jsonify({
                "chatResponse": friendly_msg,
                "isTestTrigger": True,
                "testTitle": title,
                "test_id": test_id,
                "questions": public_questions,
                "progress": current_state["pct"],
            })

        # ── Route: DOUBT SOLVE / JEE ADVANCED / CONCEPT CHAT ──────────
        sys_prompt = CONCEPT_CHAT_PROMPT
        if route_type == "jee_advanced":
            sys_prompt = JEE_ADVANCED_PROMPT
        elif route_type == "doubt_solve":
            sys_prompt = DOUBT_SOLVE_PROMPT

        mode = (current_state.get("settings") or {}).get("mode", "adaptive")
        sys_prompt = sys_prompt + PERSONA_MODIFIERS.get(mode, "")

        history_msgs = build_model_history(current_state["history"])
        contents = [
            types.Content(
                role="model" if m["role"] in ("assistant", "model") else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in history_msgs
        ]

        curr_parts = []
        if file_bytes and mime_type:
            curr_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
        curr_parts.append(types.Part.from_text(text=user_prompt))
        contents.append(types.Content(role="user", parts=curr_parts))

        raw_response, err = generate(
            route_type=route_type,
            contents=contents,
            system_instruction=sys_prompt,
            api_key=api_key,
        )
        print(f"[timing] {route_type} in {(time.perf_counter() - t0) * 1000:.0f}ms")

        if not raw_response:
            return jsonify({
                "response": friendly_error(err),
                "progress": current_state["pct"],
                "isTestTrigger": False,
                "questions": [],
            })

        clean_response = raw_response
        if "[PROGRESS_UP]" in clean_response:
            session_store.advance_progress(name)
            clean_response = clean_response.replace("[PROGRESS_UP]", "").strip()

        session_store.append_history(name, user_prompt, clean_response, route=route_type)
        send_chat_log_to_discord(name, current_state["goal"], current_state["pct"], user_prompt, clean_response, route_type)

        return jsonify({
            "response": clean_response,
            "progress": current_state["pct"],
            "isTestTrigger": False,
            "testTitle": "",
            "questions": [],
        })

    except Exception as e:
        print(f"[/guide Critical ERROR]: {e}")
        return jsonify({"response": "Nino encountered an unexpected system error. Please try again."}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
