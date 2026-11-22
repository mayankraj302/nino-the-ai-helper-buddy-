import json
import os
import random
import re
import math
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import requests
from flask import Flask, jsonify, make_response, render_template, request
from google import genai
from google.genai import types

app = Flask(__name__, template_folder="templates")
application = app  # Explicit alias for WSGI servers (AWS Elastic Beanstalk, etc.)

# ======================================================================
# TUNING KNOBS (speed vs. quality). Change these, not the code.
# ======================================================================
FAST_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]
REASONING_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

GEMINI_TIMEOUT_MS = 25000
TEST_BATCH_SIZE = 3
TEST_THINKING_BUDGET = 1024

THINKING = {
    "chat": (0, 700),
    "doubt": (512, 900),
    "solution": (1024, 2200),
    "visual": (768, 3000),
}

# ---------------- File validation ----------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- Usage limiting & keys (UNCHANGED) ----------------
DAILY_LIMIT = 15
_usage = {}


def get_visitor_id():
    return request.cookies.get("nino_vid") or request.remote_addr or "unknown"


def within_daily_limit(visitor_id):
    key = (visitor_id, date.today().isoformat())
    count = _usage.get(key, 0)
    if count >= DAILY_LIMIT:
        return False
    _usage[key] = count + 1
    return True


def get_api_key():
    user_key = (request.headers.get("X-User-Key") or "").strip()
    if user_key:
        return user_key, False
    return os.environ.get("GEMINI_API_KEY"), True


# ---------------- Background work ----------------
_bg_pool = ThreadPoolExecutor(max_workers=4)
_test_pool = ThreadPoolExecutor(max_workers=8)


def run_bg(fn, *args):
    try:
        _bg_pool.submit(fn, *args)
    except Exception as e:
        print(f"[bg ERROR]: {e}")


CUTOFF_MESSAGE = (
    "That test got cut off while I was generating it. Ask me again and I'll rebuild it."
)

# ======================================================================
# Rendering & LaTeX Leak Sanitizer
# ======================================================================
_DOUBLE_SLASH_RE = re.compile(r"\\\\(?=[a-zA-Z]+\vert{}[^a-zA-Z0-9\s])") _NAKED_LATEX_CMD_RE = re.compile( r"(?<!\$)(?<!\\)\\(?:frac|sqrt|theta|alpha|beta|gamma|delta|rightleftharpoons|"
    r"rightarrow|leftarrow|text|mathrm|mathbf|cdot|times|div|pm|mp|leq|geq|neq|"
    r"approx|sum|int|log|ln|sin|cos|tan)\b(?![^{}]*\})"
)


def normalize_latex_string(text):
    """
    Fixes over-escaped or leaked LaTeX sequences after JSON decoding so
    frontend math renderers (KaTeX / MathJax) receive clean single-backslash LaTeX.
    """
    if not isinstance(text, str) or not text:
        return text or ""

    # Preserve raw <svg>...</svg> blocks untouched
    if "<svg" in text.lower() and "</svg>" in text.lower():
        parts = re.split(r"(<svg\b.*?</svg>)", text, flags=re.DOTALL | re.IGNORECASE)
        return "".join(
            part if part.lower().startswith("<svg") else normalize_latex_string(part)
            for part in parts
        )

    # Collapse doubled backslashes inside JSON-decoded strings into single backslashes
    cleaned = _DOUBLE_SLASH_RE.sub(r"\\", text)

    # Wrap standalone un-delimited LaTeX commands in inline $...$ if needed
    def _wrap_cmd(match):
        token = match.group(0)
        return f"${token}$"      cleaned = _NAKED_LATEX_CMD_RE.sub(_wrap_cmd, cleaned)      # Collapse accidental double-dollar empty/broken pairs like $$  $$ cleaned = re.sub(r"\$\s*\$\$", "$$", cleaned)
    return cleaned.strip()


def looks_like_test_json(text):
    text = text or ""
    if re.match(r'\s*(?:```(?:json)?\s*)?\{\s*"chatResponse"', text, re.IGNORECASE):
        return True
    return bool(re.search(r'"questions"\s*:', text) and re.search(r'"options"\s*:', text))


def parse_ai_json(text):
    if not text:
        return None
    json_str = ""
    markdown_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if markdown_match:
        json_str = markdown_match.group(1)
    else:
        match = re.search(r'\{\s*"chatResponse"', text, re.IGNORECASE)
        if match:
            start_idx = match.start()
            end_idx = text.rfind("}")
            if end_idx > start_idx:
                json_str = text[start_idx : end_idx + 1]
        else:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx : end_idx + 1]

    if not json_str:
        return None
    try:
        parsed = json.loads(json_str)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    keys = {str(k).lower() for k in parsed}
    if "chatresponse" not in keys and "questions" not in keys:
        return None
    return parsed


def unpack_envelope(envelope):
    n = {str(k).lower(): v for k, v in envelope.items()}
    questions = n.get("questions") or []
    if not isinstance(questions, list):
        questions = []
    cleaned_questions = []
    for q in questions:
        cq = clean_question(q)
        if cq:
            cleaned_questions.append(cq)
    return {
        "chat_text": normalize_latex_string(n.get("chatresponse") or ""),
        "is_trigger": n.get("istesttrigger", True),
        "title": normalize_latex_string(n.get("testtitle") or "Evaluation Matrix"),
        "questions": cleaned_questions,
    }


# ---------------- Session storage ----------------
class SessionManager:
    def __init__(self, filepath="local_memory.json"):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._sessions = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[SessionManager ERROR - Load]: {e}")
        return {}

    def _save_memory(self):
        with self._lock:
            try:
                tmp = self.filepath + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._sessions, f, ensure_ascii=False)
                os.replace(tmp, self.filepath)
            except Exception as e:
                print(f"[SessionManager ERROR - Save]: {e}")

    def get_or_create(self, username, default_interest):
        if username not in self._sessions:
            self._sessions[username] = {
                "pct": 10,
                "goal": default_interest,
                "history": [],
                "settings": {"mode": "adaptive", "theme": "dark"},
            }
            self._save_memory()
        return self._sessions[username]

    def update_goal(self, username, interest):
        session = self.get_or_create(username, interest)
        if interest and interest != session["goal"]:
            session["goal"] = interest
            self._save_memory()
        return session

    def delete_session(self, username):
        if username in self._sessions:
            del self._sessions[username]
            self._save_memory()
            return True
        return False

    def migrate_legacy_history(self):
        changed = False
        for session in self._sessions.values():
            for msg in session.get("history", []):
                if msg.get("role") not in ("model", "assistant"):
                    continue
                if "test" in msg or "model_text" in msg:
                    continue
                content = msg.get("content") or ""
                envelope = parse_ai_json(content)
                if envelope is not None:
                    info = unpack_envelope(envelope)
                    has_test = len(info["questions"]) > 0
                    msg["content"] = info["chat_text"] or (
                        f"Here is your {info['title']} test." if has_test else content
                    )
                    if has_test:
                        msg["test"] = {
                            "title": info["title"],
                            "questions": info["questions"],
                        }
                    msg["model_text"] = json.dumps(envelope, ensure_ascii=False)
                    changed = True
                elif looks_like_test_json(content):
                    msg["content"] = CUTOFF_MESSAGE
                    changed = True
        if changed:
            self._save_memory()

    def advance_progress(self, username, step=12, maximum=100):
        if username in self._sessions:
            self._sessions[username]["pct"] = min(
                self._sessions[username]["pct"] + step, maximum
            )
            self._save_memory()

    def append_history(self, username, user_content, ai_content, test=None, model_text=None):
        if username in self._sessions:
            history = self._sessions[username]["history"]
            history.append({"role": "user", "content": user_content})
            ai_entry = {"role": "model", "content": normalize_latex_string(ai_content)}
            if test:
                ai_entry["test"] = test
            if model_text:
                ai_entry["model_text"] = model_text
            history.append(ai_entry)
            self._sessions[username]["history"] = history[-40:]
            self._save_memory()


session_store = SessionManager()
session_store.migrate_legacy_history()

# ======================================================================
# Gemini plumbing
# ======================================================================
_clients = {}
_clients_lock = threading.Lock()


def get_client(api_key):
    with _clients_lock:
        client = _clients.get(api_key)
        if client is None:
            if len(_clients) >= 50:
                _clients.pop(next(iter(_clients)))
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
                )
            except Exception:
                client = genai.Client(api_key=api_key)
            _clients[api_key] = client
        return client


def _make_config(
    system_instruction,
    temperature,
    max_tokens,
    thinking_budget,
    json_mode,
    use_thinking,
):
    kw = dict(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if json_mode:
        kw["response_mime_type"] = "application/json"
    if use_thinking and thinking_budget is not None:
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    return types.GenerateContentConfig(**kw)


def generate_text(
    contents,
    system_instruction,
    api_key,
    models=FAST_MODELS,
    temperature=0.7,
    max_tokens=1024,
    thinking_budget=None,
    json_mode=False,
):
    if not api_key:
        return None, "no_key"
    try:
        client = get_client(api_key)
    except Exception as e:
        print(f"[Gemini init ERROR]: {e}")
        return None, "init"

    last_error = None
    for model in models:
        for use_thinking in (True, False):
            if not use_thinking and thinking_budget is None:
                break
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=_make_config(
                        system_instruction,
                        temperature,
                        max_tokens,
                        thinking_budget,
                        json_mode,
                        use_thinking,
                    ),
                )
                text = getattr(response, "text", None)
                if text:
                    return text, None
                last_error = "empty_response"
                if use_thinking and thinking_budget is not None:
                    continue
                break
            except Exception as e:
                last_error = str(e)
                print(f"[Gemini ERROR - {model}]: {last_error[:300]}")
                low = last_error.lower()
                if use_thinking and thinking_budget is not None and (
                    "thinking" in low or "invalid_argument" in low or "400" in low
                ):
                    continue
                break
    return None, last_error


def friendly_error(err):
    if err == "no_key":
        return "Nino isn't configured yet. Please add a Gemini key in Settings or try later."
    if err == "init":
        return "Nino couldn't start the AI service. Please try again later."
    if err and ("429" in err or "quota" in err.lower()):
        return (
            "Nino is very busy right now (daily AI quota reached). "
            "Please try again later, or add your own Gemini key in Settings."
        )
    return "Nino couldn't answer that just now. Please try again in a moment."


# ======================================================================
# Prompts & Test Generation
# ======================================================================
SYSTEM_INSTRUCTIONS = r"""You are **Nino**, an AI companion built by **Mayank** to give IIT-JEE aspirants a free, judgment-free space to work through academic pressure, doubts and burnout, and also a sharp JEE tutor who solves doubts.
Always output valid, clean math expressions using `$...$` for inline math and `$$...$$` for display math, with single backslashes (e.g., `$\frac{mv}{qB}$`, `$\rightleftharpoons$`) and no raw unescaped control sequences.
"""

TEST_SYSTEM = r"""You are a JEE paper-setting committee member (JEE Advanced + tough JEE Main level).
Return ONLY valid JSON with keys: {"questions":[{"scratchpad":"...","question":"...","solution":"...","options":["...","...","...","..."],"correct":0,"mistakeType":"Concept"}]}.
Use valid single-backslash LaTeX inside `$...$` for all formulas and chemical/math expressions.
"""

TIER1 = {
    "Physics": [
        "Modern Physics",
        "Electrostatics",
        "Current Electricity",
        "Magnetism and EMI",
        "Rotational Mechanics",
        "SHM and Waves",
    ],
    "Chemistry": [
        "GOC and Reaction Mechanism",
        "Ionic and Chemical Equilibrium",
        "Coordination Chemistry",
        "Electrochemistry",
        "Mole Concept and Stoichiometry",
        "p-Block Elements",
    ],
    "Maths": [
        "Functions and Graphs",
        "Definite Integration and Area",
        "Coordinate Geometry (Circles and Conics)",
        "Probability and Permutations-Combinations",
        "Complex Numbers",
        "Application of Derivatives",
    ],
}

TIER2 = {
    "Physics": [
        "Ray and Wave Optics",
        "Thermodynamics and KTG",
        "Work Energy Power",
        "Centre of Mass and Momentum",
        "Gravitation",
    ],
    "Chemistry": [
        "Chemical Thermodynamics",
        "Chemical Kinetics",
        "Solutions",
        "Aldehydes, Ketones and Carboxylic Acids",
        "Atomic Structure",
    ],
    "Maths": [
        "Sequences and Series",
        "Matrices and Determinants",
        "Vectors and 3D Geometry",
        "Differential Equations",
        "Binomial Theorem",
    ],
}

SUBJECT_ORDER = ["Physics", "Chemistry", "Maths"]
TWISTS = [
    "a hidden constraint the student must derive before any formula applies",
    "a limiting or boundary case that decides which formula branch is valid",
    "two competing effects where the dominant one depends on the given parameters",
    "a symmetry that collapses a long calculation, which the student must spot",
    "a standard result used in a non-standard configuration",
    "case-based parameter dependence (the answer changes with the parameter range)",
    "a domain restriction or extraneous-solution trap",
    "two chapters linked through a shared quantity",
    "a multi-step deduction where the first step is easy to skip",
    "a common-sense-looking shortcut that is actually invalid here",
]

SUBJECT_WORDS = {
    "physics": "Physics",
    "phy": "Physics",
    "chemistry": "Chemistry",
    "chem": "Chemistry",
    "maths": "Maths",
    "math": "Maths",
    "mathematics": "Maths",
}

FILLER_WORDS = set(
    """generate give make create send prepare set me a an the some mock test tests quiz
    practice questions question ques mcq mcqs on for of in about jee main mains advanced
    tough hard difficult challenging my take please pls paper and""".split()
)

TEST_REQ_RE = re.compile(
    r"\b(mock\s*tests?|test\s+me|take\s+my\s+test|quiz(?:\s+me)?|practice\s+(?:test|questions?|set|paper)|"
    r"(?:give|make|create|generate|send|prepare|set)\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?(?:\d+\s+)?(?:[a-z]+\s+){0,3}"
    r"(?:tests?|questions?|ques|mcqs?))\b",
    re.I,
)
NOT_TEST_RE = re.compile(
    r"\b(hint|solve|solution|explain|doubt|stuck|help|why|wrong|mistake|diagnostic|analy[sz]e|review|score)\b",
    re.I,
)


def is_test_request(text):
    text = text or ""
    if len(text) > 300:
        return False
    return bool(TEST_REQ_RE.search(text)) and not NOT_TEST_RE.search(text)


def parse_test_request(text):
    low = (text or "").lower()
    m = re.search(r"\b(\d{1,2})\b", low)
    count = int(m.group(1)) if m else 5
    count = max(2, min(count, 10))
    subject, hint = None, []
    for w in re.findall(r"[a-z][a-z\-']*", low):
        if w in SUBJECT_WORDS:
            subject = subject or SUBJECT_WORDS[w]
        elif w not in FILLER_WORDS:
            hint.append(w)
    return count, subject, " ".join(hint)[:60]


def _ordered_topics(subject, n):
    t1, t2 = TIER1[subject][:], TIER2[subject][:]
    random.shuffle(t1)
    random.shuffle(t2)
    pool = t1 + t2
    return [pool[i % len(pool)] for i in range(n)]


def build_specs(count, subject, hint):
    twists = [TWISTS[i % len(TWISTS)] for i in range(count)]
    random.shuffle(twists)
    specs = []
    if hint:
        for j in range(count):
            topic = (
                hint
                if count == 1
                else f"{hint} (cover sub-area {j + 1} of {count} of this topic and stay clear of the other sub-areas)"
            )
            specs.append(
                {
                    "subject": subject or "identify from the topic",
                    "topic": topic,
                    "twist": twists[j],
                }
            )
    elif subject:
        for j, t in enumerate(_ordered_topics(subject, count)):
            specs.append({"subject": subject, "topic": t, "twist": twists[j]})
    else:
        base, extra = divmod(count, 3)
        j = 0
        for si, s in enumerate(SUBJECT_ORDER):
            n = base + (1 if si < extra else 0)
            for t in _ordered_topics(s, n):
                specs.append({"subject": s, "topic": t, "twist": twists[j]})
                j += 1
    return specs


def chunk_specs(specs, size=TEST_BATCH_SIZE):
    k = max(1, math.ceil(len(specs) / size))
    base, extra = divmod(len(specs), k)
    out, i = [], 0
    for b in range(k):
        n = base + (1 if b < extra else 0)
        out.append(specs[i : i + n])
        i += n
    return out


def test_title(subject, hint, specs):
    if hint:
        return f"{subject + ': ' if subject else ''}{hint.title()}"
    if subject:
        names = [s["topic"] for s in specs[:2]]
        return f"{subject}: " + " & ".join(names) + (" +more" if len(specs) > 2 else "")
    return "Mixed PCM Challenge"


_BACKSLASH = re.compile(r"\\\\|\\(?! [\"/]|u[0-9a-fA-F]{4})")


def fix_latex_backslashes(raw):
    return _BACKSLASH.sub(
        lambda m: m.group(0) if m.group(0) == "\\\\" else "\\\\", raw
    )


def parse_json_loose(text):
    if not text:
        return None
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return None
    body = t[s : e + 1]
    for cand in (fix_latex_backslashes(body), body):
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except (ValueError, TypeError):
            continue
    return None


_GIVEAWAY = re.compile(
    r"\b(all|none|neither|both)\b[^.]{0,12}\b(above|these|of)\b|\bnone of\b|\ball of\b",
    re.I,
)
_COT_LEAK_RE = re.compile(
    r"(?:^|[.?!]\s+)(?:wait\b|let\s+us\s+(?:frame|use|consider|check|reframe)|or\s+rather\b|actually\b|instead\s+let)",
    re.I,
)


def _norm_option(opt):
    s = re.sub(r"\s+", "", opt.strip().lower())
    return s.strip("$")


def clean_question(q):
    if not isinstance(q, dict):
        return None
    n = {str(k).lower(): v for k, v in q.items()}
    text = normalize_latex_string(str(n.get("question") or "").strip())
    opts = n.get("options")
    if len(text) < 25 or _COT_LEAK_RE.search(text):
        return None
    if not isinstance(opts, list) or len(opts) != 4:
        return None
    opts = [normalize_latex_string(str(o).strip()) for o in opts]
    if any(not o for o in opts) or len({_norm_option(o) for o in opts}) < 4:
        return None
    try:
        correct = int(n.get("correct"))
    except (TypeError, ValueError):
        return None
    if not 0 <= correct <= 3:
        return None
    mt = str(n.get("mistaketype") or "Concept").strip().capitalize()
    if mt not in ("Concept", "Formula", "Silly"):
        mt = "Concept"
    if not any(_GIVEAWAY.search(o) for o in opts):
        order = list(range(4))
        random.shuffle(order)
        opts = [opts[i] for i in order]
        correct = order.index(correct)
    return {"question": text, "options": opts, "correct": correct, "mistakeType": mt}


def generate_test_batch(specs, api_key):
    valid, err = [], None
    for _attempt in range(2):
        need = len(specs) - len(valid)
        if need <= 0:
            break
        todo = specs[len(valid) :] if valid else specs
        lines = "\n".join(
            f"{i}. Subject: {s['subject']} | Topic: {s['topic']} | Required twist: {s['twist']}"
            for i, s in enumerate(todo, 1)
        )
        prompt = (
            f"Write exactly {len(todo)} questions, one for each line below, in this order.\n"
            f"{lines}\n"
            f"Use the 'scratchpad' field to verify all physics/chemistry/math before writing 'question'.\n"
            f"Return only the JSON object."
        )
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        max_out = TEST_THINKING_BUDGET + 850 * len(todo) + 400
        text, err = generate_text(
            contents,
            TEST_SYSTEM,
            api_key,
            models=FAST_MODELS,
            temperature=0.55,
            max_tokens=max_out,
            thinking_budget=TEST_THINKING_BUDGET,
            json_mode=True,
        )
        data = parse_json_loose(text)
        qs = (data or {}).get("questions") or (data or {}).get("Questions") or []
        for q in qs if isinstance(qs, list) else []:
            cq = clean_question(q)
            if cq:
                valid.append(cq)
    valid = valid[: len(specs)]
    return valid, err


def build_test(count, subject, hint, api_key):
    specs = build_specs(count, subject, hint)
    batches = chunk_specs(specs)
    futures = [_test_pool.submit(generate_test_batch, b, api_key) for b in batches]
    questions, err = [], None
    for f in futures:
        try:
            qs, e = f.result(timeout=45)
            questions.extend(qs)
            err = err or e
        except Exception as e:
            err = err or str(e)
    seen, unique = set(), []
    for q in questions:
        key = re.sub(r"\W+", "", q["question"].lower())[:60]
        if key not in seen:
            seen.add(key)
            unique.append(q)
    for i, q in enumerate(unique, 1):
        q["id"] = i
    return test_title(subject, hint, specs), unique, err


# ======================================================================
# Chat call
# ======================================================================
FULL_SOLUTION_RE = re.compile(
    r"full solution|complete solution|entire solution|step[ -]by[ -]step|detailed solution",
    re.I,
)
VISUAL_RE = re.compile(r"visual|diagram|sketch|draw|svg", re.I)
DOUBT_RE = re.compile(
    r"solve|evaluate|integrat|differentiat|derivative|integral|limit|equation|find the|calculate|compute|prove|"
    r"value of|how many|which of the following|question|problem|numerical|mechanism|reaction|oxidation|"
    r"\bph\b|mole|\bforce\b|current|voltage|field|probability|matrix|sin|cos|tan|log|=|\^|\\|∫|√|\d\s*[+\-*/×]\s*\d",
    re.I,
)


def classify_chat_kind(prompt, has_file):
    if VISUAL_RE.search(prompt):
        return "visual"
    if FULL_SOLUTION_RE.search(prompt):
        return "solution"
    if has_file or DOUBT_RE.search(prompt):
        return "doubt"
    return "chat"


def build_model_history(history):
    recent = history[-12:]
    out = []
    for i, m in enumerate(recent):
        keep_full = i >= len(recent) - 4
        content = (m.get("model_text") if keep_full else None) or m["content"]
        out.append({"role": m["role"], "content": content})
    return out


def ask_ai(prompt, message_history, api_key, file_bytes=None, mime_type=None, kind="chat"):
    formatted_contents = []
    for msg in message_history:
        role = "model" if msg["role"] in ["assistant", "model"] else "user"
        formatted_contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
    current_parts = []
    if file_bytes and mime_type:
        current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
    current_parts.append(types.Part.from_text(text=prompt))
    formatted_contents.append(types.Content(role="user", parts=current_parts))
    think, visible = THINKING[kind]
    selected_models = (
        REASONING_MODELS
        if (kind in ("solution", "visual") or bool(file_bytes))
        else FAST_MODELS
    )
    return generate_text(
        formatted_contents,
        SYSTEM_INSTRUCTIONS,
        api_key,
        models=selected_models,
        temperature=0.7 if kind == "chat" else 0.3,
        max_tokens=think + visible,
        thinking_budget=think,
    )


# ======================================================================
# Discord & Routes
# ======================================================================
def _post_webhook(url, payload):
    try:
        requests.post(url, json=payload, timeout=6)
    except Exception as e:
        print(f"[Discord ERROR]: {e}")


def send_log_to_discord(name, user_goal, current_pct, user_asked, ai_answered):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return
    short_ai = ai_answered[:800] + "..." if len(ai_answered) > 800 else ai_answered
    content = (
        f"🚀 Mission Log\n"
        f"User: {name}\n"
        f"Track: {user_goal[:120]}\n"
        f"Progress: {current_pct}%\n\n"
        f"Asked:\n{user_asked[:500]}\n\n"
        f"Answered:\n{short_ai}"
    )[:1900]
    run_bg(
        _post_webhook,
        webhook_url,
        {"content": content, "allowed_mentions": {"parse": []}},
    )


FEEDBACK_LABELS = {
    "bug": ("🐞 Bug", 0xFF453A),
    "wrong_answer": ("❌ Wrong answer", 0xFF9F0A),
    "idea": ("💡 Idea", 0x30D158),
    "other": ("💬 Other", 0x0A84FF),
}
_feedback_usage = {}
FEEDBACK_PER_HOUR = 10


def send_feedback_to_discord(fb):
    url = os.environ.get("DISCORD_FEEDBACK_WEBHOOK_URL") or os.environ.get(
        "DISCORD_WEBHOOK_URL"
    )
    if not url:
        return
    label, color = FEEDBACK_LABELS[fb["category"]]
    stars = ("★" * fb["rating"] + "☆" * (5 - fb["rating"])) if fb["rating"] else "No rating"
    fields = [
        {"name": "Rating", "value": stars, "inline": True},
        {"name": "Chat id", "value": fb["name"][:60] or "n/a", "inline": True},
        {"name": "Device", "value": f"{fb['screen']} · {fb['theme']}"[:60], "inline": True},
    ]
    if fb["context"]:
        fields.append(
            {"name": "About this answer", "value": fb["context"][:900], "inline": False}
        )
    payload = {
        "content": f"📝 New feedback · {label} · {stars}",
        "embeds": [
            {
                "title": label,
                "description": (fb["message"] or "(rating only)")[:1800],
                "color": color,
                "fields": fields,
                "timestamp": fb["time"],
            }
        ],
        "allowed_mentions": {"parse": []},
    }
    run_bg(_post_webhook, url, payload)


@app.route("/")
def home():
    resp = make_response(render_template("index.html"))
    if not request.cookies.get("nino_vid"):
        resp.set_cookie(
            "nino_vid",
            os.urandom(12).hex(),
            max_age=60 * 60 * 24 * 365,
            samesite="Lax",
            secure=True,
            httponly=True,
        )
    return resp


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/feedback", methods=["POST"])
def feedback():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message") or "").strip()[:1500]
    try:
        rating = max(0, min(5, int(body.get("rating") or 0)))
    except (TypeError, ValueError):
        rating = 0
    if not message and not rating:
        return jsonify({"status": "error", "error": "Empty feedback"}), 400
    bucket = (get_visitor_id(), datetime.now(timezone.utc).strftime("%Y%m%d%H"))
    if _feedback_usage.get(bucket, 0) >= FEEDBACK_PER_HOUR:
        return jsonify(
            {"status": "error", "error": "Too many feedback messages, try later"}
        ), 429
    _feedback_usage[bucket] = _feedback_usage.get(bucket, 0) + 1
    category = body.get("category")
    fb = {
        "category": category if category in FEEDBACK_LABELS else "other",
        "rating": rating,
        "message": message,
        "context": str(body.get("context") or "").strip()[:900],
        "name": str(body.get("name") or "Anonymous")[:60],
        "screen": str(body.get("screen") or "?")[:20],
        "theme": str(body.get("theme") or "?")[:10],
        "time": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(fb, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[feedback log ERROR]: {e}")
    send_feedback_to_discord(fb)
    return jsonify({"status": "ok"}), 200


@app.route("/api/profile", methods=["GET"])
def get_profile():
    name = request.args.get("name", "Anonymous")
    session = session_store.get_or_create(name, "General Study Optimization")
    return jsonify(
        {"name": name, "goal": session["goal"], "progress": session["pct"]}
    )


@app.route("/api/settings", methods=["GET", "POST"])
def handle_settings():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        name = body.get("name", "Anonymous")
        new_settings = body.get("settings", {})
        session = session_store.get_or_create(name, "General Study Optimization")
        if not isinstance(new_settings, dict):
            new_settings = {}
        session.setdefault("settings", {}).update(new_settings)
        session_store._save_memory()
        return jsonify({"status": "success", "settings": session["settings"]})
    name = request.args.get("name", "Anonymous")
    session = session_store.get_or_create(name, "General Study Optimization")
    return jsonify(
        {"settings": session.get("settings", {"mode": "adaptive", "theme": "dark"})}
    )


@app.route("/api/chat/<chat_id>", methods=["GET"])
def get_chat_history(chat_id):
    session = session_store._sessions.get(chat_id, {})
    return jsonify(
        {
            "history": [
                {k: v for k, v in m.items() if k != "model_text"}
                for m in session.get("history", [])
            ],
            "goal": session.get("goal", "General Study Optimization"),
            "pct": session.get("pct", 10),
        }
    )


@app.route("/api/chat/<chat_id>", methods=["DELETE"], endpoint="delete_chat")
def delete_chat(chat_id):
    deleted = session_store.delete_session(chat_id)
    return jsonify({"status": "deleted" if deleted else "not_found"}), 200


@app.route("/run_sandbox", methods=["POST"])
def run_sandbox():
    if os.environ.get("ENABLE_SANDBOX") != "1":
        return jsonify({"error": "Sandbox is disabled"}), 403
    data = request.get_json(silent=True) or {}
    code_to_run = data.get("code", "")
    if not code_to_run:
        return jsonify({"error": "No code provided"}), 400
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
        temp_file.write(code_to_run)
        temp_path = temp_file.name
    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            "128m",
            "-v",
            f"{temp_path}:/app/script.py:ro",
            "python:3.10-slim",
            "python",
            "/app/script.py",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout if result.returncode == 0 else result.stderr
        return jsonify(
            {"output": output, "status": "success" if result.returncode == 0 else "error"}
        )
    except subprocess.TimeoutExpired:
        return jsonify({"output": "Execution timed out (10s limit).", "status": "error"})
    except Exception as e:
        return jsonify({"output": str(e), "status": "error"})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/guide", methods=["POST"])
def guide():
    t0 = time.perf_counter()
    try:
        if request.content_type and "multipart/form-data" in request.content_type:
            raw_name = request.form.get("name", "").strip()
            interest = request.form.get("interest", "General Optimization Strategy").strip()
            followup = request.form.get("followup", "").strip()
            file = request.files.get("file")
            file_bytes = None
            mime_type = None
            if file and file.filename != "" and allowed_file(file.filename):
                file_bytes = file.read()
                mime_type = file.content_type
            if not followup:
                followup = "solve this question"
        else:
            data = request.get_json(silent=True) or {}
            raw_name = (data.get("name") or "").strip()
            interest = (data.get("interest") or "General Optimization Strategy").strip()
            followup = (data.get("followup") or "").strip()
            file_bytes = None
            mime_type = None

        name = raw_name if raw_name else "Anonymous"
        current_state = session_store.update_goal(name, interest)
        user_prompt = followup if followup else f"Guide me for {current_state['goal']}"
        api_key, is_owner_key = get_api_key()
        if is_owner_key and not within_daily_limit(get_visitor_id()):
            return (
                jsonify(
                    {
                        "response": "You've used today's free messages. Come back tomorrow, or add your own Gemini key in Settings to keep going.",
                        "progress": 0,
                        "isTestTrigger": False,
                        "testTitle": "",
                        "questions": [],
                    }
                ),
                429,
            )

        updated_pct = current_state["pct"]

        if not file_bytes and is_test_request(user_prompt):
            count, subject, hint = parse_test_request(user_prompt)
            title, questions, err = build_test(count, subject, hint, api_key)
            print(
                f"[timing] test x{count} -> {len(questions)} ok in {(time.perf_counter() - t0) * 1000:.0f}ms"
            )
            if not questions:
                msg = CUTOFF_MESSAGE if not err or err == "empty_response" else friendly_error(err)
                return jsonify(
                    {
                        "response": msg,
                        "progress": updated_pct,
                        "isTestTrigger": False,
                        "testTitle": "",
                        "questions": [],
                    }
                )
            friendly = (
                f"Here's your {title} set. These are built to be tough, so think before you tap."
            )
            session_store.append_history(
                name,
                user_prompt,
                friendly,
                test={"title": title, "questions": questions},
                model_text=json.dumps(
                    {"chatResponse": friendly, "testTitle": title, "questions": questions},
                    ensure_ascii=False,
                ),
            )
            send_log_to_discord(
                name,
                current_state["goal"],
                updated_pct,
                user_prompt,
                f"[Test: {title} - {len(questions)} questions] {friendly}",
            )
            return jsonify(
                {
                    "chatResponse": friendly,
                    "isTestTrigger": True,
                    "testTitle": title,
                    "questions": questions,
                    "progress": updated_pct,
                }
            )

        kind = classify_chat_kind(user_prompt, bool(file_bytes))
        raw_response, err = ask_ai(
            prompt=user_prompt,
            message_history=build_model_history(current_state["history"]),
            api_key=api_key,
            file_bytes=file_bytes,
            mime_type=mime_type,
            kind=kind,
        )
        print(f"[timing] {kind} in {(time.perf_counter() - t0) * 1000:.0f}ms")
        if not raw_response:
            return jsonify(
                {
                    "response": friendly_error(err),
                    "progress": updated_pct,
                    "isTestTrigger": False,
                    "testTitle": "",
                    "questions": [],
                }
            )

        clean_response = raw_response
        if "[PROGRESS_UP]" in clean_response:
            session_store.advance_progress(name)
            clean_response = clean_response.replace("[PROGRESS_UP]", "").strip()

        envelope = parse_ai_json(clean_response)
        if envelope is not None:
            info = unpack_envelope(envelope)
            has_test = len(info["questions"]) > 0
            friendly = info["chat_text"] or (
                f"Here is your {info['title']} test."
                if has_test
                else normalize_latex_string(clean_response)
            )
            session_store.append_history(
                name,
                user_prompt,
                friendly,
                test={"title": info["title"], "questions": info["questions"]}
                if has_test
                else None,
                model_text=json.dumps(envelope, ensure_ascii=False),
            )
            log_text = (
                f"[Test: {info['title']} - {len(info['questions'])} questions] {friendly}"
                if has_test
                else friendly
            )
            send_log_to_discord(
                name, current_state["goal"], updated_pct, user_prompt, log_text
            )
            return jsonify(
                {
                    "chatResponse": friendly,
                    "isTestTrigger": info["is_trigger"],
                    "testTitle": info["title"],
                    "questions": info["questions"],
                    "progress": updated_pct,
                }
            )

        if looks_like_test_json(clean_response):
            session_store.append_history(name, user_prompt, CUTOFF_MESSAGE)
            send_log_to_discord(
                name,
                current_state["goal"],
                updated_pct,
                user_prompt,
                "[Broken test JSON] " + clean_response,
            )
            return jsonify(
                {
                    "response": CUTOFF_MESSAGE,
                    "progress": updated_pct,
                    "isTestTrigger": False,
                    "testTitle": "",
                    "questions": [],
                }
            )

        sanitized_response = normalize_latex_string(clean_response)
        session_store.append_history(name, user_prompt, sanitized_response)
        send_log_to_discord(
            name, current_state["goal"], updated_pct, user_prompt, sanitized_response
        )
        return jsonify(
            {
                "response": sanitized_response,
                "progress": updated_pct,
                "isTestTrigger": False,
                "testTitle": "",
                "questions": [],
            }
        )
    except Exception as e:
        print(f"[/guide ERROR]: {e}")
        return jsonify(
            {"response": "Nino hit an unexpected error. Please try again."}
        ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
