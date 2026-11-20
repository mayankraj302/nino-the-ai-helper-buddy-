"""
engine.py — Unified Multi-Model Engine & Intent Router for Nino v3.

CRITICAL ARCHITECTURE:
- Leaf-adjacent engine: imports ONLY standard libraries and prompts.
- 3 dedicated roles: 'chat', 'solver', and 'test'.
- Auto-healing developer instructions, 429 backoff, and 503/504/404 cascade fallback.
- Thinking level safety: 'minimal' strictly restricted to models with 'flash-lite'.
- Bulletproof intent detection: never slips on wording.
- Solve-first structured test generation with option realignment and fast parallel batches.
"""

import json
import random
import re
import threading
import time

from google import genai
from google.genai import types

try:
    from prompts import CHAT_PROMPT, SOLVER_PROMPT, TEST_PROMPT, CHAPTER_FOCUS
except ImportError:
    try:
        from prompts import (
            CONCEPT_CHAT_PROMPT as CHAT_PROMPT,
            DOUBT_SOLVE_PROMPT as SOLVER_PROMPT,
            TEST_GENERATE_PROMPT as TEST_PROMPT,
            CHAPTER_FOCUS,
        )
    except Exception:
        CHAT_PROMPT = "You are Nino, a warm, sharp IIT-JEE mentor and companion."
        SOLVER_PROMPT = "You are Nino, veteran JEE Advanced faculty who solves doubts rigorously."
        TEST_PROMPT = "You are the chief JEE Main paper-setter. Write ORIGINAL single-correct MCQs."
        CHAPTER_FOCUS = {}

# ═══════════════════════════════════════════════════════════════════════
# 1. ROLES, MODELS & CASCADES
# ═══════════════════════════════════════════════════════════════════════
ROLES_CONFIG = {
    "chat": {
        "primary": {"model": "gemini-3.8-flash", "thinking": "low", "code_exec": False},
        "fallbacks": [
            {"model": "gemini-3.5-flash-lite", "thinking": "minimal", "code_exec": False},
            {"model": "gemini-2.5-flash", "thinking": "low", "code_exec": False},
        ],
        "default_tokens": 1500,
        "temperature": 0.7,
        "system_prompt": CHAT_PROMPT,
    },
    "solver": {
        "primary": {"model": "gemini-4-hydrogen", "thinking": "high", "code_exec": True},
        "fallbacks": [
            {"model": "gemini-3.8-flash", "thinking": "high", "code_exec": True},
            {"model": "gemini-2.5-pro", "thinking": "high", "code_exec": True},
            {"model": "gemini-2.5-flash", "thinking": "high", "code_exec": True},
        ],
        "default_tokens": 4000,
        "temperature": 0.2,
        "system_prompt": SOLVER_PROMPT,
    },
    "test": {
        "primary": {"model": "gemini-3.8-flash", "thinking": "high", "code_exec": False},
        "fallbacks": [
            {"model": "gemini-4-hydrogen", "thinking": "high", "code_exec": False},
            {"model": "gemini-2.5-flash", "thinking": "low", "code_exec": False},
        ],
        "default_tokens": 3000,
        "temperature": 0.3,
        "system_prompt": TEST_PROMPT,
    },
}

_LEVEL_BUDGETS = {
    "minimal": 128,
    "low": 512,
    "medium": 1024,
    "high": 2048,
    "xhigh": 4096,
}

# ═══════════════════════════════════════════════════════════════════════
# 2. CLIENT CACHING & CONFIG BUILDERS
# ═══════════════════════════════════════════════════════════════════════
_clients = {}
_clients_lock = threading.Lock()
DEFAULT_TIMEOUT_MS = 60_000

def _get_client(api_key: str, timeout_ms: int = DEFAULT_TIMEOUT_MS):
    cache_key = (api_key, timeout_ms)
    with _clients_lock:
        client = _clients.get(cache_key)
        if client is None:
            if len(_clients) >= 50:
                _clients.pop(next(iter(_clients)))
            try:
                client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
            except Exception:
                client = genai.Client(api_key=api_key)
            _clients[cache_key] = client
        return client

def _build_thinking_config(level: str | None, model_name: str):
    """
    Builds ThinkingConfig safely.
    'minimal' is ONLY passed to models containing 'flash-lite' (otherwise mapped to 'low')
    to prevent API 400 validation errors.
    Falls back from thinking_level to thinking_budget if needed.
    """
    if not level:
        return None
    model_lower = (model_name or "").lower()
    if "2.0" in model_lower:
        return None

    clean_level = str(level).strip().lower()
    if clean_level == "minimal" and "flash-lite" not in model_lower:
        clean_level = "low"

    try:
        return types.ThinkingConfig(thinking_level=clean_level.upper())
    except Exception:
        pass

    budget = _LEVEL_BUDGETS.get(clean_level, 1024)
    try:
        return types.ThinkingConfig(thinking_budget=budget)
    except Exception:
        return None

def _get_tools(code_exec: bool):
    """Safely construct code execution tools or return None."""
    if not code_exec:
        return None
    for cls_name in ("ToolCodeExecution", "CodeExecution"):
        cls = getattr(types, cls_name, None)
        if cls is not None:
            try:
                return [types.Tool(code_execution=cls())]
            except Exception:
                pass
    try:
        return [types.Tool(code_execution={})]
    except Exception:
        pass
    return None

def _prepare_prefixed_contents(contents, system_instruction: str):
    """Prepend system prompt into user contents when developer instruction is rejected."""
    if not system_instruction:
        return contents
    prefix_text = f"[System Instructions: {system_instruction}]\n\n"
    if isinstance(contents, str):
        return prefix_text + contents
    if isinstance(contents, list):
        if not contents:
            return [prefix_text]
        first = contents[0]
        if isinstance(first, str):
            return [prefix_text + first] + contents[1:]
        elif isinstance(first, types.Content):
            new_part = types.Part.from_text(text=prefix_text)
            new_content = types.Content(role=first.role or "user", parts=[new_part] + list(first.parts))
            return [new_content] + contents[1:]
        elif hasattr(first, "parts"):
            new_part = types.Part.from_text(text=prefix_text)
            new_content = types.Content(role="user", parts=[new_part] + list(first.parts))
            return [new_content] + contents[1:]
        else:
            return [prefix_text] + list(contents)
    return contents

# ═══════════════════════════════════════════════════════════════════════
# 3. ERROR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════
def _is_fatal(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in ("401", "403", "api_key_invalid", "permission_denied", "unauthenticated", "invalid api key"))

def _is_rate_limit(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in ("429", "quota", "resource_exhausted", "rate limit"))

def _is_cascade_needed(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in (
        "503", "unavailable", "high demand", "overloaded",
        "504", "timeout", "deadline_exceeded", "timed out",
        "404", "not_found", "not found"
    ))

def friendly_error(err: str) -> str:
    """Returns a user-friendly error string for common API failure modes."""
    if not err:
        return "An unknown error occurred. Please try sending your message again."
    err_str = str(err)
    if err_str == "no_key":
        return "Nino isn't configured yet. Please add a Gemini key in Settings or try later."
    if _is_fatal(err_str):
        return "That Gemini API key looks invalid or lacks permission. Please check it in Settings."
    if _is_rate_limit(err_str):
        return "Nino is receiving heavy traffic (daily AI quota reached). Please wait a moment or add your own key in Settings."
    if _is_cascade_needed(err_str):
        return "Google's preview model is currently overloaded. Nino is rerouting your request, please tap send again!"
    return f"Nino encountered an error: {err_str[:80]}. Please try sending your message again."

# ═══════════════════════════════════════════════════════════════════════
# 4. SINGLE MODEL EXECUTION WITH DEV-INSTRUCTION AUTO-HEALING
# ═══════════════════════════════════════════════════════════════════════
def _invoke_model_attempt(client, model_id: str, contents, system_inst: str,
                          thinking_level: str | None, code_exec: bool,
                          max_tokens: int, temp: float, json_schema=None, json_mode=False):
    """
    Attempts model execution with multi-stage developer instruction auto-healing:
      1) Plain string system_instruction
      2) Structured types.Content(parts=[types.Part.from_text(...)])
      3) User message prefixing
    """
    modes = []
    if system_inst:
        modes = [
            ("string", system_inst),
            ("content", types.Content(parts=[types.Part.from_text(text=system_inst)])),
            ("prefixed", None)
        ]
    else:
        modes = [("none", None)]

    for mode, sys_param in modes:
        eff_contents = contents
        if mode == "prefixed" and system_inst:
            eff_contents = _prepare_prefixed_contents(contents, system_inst)

        thinking_opts = (thinking_level, None) if thinking_level else (None,)
        tools_opts = (code_exec, False) if code_exec else (False,)

        dev_err = False
        for t_opt in thinking_opts:
            for c_opt in tools_opts:
                kw = {
                    "temperature": temp,
                    "max_output_tokens": max_tokens,
                }
                if sys_param is not None:
                    kw["system_instruction"] = sys_param

                tc = _build_thinking_config(t_opt, model_id)
                if tc:
                    kw["thinking_config"] = tc

                tools = _get_tools(c_opt)
                if tools:
                    kw["tools"] = tools

                if json_schema is not None:
                    kw["response_mime_type"] = "application/json"
                    kw["response_schema"] = json_schema
                elif json_mode:
                    kw["response_mime_type"] = "application/json"

                try:
                    config = types.GenerateContentConfig(**kw)
                    resp = client.models.generate_content(
                        model=model_id,
                        contents=eff_contents,
                        config=config
                    )
                    text = getattr(resp, "text", None)
                    if text:
                        return text, None
                except Exception as e:
                    err_str = str(e)
                    err_low = err_str.lower()

                    if _is_fatal(err_str) or _is_cascade_needed(err_str) or _is_rate_limit(err_str):
                        return None, err_str

                    if "developer instruction" in err_low or "developer_instruction" in err_low:
                        dev_err = True
                        break

                    if "thinking" in err_low or "tool" in err_low or "invalid_argument" in err_low:
                        continue

                    return None, err_str

            if dev_err:
                break

    return None, "empty_response"

# ═══════════════════════════════════════════════════════════════════════
# 5. PUBLIC CALL_GEMINI WITH RETRY & FALLBACK
# ═══════════════════════════════════════════════════════════════════════
def call_gemini(role: str, contents, api_key: str, json_schema=None, json_mode: bool = False,
                system_instruction: str = None) -> tuple[str | None, str | None]:
    """
    Safe Gemini API caller with automatic retry and model fallback.
    - Catches 429 and does short exponential backoff with jitter.
    - If primary fails (503, 404, 429), automatically tries the fallback model.
    - Auto-heals developer instructions.
    Returns: (text, error)
    """
    if not api_key:
        return None, "no_key"

    role_key = (role or "chat").lower()
    config = ROLES_CONFIG.get(role_key, ROLES_CONFIG["chat"])
    models_to_try = [config["primary"]] + config["fallbacks"]
    effective_system = system_instruction if system_instruction is not None else config.get("system_prompt", "")

    try:
        client = _get_client(api_key)
    except Exception as exc:
        return None, f"client_init: {exc}"

    last_error = None
    for model_cfg in models_to_try:
        model_id = model_cfg["model"]
        thinking = model_cfg.get("thinking")
        code_exec = model_cfg.get("code_exec", False)
        max_tokens = config.get("default_tokens", 2000)
        temp = config.get("temperature", 0.5)

        max_retries = 2
        for attempt in range(max_retries + 1):
            text, err = _invoke_model_attempt(
                client=client,
                model_id=model_id,
                contents=contents,
                system_inst=effective_system,
                thinking_level=thinking,
                code_exec=code_exec,
                max_tokens=max_tokens,
                temp=temp,
                json_schema=json_schema,
                json_mode=json_mode,
            )
            if text:
                return text, None

            last_error = err
            if _is_fatal(err):
                return None, err

            if _is_rate_limit(err):
                if attempt < max_retries:
                    delay = 1.5 * (1.3 ** attempt) + random.uniform(0.1, 0.8)
                    time.sleep(delay)
                    continue
                else:
                    break

            if _is_cascade_needed(err):
                break

            break

    return None, last_error

# ═══════════════════════════════════════════════════════════════════════
# 6. BULLETPROOF TRIGGER DETECTION
# ═══════════════════════════════════════════════════════════════════════
_TEST_ACTIONS = {
    'generate', 'give', 'make', 'create', 'set', 'curate', 'prepare',
    'test', 'quiz', 'drill', 'practice', 'draft', 'build', 'conduct',
    'start', 'take', 'design', 'frame', 'formulate', 'send', 'ask'
}

_TEST_TARGETS = {
    'test', 'tests', 'mock', 'mocks', 'quiz', 'quizzes', 'paper', 'papers',
    'question', 'questions', 'ques', 'mcq', 'mcqs', 'problem', 'problems',
    'dpp', 'exercise', 'exercises'
}

_TEST_NEGATIVE_GUARDS = [
    r'\bsolve\b', r'\bsolving\b', r'\bsolution\b', r'\bexplain\s+why\b',
    r'\bexplain\s+this\b', r'\bhint\b', r'\bstuck\b', r'\bdoubt\b',
    r'\bdoubts\b', r'\bwhy\s+does\b', r'\bhow\s+to\s+solve\b', r'\bcheck\s+my\b',
    r'\bverify\b', r'\bmy\s+mistake\b', r'\bwhat\s+did\s+i\s+do\s+wrong\b',
    r'\bwhere\s+am\s+i\s+wrong\b', r'\bhelp\s+me\s+solve\b'
]

_SOLVER_KEYWORDS = [
    r'\bsolve\b', r'\bevaluate\b', r'\bcalculate\b', r'\bcompute\b',
    r'\bderivat(?:ive|ion)?\b', r'\bdifferentiat(?:e|ion)?\b',
    r'\bintegrat(?:e|ion|al)?\b', r'\blimits?\b', r'\bequation\b',
    r'\bfind\s+(?:the\s+)?(?:value|ratio|roots?|maximum|minimum|area|magnitude|force|velocity|acceleration|current|charge|potential)\b',
    r'\bprove\b', r'\bmechanism\b', r'\breaction\b', r'\boxidation\b',
    r'\bph\b', r'\bmole\b', r'\bvoltage\b', r'\bprobability\b', r'\bmatrix\b',
    r'\bdeterminant\b', r'\bvector\b', r'\bkinematics\b', r'\btorque\b',
    r'\bmomentum\b', r'\bcapacit(?:or|ance)\b', r'\bresistan(?:ce|t)\b',
    r'\bthermodynamics\b', r'\bentropy\b', r'\benthalpy\b', r'\bequilibrium\b'
]

_MATH_SYMBOLS = [
    '∫', '√', '∑', '∏', '∂', '∆', '≠', '≤', '≥', '±', '×', '÷',
    r'\int', r'\frac', r'\sqrt', r'\sum', r'\lim', r'\partial'
]

def detect_task(text: str, has_photo: bool = False, deep_solve: bool = False) -> str:
    """
    Bulletproof intent router:
      a. Photo attached OR deep_solve toggle active -> 'solver'
      b. Action + Target detection for tests with negative guards -> 'test'
      c. Math symbols / equations / solve words -> 'solver'
      d. Default -> 'chat'
    Returns: 'test' | 'solver' | 'chat'
    """
    # a. Photo attached OR deep_solve toggle active -> 'solver'
    if has_photo or deep_solve:
        return 'solver'

    clean_text = (text or "").strip()
    if not clean_text:
        return 'chat'

    low = clean_text.lower()
    tokens = set(re.findall(r'[a-z]+', low))

    # b. Action + Target detection for tests WITH negative guards
    has_neg_guard = any(re.search(pat, low) for pat in _TEST_NEGATIVE_GUARDS)
    if not has_neg_guard:
        has_action = bool(tokens & _TEST_ACTIONS)
        has_target = bool(tokens & _TEST_TARGETS)

        standalone = bool(
            re.search(r'\b(mock\s+tests?|practice\s+tests?|jee\s+mock|take\s+(?:a\s+)?test|quiz\s+me)\b', low) or
            re.search(r'\b\d+\s*(?:mcqs?|questions?|problems?|ques)\b', low)
        )

        if (has_action and has_target) or standalone:
            return 'test'

    # c. Math symbols / equations / solve words
    for sym in _MATH_SYMBOLS:
        if sym in clean_text:
            return 'solver'

    if any(re.search(pat, low) for pat in _SOLVER_KEYWORDS):
        return 'solver'

    if re.search(r'\d\s*[+\-*/×=^]\s*\d', low) or re.search(r'[a-z]\s*=\s*[-+]?\d', low):
        return 'solver'

    # d. Default -> 'chat'
    return 'chat'

# ═══════════════════════════════════════════════════════════════════════
# 7. CHAPTER TIERS, TWISTS & FORMAT SPECIFICATIONS FOR TESTS
# ═══════════════════════════════════════════════════════════════════════
TIER1 = {
    "Physics": [
        "Kinematics and Projectile Motion", "Laws of Motion and Friction", "SHM and Waves",
        "Current Electricity", "Electrostatics and Capacitors", "Magnetism and EMI"
    ],
    "Chemistry": [
        "Ionic and Chemical Equilibrium", "Coordination Chemistry and d-f Block",
        "Electrochemistry and Redox", "GOC and Reaction Mechanism",
        "Mole Concept and Stoichiometry", "Chemical Bonding and Molecular Structure"
    ],
    "Maths": [
        "Sets, Relations and Functions", "Limits, Continuity and Application of Derivatives",
        "Definite Integration and Area", "Coordinate Geometry (Circles and Conics)",
        "Vectors and 3D Geometry", "Trigonometry and Inverse Trigonometric Functions"
    ],
}

TIER2 = {
    "Physics": [
        "Work Energy Power and Centre of Mass", "Rotational Mechanics", "Modern Physics",
        "Ray and Wave Optics", "Thermodynamics and KTG"
    ],
    "Chemistry": [
        "Chemical Thermodynamics", "Chemical Kinetics", "Solutions and Colligative Properties",
        "p-Block and s-Block Elements", "Aldehydes, Ketones, Carboxylic Acids and Biomolecules"
    ],
    "Maths": [
        "Probability, Statistics and Permutations-Combinations", "Complex Numbers and Quadratic Equations",
        "Differential Equations", "Matrices and Determinants", "Sequences, Series and Binomial Theorem"
    ],
}

SUBJECT_ORDER = ["Physics", "Chemistry", "Maths"]

TWISTS = [
    "a hidden constraint the student must derive before any formula applies",
    "a limiting or boundary case that decides which formula branch is valid",
    "two competing effects where the dominant one depends on the given parameters",
    "a symmetry that collapses a long calculation, which the student must spot",
    "a standard result used in a non-standard configuration",
    "case-based parameter dependence (the answer changes with parameter range)",
    "a domain restriction or extraneous-solution trap",
    "an idea from a second chapter that enters through a shared quantity",
    "a multi-step deduction where the first step is easy to skip",
    "a common-sense-looking shortcut that is actually invalid here",
]

FORMAT_NAMES = {
    "A": "Standard MCQ (stem + 4 choices)",
    "B": "Numerical-style (answer is one integer/exact scalar; 4 distinct values)",
    "C": "Statement-based (Statement I / Statement II with standard 4 combinations)",
    "D": "Count-based (How many of the given items satisfy condition)",
    "E": "Match-the-list (List-I A-D & List-II P-S; 4 distinct mappings)",
}

FORMAT_WEIGHTS = {
    "Physics":   {"A": 5, "B": 4, "C": 1},
    "Chemistry": {"A": 3, "B": 3, "C": 2, "D": 1, "E": 1},
    "Maths":     {"A": 5, "B": 5},
}
_DEFAULT_FORMATS = {"A": 5, "B": 4}
BATCH_SIZE = 3

# ═══════════════════════════════════════════════════════════════════════
# 8. SOLVE-FIRST STRICT JSON SCHEMA FOR TEST QUESTIONS
# ═══════════════════════════════════════════════════════════════════════
QUESTION_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "subject": {
            "type": "STRING",
            "enum": ["Physics", "Chemistry", "Maths"]
        },
        "chapter": {
            "type": "STRING"
        },
        "question_latex": {
            "type": "STRING",
            "description": "The exact problem statement in LaTeX ($...$). NEVER reference diagrams, figures, or external images."
        },
        "step_by_step_derivation": {
            "type": "STRING",
            "description": "Full formal symbolic derivation, numerical substitution, boundary check, and common trap analysis."
        },
        "final_verified_answer": {
            "type": "STRING",
            "description": "The computed final scalar or expression that matches the intended correct option."
        },
        "options": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Exactly 4 distinct plausible options. 1 matches final_verified_answer, 3 realistic distractors."
        },
        "correct_index": {
            "type": "INTEGER",
            "description": "0-based index (0, 1, 2, or 3) of the correct answer in the options array."
        },
        "mistakeType": {
            "type": "STRING",
            "enum": ["Concept", "Formula", "Silly"]
        }
    },
    "required": [
        "subject",
        "chapter",
        "question_latex",
        "step_by_step_derivation",
        "final_verified_answer",
        "options",
        "correct_index",
        "mistakeType"
    ]
}

TEST_BATCH_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": QUESTION_ITEM_SCHEMA
        }
    },
    "required": ["questions"]
}

# ═══════════════════════════════════════════════════════════════════════
# 9. PHANTOM DIAGRAM FILTER & OPTION SHUFFLING
# ═══════════════════════════════════════════════════════════════════════
_PHANTOM_DIAGRAM_PATTERNS = [
    re.compile(r'\b(as\s+shown\s+in\s+(?:the\s+)?(?:figure|diagram|circuit|image|graph))\b', re.IGNORECASE),
    re.compile(r'\b(in\s+the\s+(?:figure|diagram|circuit|image|graph)\s+(?:below|above|shown))\b', re.IGNORECASE),
    re.compile(r'\b(refer\s+(?:to\s+)?(?:the\s+)?(?:figure|diagram|circuit|image|graph))\b', re.IGNORECASE),
    re.compile(r'\b(shown\s+in\s+the\s+(?:adjacent\s+)?(?:figure|diagram|circuit|image|graph))\b', re.IGNORECASE),
    re.compile(r'\b(from\s+the\s+(?:given\s+)?(?:figure|diagram|circuit|image|graph))\b', re.IGNORECASE),
    re.compile(r'\b(given\s+in\s+(?:the\s+)?(?:figure|diagram|circuit|image|graph))\b', re.IGNORECASE),
    re.compile(r'\b(see\s+(?:the\s+)?(?:figure|diagram|circuit|image|graph))\b', re.IGNORECASE),
    re.compile(r'\[image\]', re.IGNORECASE),
    re.compile(r'\[figure\]', re.IGNORECASE),
    re.compile(r'\[diagram\]', re.IGNORECASE),
]

def clean_question(text: str) -> str | None:
    """Validates question text, rejecting phantom diagrams and unworkably short stems."""
    if not text:
        return None
    cleaned = text.strip()
    if len(cleaned) < 25:
        return None
    for pat in _PHANTOM_DIAGRAM_PATTERNS:
        if pat.search(cleaned):
            return None
    return cleaned

_PLAIN_NUM = re.compile(r'^\$?\s*[-+]?\d+(?:\.\d+)?\s*\$?$')
_GIVEAWAY = re.compile(r'\b(all|none|neither|both)\b[^.]{0,12}\b(above|these|of)\b|\bnone of\b|\ball of\b', re.I)
_STATEMENT_OPT = re.compile(r'^\s*(both\s+statement|statement\s+i)', re.I)

def _num_value(o):
    s = str(o).replace(',', '').strip()
    return float(s.strip('$ ')) if _PLAIN_NUM.match(s) else None

def _realign_options(options, correct_idx):
    """Normalize options array, shuffle safely, and return (shuffled_opts, new_correct_idx)."""
    clean_opts = [str(x).strip() for x in options]
    vals = [_num_value(o) for o in clean_opts]

    if any(_GIVEAWAY.search(o) for o in clean_opts) or sum(bool(_STATEMENT_OPT.match(o)) for o in clean_opts) >= 3:
        order = list(range(4))
    elif all(v is not None for v in vals):
        order = sorted(range(4), key=lambda i: vals[i])
    else:
        order = list(range(4))
        random.shuffle(order)

    shuffled = [clean_opts[i] for i in order]
    new_idx = order.index(correct_idx)
    return shuffled, new_idx

def _scoped_focus_block(specs) -> str:
    if not CHAPTER_FOCUS:
        return ""
    needed = []
    seen = set()
    for sp in specs:
        t = sp.get("topic", "")
        if t in CHAPTER_FOCUS and t not in seen:
            seen.add(t)
            needed.append(f"- {t}: {CHAPTER_FOCUS[t]}")
    if not needed:
        return ""
    return "\n\nFOCUS SUB-TOPICS FOR THIS BATCH:\n" + "\n".join(needed)

# ═══════════════════════════════════════════════════════════════════════
# 10. TEST PARSER & BUILDER
# ═══════════════════════════════════════════════════════════════════════
SUBJECT_WORDS = {
    "physics": "Physics", "phy": "Physics",
    "chemistry": "Chemistry", "chem": "Chemistry",
    "maths": "Maths", "math": "Maths", "mathematics": "Maths"
}
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}
FILLER_WORDS = set("""generate give make create send prepare set me a an the some mock test tests quiz practice
questions question ques mcq mcqs on for of in about jee main mains advanced tough hard difficult challenging my take
please pls paper and class level today tomorrow now quick short full another new some can you do me a favor drill exercise exercises""".split()) | set(NUMBER_WORDS)
_NOT_A_COUNT = {"class", "grade", "std", "chapter", "ch", "unit", "paper", "set", "shift", "day", "year", "jee", "session"}

def parse_test_request(text: str) -> tuple[int, str | None, str]:
    """
    Extracts (count, subject, hint) from natural language prompt.
    Defaults to 5 questions if unspecified. Clamped between 2 and 10.
    """
    low = (text or "").lower()
    count = None
    for m in re.finditer(r'\b(\d{1,2})\b', low):
        prev = low[:m.start()].split()
        if not prev or prev[-1] not in _NOT_A_COUNT:
            count = int(m.group(1))
            break
    if count is None:
        count = next((NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", low) if w in NUMBER_WORDS), 5)
    count = max(2, min(count, 10))

    subject, hint = None, []
    for w in re.findall(r"[a-z][a-z\-']*", low):
        if w in SUBJECT_WORDS:
            subject = subject or SUBJECT_WORDS[w]
        elif w not in FILLER_WORDS:
            hint.append(w)
    return count, subject, " ".join(hint)[:60]

def build_specs(count: int, subject: str | None, hint: str | None):
    """Constructs question blueprints across syllabus tiers, twists, and formats."""
    twists = [TWISTS[i % len(TWISTS)] for i in range(count)]
    random.shuffle(twists)
    levels = (["Medium+", "Hard", "Medium+"] * count)[:count]
    random.shuffle(levels)

    specs = []
    if hint:
        for j in range(count):
            topic = hint if count == 1 else f"{hint} (sub-area {j+1}/{count})"
            specs.append({"subject": subject or "identify from topic", "topic": topic, "twist": twists[j]})
    elif subject:
        t1, t2 = TIER1.get(subject, [])[:], TIER2.get(subject, [])[:]
        random.shuffle(t1)
        random.shuffle(t2)
        n2 = min(len(t2), round(count * 0.35))
        topics = [t1[i % len(t1)] for i in range(count - n2)] + [t2[i % len(t2)] for i in range(n2)] if t1 else [subject] * count
        random.shuffle(topics)
        for j, t in enumerate(topics):
            specs.append({"subject": subject, "topic": t, "twist": twists[j]})
    else:
        base, extra = divmod(count, 3)
        j = 0
        for si, s in enumerate(SUBJECT_ORDER):
            n = base + (1 if si < extra else 0)
            t1, t2 = TIER1[s][:], TIER2[s][:]
            random.shuffle(t1)
            random.shuffle(t2)
            n2 = min(len(t2), round(n * 0.35))
            topics = [t1[i % len(t1)] for i in range(n - n2)] + [t2[i % len(t2)] for i in range(n2)]
            for t in topics:
                specs.append({"subject": s, "topic": t, "twist": twists[j]})
                j += 1

    for sp, lv in zip(specs, levels):
        sp["level"] = lv
        w = FORMAT_WEIGHTS.get(sp["subject"], _DEFAULT_FORMATS)
        sp["format"] = FORMAT_NAMES[random.choices(list(w), weights=list(w.values()))[0]]
    return specs

def chunk_specs(specs, size: int = BATCH_SIZE):
    """Partitions specs into batches of size 3-5."""
    total = len(specs)
    if total == 0:
        return []
    k = max(1, (total + size - 1) // size)
    base, extra = divmod(total, k)
    out, i = [], 0
    for b in range(k):
        n = base + (1 if b < extra else 0)
        out.append(specs[i:i + n])
        i += n
    return out

def _generate_batch(specs, api_key: str):
    lines = "\n".join(
        f"{i}. Subject: {s['subject']} | Topic: {s['topic']} | Level: {s['level']} | "
        f"Format: {s['format']} | Required Twist: {s['twist']}"
        for i, s in enumerate(specs, 1)
    )
    prompt = (
        f"Generate exactly {len(specs)} original JEE MCQs matching these specifications:\n{lines}\n"
        "Remember: Derivation must come first. Text-only solvable. Output strict JSON conforming to schema."
    )
    system_inst = TEST_PROMPT + _scoped_focus_block(specs)
    contents = [prompt]

    text, err = call_gemini(
        role="test",
        contents=contents,
        api_key=api_key,
        json_schema=TEST_BATCH_RESPONSE_SCHEMA,
        system_instruction=system_inst
    )
    if not text:
        return [], err

    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception as e:
                return [], f"json_parse_error: {e}"
        else:
            return [], "no_valid_json"

    raw_qs = data.get("questions", [])
    valid_qs = []
    for q in raw_qs:
        clean_text = clean_question(q.get("question_latex"))
        if not clean_text:
            continue

        opts = q.get("options")
        derivation = str(q.get("step_by_step_derivation") or "").strip()
        correct_idx = q.get("correct_index")
        mt = str(q.get("mistakeType") or "Concept").capitalize()
        if mt not in ("Concept", "Formula", "Silly"):
            mt = "Concept"

        if (
            len(derivation) < 20 or
            not isinstance(opts, list) or len(opts) != 4 or
            correct_idx not in (0, 1, 2, 3)
        ):
            continue

        shuffled_opts, new_correct = _realign_options(opts, correct_idx)
        item = {
            "subject": q.get("subject", "Physics"),
            "chapter": q.get("chapter", "Core"),
            "question": clean_text,
            "options": shuffled_opts,
            "correct": new_correct,
            "solution": derivation,
            "mistakeType": mt
        }
        valid_qs.append(item)

    return valid_qs, None

def build_test(count: int, subject: str | None, hint: str | None, api_key: str) -> tuple[str, list[dict], str | None]:
    """
    Generates a full practice test in small parallel batches (3-5 items)
    to achieve 4-7s execution and avoid 504 timeouts.
    Returns: (title, questions_list, err)
    """
    count = max(2, min(int(count or 5), 10))
    specs = build_specs(count, subject, hint)
    batches = chunk_specs(specs, size=BATCH_SIZE)

    results = [None] * len(batches)
    threads = []

    def _worker(idx, b_specs):
        qs, e = _generate_batch(b_specs, api_key)
        results[idx] = (qs, e)

    for i, b in enumerate(batches):
        t = threading.Thread(target=_worker, args=(i, b))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=45.0)

    all_questions = []
    err = None
    for res in results:
        if res is not None:
            qs, e = res
            all_questions.extend(qs)
            if e and not err:
                err = e
        else:
            if not err:
                err = "batch_timeout"

    # Deduplicate
    seen = set()
    unique = []
    for q in all_questions:
        key = re.sub(r'\W+', '', q["question"].lower())[:60]
        if key not in seen:
            seen.add(key)
            unique.append(q)

    for idx, q in enumerate(unique, 1):
        q["id"] = idx

    title = f"{subject + ': ' if subject else ''}{hint.title()}" if hint else (
        f"{subject}: Mixed Chapters" if subject else "Mixed PCM Challenge"
    )
    return title, unique, err
