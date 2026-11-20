"""
engine.py — High-Speed Multi-Model Engine & Intent Router for Nino v3.

KEY PERFORMANCE OPTIMIZATIONS (ZERO QUALITY LOSS):
- Preserves exact model & thinking levels:
    chat   -> gemini-3.8-flash (thinking=low)
    solver -> gemini-4-hydrogen (thinking=high, code_exec=True)
    test   -> gemini-3.8-flash (thinking=high)
- Removes deprecated gemini-2.5-* fallbacks (retiring Dec 1, 2026); uses 3.6-flash / 3.5-flash-lite.
- Omits sub-1.0 temperature overrides on Gemini 3/4 thinking models (prevents thinking loops).
- Safe combined max_output_tokens budgets (thinking + output share the cap on Gemini 3/4).
- Eliminates 12x silent retry loop on empty responses & uses per-role HTTP timeouts.
- Fixes detect_task() so conceptual/strategy chat never gets hijacked by heavy solver.
- Parallelizes test generation to 1 question per concurrent worker (<=5 questions) or 2 per worker (6-10),
  cutting test generation wall-clock latency by 3x-4x while keeping thinking=high.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
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
# 1. ROLES, MODELS & CASCADES (Exact Thinking Levels Preserved)
# ═══════════════════════════════════════════════════════════════════════
ROLES_CONFIG = {
    "chat": {
        "primary": {"model": "gemini-3.8-flash", "thinking": "low", "code_exec": False},
        "fallbacks": [
            {"model": "gemini-3.5-flash-lite", "thinking": "minimal", "code_exec": False},
            {"model": "gemini-3.6-flash", "thinking": "low", "code_exec": False},
        ],
        "default_tokens": 4096,     # Combined thinking + visible output cap
        "timeout_ms": 22_000,
        "system_prompt": CHAT_PROMPT,
    },
    "solver": {
        "primary": {"model": "gemini-4-hydrogen", "thinking": "high", "code_exec": True},
        "fallbacks": [
            {"model": "gemini-3.8-flash", "thinking": "high", "code_exec": True},
            {"model": "gemini-3.6-flash", "thinking": "high", "code_exec": True},
        ],
        "default_tokens": 12288,    # Safe headroom for high thinking + full derivation
        "timeout_ms": 48_000,
        "system_prompt": SOLVER_PROMPT,
    },
    "test": {
        "primary": {"model": "gemini-3.8-flash", "thinking": "high", "code_exec": False},
        "fallbacks": [
            {"model": "gemini-4-hydrogen", "thinking": "high", "code_exec": False},
            {"model": "gemini-3.6-flash", "thinking": "high", "code_exec": False},
        ],
        "default_tokens": 8192,     # Safe headroom for high thinking + 1-2 structured MCQs
        "timeout_ms": 34_000,
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

# Models that support thinking_level="MINIMAL"
_MINIMAL_SUPPORTED_SUBSTRINGS = ("flash-lite", "3.6-flash", "3-flash-preview")

# ═══════════════════════════════════════════════════════════════════════
# 2. CLIENT CACHING & CONFIG BUILDERS
# ═══════════════════════════════════════════════════════════════════════
_clients = {}
_clients_lock = threading.Lock()
DEFAULT_TIMEOUT_MS = 35_000

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
    Builds ThinkingConfig safely without invalid API calls.
    'minimal' is only sent to models that explicitly support it (e.g., flash-lite, 3.6-flash);
    otherwise mapped to 'low' to prevent 400 validation errors on 3.8-flash / Gemini 4.
    """
    if not level:
        return None
    model_lower = (model_name or "").lower()
    if "2.0" in model_lower:
        return None

    clean_level = str(level).strip().lower()
    if clean_level == "minimal" and not any(s in model_lower for s in _MINIMAL_SUPPORTED_SUBSTRINGS):
        clean_level = "low"

    # Gemini 2.5 legacy models only accept thinking_budget
    if "2.5" in model_lower:
        budget = _LEVEL_BUDGETS.get(clean_level, 1024)
        try:
            return types.ThinkingConfig(thinking_budget=budget)
        except Exception:
            return None

    try:
        return types.ThinkingConfig(thinking_level=clean_level.upper(), include_thoughts=False)
    except Exception:
        try:
            return types.ThinkingConfig(thinking_level=clean_level.upper())
        except Exception:
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
    """Prepend system prompt into user contents only when developer instruction is rejected."""
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
        if isinstance(first, types.Content):
            new_part = types.Part.from_text(text=prefix_text)
            new_content = types.Content(role=first.role or "user", parts=[new_part] + list(first.parts))
            return [new_content] + contents[1:]
        if hasattr(first, "parts"):
            new_part = types.Part.from_text(text=prefix_text)
            new_content = types.Content(role="user", parts=[new_part] + list(first.parts))
            return [new_content] + contents[1:]
        return [prefix_text] + list(contents)
    return contents

def _build_gen_config(sys_param, thinking_level: str | None, model_id: str,
                      code_exec: bool, max_tokens: int,
                      json_schema=None, json_mode: bool = False):
    """
    Constructs GenerateContentConfig.
    Intentionally omits sub-1.0 temperature on Gemini 3/4 thinking models per Google guidance
    (low temperature causes reasoning loops & latency spikes on thinking models).
    """
    kw = {"max_output_tokens": max_tokens}
    if sys_param is not None:
        kw["system_instruction"] = sys_param

    tc = _build_thinking_config(thinking_level, model_id)
    if tc:
        kw["thinking_config"] = tc

    tools = _get_tools(code_exec)
    if tools:
        kw["tools"] = tools

    if json_schema is not None:
        kw["response_mime_type"] = "application/json"
        kw["response_schema"] = json_schema
    elif json_mode:
        kw["response_mime_type"] = "application/json"

    return types.GenerateContentConfig(**kw)

def _extract_response_text(resp) -> str | None:
    """Safely extracts visible non-thought text from a Gemini response."""
    text = getattr(resp, "text", None)
    if text:
        return text
    try:
        candidates = getattr(resp, "candidates", None) or []
        parts = getattr(candidates[0].content, "parts", None) or []
        chunks = [
            p.text for p in parts
            if getattr(p, "text", None) and not getattr(p, "thought", False)
        ]
        joined = "".join(chunks).strip()
        return joined or None
    except Exception:
        return None

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
        "404", "not_found", "not found", "500", "internal"
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
        return "Google's model is currently busy. Nino is rerouting your request — please tap send again!"
    return f"Nino encountered an error: {err_str[:80]}. Please try sending your message again."

# ═══════════════════════════════════════════════════════════════════════
# 4. FAST SINGLE-MODEL INVOCATION (NO SILENT 12x RETRY TRAP)
# ═══════════════════════════════════════════════════════════════════════
def _invoke_model_attempt(client, model_id: str, contents, system_inst: str,
                          thinking_level: str | None, code_exec: bool,
                          max_tokens: int, json_schema=None, json_mode: bool = False):
    """
    Executes a single model call with targeted auto-healing ONLY on specific API errors:
      - 'developer instruction' rejected -> retry with prefixed system prompt
      - 'tool' / 'code_execution' rejected -> retry without tool
      - 'thinking' rejected -> retry without thinking config
    Immediately exits on valid empty response instead of looping 12 times.
    """
    sys_modes = [("string", system_inst), ("prefixed", None)] if system_inst else [("none", None)]
    eff_thinking = thinking_level
    eff_code_exec = code_exec

    for mode, sys_param in sys_modes:
        eff_contents = _prepare_prefixed_contents(contents, system_inst) if mode == "prefixed" else contents

        for _ in range(3):  # At most 2 targeted parameter heals
            try:
                config = _build_gen_config(
                    sys_param=sys_param,
                    thinking_level=eff_thinking,
                    model_id=model_id,
                    code_exec=eff_code_exec,
                    max_tokens=max_tokens,
                    json_schema=json_schema,
                    json_mode=json_mode,
                )
                resp = client.models.generate_content(
                    model=model_id,
                    contents=eff_contents,
                    config=config,
                )
                text = _extract_response_text(resp)
                if text:
                    return text, None
                # Valid HTTP 200 but empty text -> don't hammer same model; let caller cascade
                return None, "empty_response"

            except Exception as e:
                err_str = str(e)
                err_low = err_str.lower()

                if _is_fatal(err_str) or _is_cascade_needed(err_str) or _is_rate_limit(err_str):
                    return None, err_str

                if "developer instruction" in err_low or "developer_instruction" in err_low:
                    break  # Move to next sys_mode ("prefixed")

                if eff_code_exec and ("tool" in err_low or "code_execution" in err_low):
                    eff_code_exec = False
                    continue

                if eff_thinking and ("thinking" in err_low or "invalid_argument" in err_low):
                    eff_thinking = None
                    continue

                return None, err_str

    return None, "empty_response"

# ═══════════════════════════════════════════════════════════════════════
# 5. PUBLIC CALL_GEMINI & STREAM_GEMINI WITH FAST FALLBACK
# ═══════════════════════════════════════════════════════════════════════
def call_gemini(role: str, contents, api_key: str, json_schema=None, json_mode: bool = False,
                system_instruction: str = None) -> tuple[str | None, str | None]:
    """
    Fast, resilient Gemini API caller:
    - Uses per-role timeout so stalled sockets fail over quickly.
    - On 429: one brief jittered retry, then immediately cascades to fallback model.
    - On 503/504/404/timeout: immediately cascades to fallback model.
    """
    if not api_key:
        return None, "no_key"

    role_key = (role or "chat").lower()
    config = ROLES_CONFIG.get(role_key, ROLES_CONFIG["chat"])
    models_to_try = [config["primary"]] + config["fallbacks"]
    effective_system = system_instruction if system_instruction is not None else config.get("system_prompt", "")
    timeout_ms = config.get("timeout_ms", DEFAULT_TIMEOUT_MS)
    max_tokens = config.get("default_tokens", 4096)

    try:
        client = _get_client(api_key, timeout_ms=timeout_ms)
    except Exception as exc:
        return None, f"client_init: {exc}"

    last_error = None
    for model_cfg in models_to_try:
        model_id = model_cfg["model"]
        thinking = model_cfg.get("thinking")
        code_exec = model_cfg.get("code_exec", False)

        for attempt in range(2):  # 1 retry max on 429 before fast cascade
            text, err = _invoke_model_attempt(
                client=client,
                model_id=model_id,
                contents=contents,
                system_inst=effective_system,
                thinking_level=thinking,
                code_exec=code_exec,
                max_tokens=max_tokens,
                json_schema=json_schema,
                json_mode=json_mode,
            )
            if text:
                return text, None

            last_error = err
            if _is_fatal(err):
                return None, err

            if _is_rate_limit(err) and attempt == 0:
                time.sleep(0.8 + random.uniform(0.1, 0.4))
                continue

            break  # Cascade immediately on 503/504/404/timeout/empty

    return None, last_error

def stream_gemini(role: str, contents, api_key: str, system_instruction: str = None):
    """
    Generator yielding visible text chunks in real time for SSE streaming (/guide/stream).
    Automatically falls back across models if the primary fails before emitting tokens.
    """
    if not api_key:
        raise RuntimeError("no_key")

    role_key = (role or "chat").lower()
    config = ROLES_CONFIG.get(role_key, ROLES_CONFIG["chat"])
    models_to_try = [config["primary"]] + config["fallbacks"]
    effective_system = system_instruction if system_instruction is not None else config.get("system_prompt", "")
    timeout_ms = config.get("timeout_ms", DEFAULT_TIMEOUT_MS)
    max_tokens = config.get("default_tokens", 4096)
    client = _get_client(api_key, timeout_ms=timeout_ms)

    last_err = None
    for model_cfg in models_to_try:
        model_id = model_cfg["model"]
        emitted = False
        try:
            gen_cfg = _build_gen_config(
                sys_param=effective_system,
                thinking_level=model_cfg.get("thinking"),
                model_id=model_id,
                code_exec=model_cfg.get("code_exec", False),
                max_tokens=max_tokens,
            )
            stream = client.models.generate_content_stream(
                model=model_id,
                contents=contents,
                config=gen_cfg,
            )
            for chunk in stream:
                txt = _extract_response_text(chunk)
                if txt:
                    emitted = True
                    yield txt
            if emitted:
                return
        except Exception as exc:
            last_err = str(exc)
            if emitted or _is_fatal(last_err):
                raise
            continue

    raise RuntimeError(last_err or "empty_response")

# ═══════════════════════════════════════════════════════════════════════
# 6. PRECISION INTENT ROUTER (Stops Conceptual Chat From Hitting Solver)
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

# Phrases that clearly mean conceptual chat, mentorship, or study strategy (NOT numerical solving)
_CHAT_CONCEPT_GUARDS = [
    r'\bhow\s+(?:should\s+i|to)\s+(?:study|revise|prepare|approach|master|remember|memorize|start)\b',
    r'\b(?:study\s+plan|timetable|schedule|roadmap|strategy|weightage|important\s+(?:chapters|topics))\b',
    r'\b(?:explain|what\s+is|what\s+are|difference\s+between|intuition\s+(?:behind|for)|overview\s+of|summarize|summary\s+of)\b',
    r'\b(?:burnout|stressed|anxious|demotivated|overwhelmed|backlog|procrastinat|tired|scared)\b',
    r'\b(?:tips?\s+(?:for|on)|tricks?\s+for|best\s+books?|resources?\s+for)\b',
]

# Explicit problem-solving action verbs (NOT bare chapter nouns!)
_SOLVER_ACTION_PATTERNS = [
    r'\bsolve\b', r'\bevaluate\b', r'\bcalculate\b', r'\bcompute\b',
    r'\bprove\s+that\b', r'\bshow\s+that\b', r'\bderive\b',
    r'\bfind\s+(?:the\s+)?(?:value|ratio|roots?|maximum|minimum|area|volume|magnitude|force|velocity|speed|acceleration|current|charge|potential|energy|work|power|time|distance|angle|range|height|frequency|wavelength|pressure|temperature|ph|molarity|molality|emf|entropy|enthalpy|probability|number\s+of|sum\s+of|limit\s+of|integral|derivative|equation|locus|shortest\s+distance)\b',
    r'\b(?:integrate|differentiate|factori[sz]e|simplify)\b',
    r'\b(?:step\s+by\s+step|full\s+solution|complete\s+derivation|solve\s+completely)\b',
    r'\b(?:draw\s+a\s+diagram|explain\s+visually|show\s+a\s+graph)\b',
    r'\b(?:if\s+a\s+(?:body|particle|block|car|ball|stone|sphere|disc|rod|wire|charge|electron|proton|gas|solution|matrix|function|curve))\b',
    r'\b(?:a\s+(?:block|particle|body|ball|projectile|capacitor|resistor|coil|lens|mirror|gas)\s+of\s+(?:mass|radius|length|resistance|capacitance|focal\s+length|volume))\b',
]

_MATH_SYMBOLS = (
    '∫', '√', '∑', '∏', '∂', '∆', '≠', '≤', '≥', '±', '×', '÷',
    r'\int', r'\frac', r'\sqrt', r'\sum', r'\lim', r'\partial', 'dy/dx', 'd/dx'
)

def detect_task(text: str, has_photo: bool = False, deep_solve: bool = False) -> str:
    """
    Precision intent router:
      a. Photo attached OR deep_solve toggle active -> 'solver'
      b. Test generation request (with negative guards) -> 'test'
      c. Explicit equations/math symbols OR concrete problem-solving prompt -> 'solver'
      d. Conceptual theory, strategy, burnout, or general chat -> 'chat'
    """
    if has_photo or deep_solve:
        return 'solver'

    clean_text = (text or "").strip()
    if not clean_text:
        return 'chat'

    low = clean_text.lower()
    tokens = set(re.findall(r'[a-z]+', low))

    # b. Test detection (catches direct requests upfront so chat doesn't waste a turn)
    has_neg_guard = any(re.search(pat, low) for pat in _TEST_NEGATIVE_GUARDS)
    if not has_neg_guard:
        has_action = bool(tokens & _TEST_ACTIONS)
        has_target = bool(tokens & _TEST_TARGETS)

        standalone = bool(
            re.search(r'\b(mock\s+tests?|practice\s+tests?|jee\s+mock|take\s+(?:a\s+)?test|quiz\s+me|test\s+my\s+(?:concepts?|knowledge)|check\s+if\s+i\s+know|dpp\s+(?:on|for))\b', low) or
            re.search(r'\b\d+\s*(?:mcqs?|questions?|problems?|ques)\b', low)
        )

        if (has_action and has_target) or standalone:
            return 'test'

    # c. Hard math signals -> always solver
    if any(sym in clean_text for sym in _MATH_SYMBOLS):
        return 'solver'

    has_equation = bool(
        re.search(r'\d\s*[+\-*/×=^]\s*\d', low) or
        re.search(r'\b[a-z]\s*=\s*[-+]?\d', low) or
        re.search(r'\d+\s*(?:m/s|m/s\^2|kg|ohm|Ω|μf|mf|ev|mol|atm|bar|kj|nm|cm|mm|hz|rad)\b', low)
    )
    if has_equation:
        return 'solver'

    # Explicit problem-solving phrasing -> solver (even if mixed with 'explain')
    if any(re.search(pat, low) for pat in _SOLVER_ACTION_PATTERNS):
        # Don't send "explain how to study..." or "explain what is torque" to solver
        if not any(re.search(pat, low) for pat in _CHAT_CONCEPT_GUARDS):
            return 'solver'
        # Still route to solver if they explicitly asked to solve/calculate/derive/draw
        if re.search(r'\b(solve|calculate|compute|evaluate|derive|prove|step\s+by\s+step|draw\s+a\s+diagram|explain\s+visually)\b', low):
            return 'solver'

    # d. Fast conversational mentor/theory/strategy -> 'chat'
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
MAX_PARALLEL_WORKERS = 5

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
            "description": "Self-contained JEE problem statement in LaTeX ($...$). Never mention figures, diagrams, or images."
        },
        "step_by_step_derivation": {
            "type": "STRING",
            "description": "Concise, rigorous 3-6 step symbolic + numerical derivation with final verification and 1-line trap note."
        },
        "final_verified_answer": {
            "type": "STRING",
            "description": "Exact verified scalar or expression matching the correct option."
        },
        "options": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Exactly 4 distinct options: 1 matches final_verified_answer, 3 realistic JEE distractors."
        },
        "correct_index": {
            "type": "INTEGER",
            "description": "0-based index (0, 1, 2, or 3) of the correct option."
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
    return "\n\nFOCUS SUB-TOPICS:\n" + "\n".join(needed)

# ═══════════════════════════════════════════════════════════════════════
# 10. TEST PARSER & HIGH-PARALLELISM BUILDER
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
            topic = hint if count == 1 else f"{hint} (distinct sub-concept {j+1}/{count})"
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

def chunk_specs(specs, max_workers: int = MAX_PARALLEL_WORKERS):
    """
    Adaptive parallel chunking:
    - <=5 questions -> 1 question per concurrent worker (fastest wall-clock time!)
    - 6-10 questions -> balanced batches of 2 across <=5 concurrent workers (avoids RPM 429s)
    """
    total = len(specs)
    if total == 0:
        return []
    k = min(total, max_workers)
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
        f"Format: {s['format']} | Twist: {s['twist']}"
        for i, s in enumerate(specs, 1)
    )
    prompt = (
        f"Generate exactly {len(specs)} original JEE MCQ(s):\n{lines}\n"
        "Solve first with concise derivation. Text-only solvable. Return strict JSON."
    )
    system_inst = TEST_PROMPT + _scoped_focus_block(specs)

    text, err = call_gemini(
        role="test",
        contents=prompt,
        api_key=api_key,
        json_schema=TEST_BATCH_RESPONSE_SCHEMA,
        system_instruction=system_inst,
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
        valid_qs.append({
            "subject": q.get("subject", "Physics"),
            "chapter": q.get("chapter", "Core"),
            "question": clean_text,
            "options": shuffled_opts,
            "correct": new_correct,
            "solution": derivation,
            "mistakeType": mt,
        })

    return valid_qs, None

def build_test(count: int, subject: str | None, hint: str | None, api_key: str) -> tuple[str, list[dict], str | None]:
    """
    Generates a full practice test concurrently (1 question/worker for <=5 Qs, 2 Qs/worker for 6-10 Qs)
    preserving thinking='high' on gemini-3.8-flash while cutting wall-clock wait by 3x-4x.
    Returns: (title, questions_list, err)
    """
    count = max(2, min(int(count or 5), 10))
    specs = build_specs(count, subject, hint)
    batches = chunk_specs(specs, max_workers=MAX_PARALLEL_WORKERS)

    ordered_results = [[] for _ in range(len(batches))]
    err = None

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        future_to_idx = {
            executor.submit(_generate_batch, b_specs, api_key): idx
            for idx, b_specs in enumerate(batches)
        }
        try:
            for fut in as_completed(future_to_idx, timeout=40.0):
                idx = future_to_idx[fut]
                try:
                    qs, e = fut.result()
                    ordered_results[idx] = qs
                    if e and not err:
                        err = e
                except Exception as exc:
                    if not err:
                        err = str(exc)
        except TimeoutError:
            if not err:
                err = "batch_timeout"

    all_questions = [q for bucket in ordered_results for q in bucket]

    # Deduplicate while preserving batch order
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
