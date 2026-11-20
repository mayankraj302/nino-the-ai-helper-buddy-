"""
router.py — Centralised multi-model router, intent classifier, and Gemini generate().
Handles model selection, thinking config, code-execution tool, and 429/503/404 fallback.
"""

import json
import re
import threading
from google import genai
from google.genai import types

from prompts import INTENT_CLASSIFIER_PROMPT

# ═══════════════════════════════════════════════════════════════════════
# 1.  MODEL ROUTER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
# thinking_level constraints:
#   "minimal" → gemini-3.5-flash-lite ONLY  (causes 400 on 3.8/hydrogen)
#   "low" / "medium" → gemini-3.8-flash
#   "high" / "xhigh" → gemini-4-hydrogen  (or 3.8-flash as fallback)

NINO_MODEL_ROUTER = {
    "utility": {
        "model":            "gemini-3.5-flash-lite",
        "thinking_level":   "minimal",
        "code_execution":   False,
        "max_output_tokens": 100,
        "temperature":      0.0,
        "fallback":         None,
    },
    "concept_chat": {
        "model":            "gemini-3.8-flash",
        "thinking_level":   "low",
        "code_execution":   False,
        "max_output_tokens": 1500,
        "temperature":      0.7,
        "fallback":         {"model": "gemini-3.5-flash-lite", "thinking_level": "minimal"},
    },
    "doubt_solve": {
        "model":            "gemini-4-hydrogen",
        "thinking_level":   "high",
        "code_execution":   True,
        "max_output_tokens": 4000,
        "temperature":      0.3,
        "fallback":         {"model": "gemini-3.8-flash", "thinking_level": "high", "code_execution": True},
    },
    "test_generate": {
        "model":            "gemini-4-hydrogen",
        "thinking_level":   "high",
        "code_execution":   False,       # JSON schema instead
        "max_output_tokens": 3000,
        "temperature":      0.4,
        "fallback":         {"model": "gemini-3.8-flash", "thinking_level": "high"},
    },
    "test_verify": {
        "model":            "gemini-3.8-flash",
        "thinking_level":   "high",
        "code_execution":   True,
        "max_output_tokens": 2000,
        "temperature":      0.1,
        "fallback":         None,
    },
    "jee_advanced": {
        "model":            "gemini-4-hydrogen",
        "thinking_level":   "xhigh",
        "code_execution":   True,
        "max_output_tokens": 6000,
        "temperature":      0.3,
        "fallback":         {"model": "gemini-3.8-flash", "thinking_level": "high", "code_execution": True},
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 2.  THINKING CONFIG BUILDER
# ═══════════════════════════════════════════════════════════════════════
_LEVEL_BUDGETS = {
    "minimal": 128,
    "low":     512,
    "medium":  1024,
    "high":    2048,
    "xhigh":  4096,
}


def _make_thinking_config(level: str):
    """Build ThinkingConfig, trying thinking_level (new API) then thinking_budget (legacy)."""
    if not level:
        return None
    upper = level.upper()
    # Newer SDK: string-enum thinking_level
    try:
        return types.ThinkingConfig(thinking_level=upper)
    except (TypeError, ValueError, AttributeError):
        pass
    # Legacy SDK: integer thinking_budget
    budget = _LEVEL_BUDGETS.get(level.lower(), 1024)
    try:
        return types.ThinkingConfig(thinking_budget=budget)
    except (TypeError, ValueError, AttributeError):
        return None


# ═══════════════════════════════════════════════════════════════════════
# 3.  CODE-EXECUTION TOOL
# ═══════════════════════════════════════════════════════════════════════
def _make_code_execution_tool():
    for cls_name in ("ToolCodeExecution", "CodeExecution"):
        cls = getattr(types, cls_name, None)
        if cls:
            try:
                return types.Tool(code_execution=cls())
            except (TypeError, AttributeError):
                continue
    return None


_CODE_EXEC_TOOL = _make_code_execution_tool()


# ═══════════════════════════════════════════════════════════════════════
# 4.  CLIENT CACHE  (thread-safe, one per api_key × timeout)
# ═══════════════════════════════════════════════════════════════════════
_clients: dict = {}
_clients_lock = threading.Lock()
DEFAULT_TIMEOUT_MS = 30_000


def _get_client(api_key: str, timeout_ms: int = DEFAULT_TIMEOUT_MS):
    cache_key = (api_key, timeout_ms)
    with _clients_lock:
        client = _clients.get(cache_key)
        if client is None:
            if len(_clients) >= 50:
                _clients.pop(next(iter(_clients)))
            try:
                client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
            except (TypeError, AttributeError):
                client = genai.Client(api_key=api_key)
            _clients[cache_key] = client
        return client


# ═══════════════════════════════════════════════════════════════════════
# 5.  ERROR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════
def _is_fatal(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in ("401", "403", "api_key_invalid", "permission_denied", "unauthenticated"))


def _should_fallback(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in ("429", "503", "resource_exhausted", "unavailable", "quota", "404", "not_found"))


# ═══════════════════════════════════════════════════════════════════════
# 6.  BUILD CONFIG
# ═══════════════════════════════════════════════════════════════════════
def _build_config(system_instruction, model_cfg: dict, overrides: dict, json_schema=None):
    kw: dict = {
        "system_instruction": system_instruction,
        "temperature":        overrides.get("temperature",      model_cfg.get("temperature", 0.5)),
        "max_output_tokens":  overrides.get("max_output_tokens", model_cfg.get("max_output_tokens", 2000)),
    }
    # Thinking
    level = overrides.get("thinking_level", model_cfg.get("thinking_level"))
    tc = _make_thinking_config(level)
    if tc:
        kw["thinking_config"] = tc
    # Code execution
    if overrides.get("code_execution", model_cfg.get("code_execution", False)) and _CODE_EXEC_TOOL:
        kw["tools"] = [_CODE_EXEC_TOOL]
    # JSON / schema
    if json_schema is not None:
        kw["response_mime_type"] = "application/json"
        kw["response_schema"] = json_schema
    elif overrides.get("json_mode"):
        kw["response_mime_type"] = "application/json"

    return types.GenerateContentConfig(**kw)


# ═══════════════════════════════════════════════════════════════════════
# 7.  SINGLE MODEL CALL
# ═══════════════════════════════════════════════════════════════════════
def _call_model(model_id: str, contents, system_instruction: str, api_key: str,
                model_cfg: dict, overrides: dict, json_schema=None):
    """Returns (text, None) on success or (None, error_string) on failure."""
    try:
        client = _get_client(api_key, overrides.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    except Exception as exc:
        return None, f"client_init: {exc}"

    config = _build_config(system_instruction, model_cfg, overrides, json_schema)

    # Attempt with thinking
    try:
        resp = client.models.generate_content(model=model_id, contents=contents, config=config)
        text = getattr(resp, "text", None)
        if text:
            return text, None
    except Exception as exc:
        err = str(exc)
        low = err.lower()
        # If thinking caused the error, retry without it
        if any(tok in low for tok in ("thinking", "invalid_argument", "400")):
            try:
                cfg2 = _build_config(system_instruction, {**model_cfg, "thinking_level": None}, overrides, json_schema)
                resp2 = client.models.generate_content(model=model_id, contents=contents, config=cfg2)
                text2 = getattr(resp2, "text", None)
                if text2:
                    return text2, None
            except Exception as exc2:
                return None, str(exc2)
        return None, err

    return None, "empty_response"


# ═══════════════════════════════════════════════════════════════════════
# 8.  PUBLIC: generate()
# ═══════════════════════════════════════════════════════════════════════
def generate(route_type: str, contents, system_instruction: str, api_key: str,
             json_schema=None, **overrides):
    """
    Generate content via the model assigned to *route_type*.
    Falls back on 429/503/404 — NEVER downgrades math routes to Flash-Lite.

    Returns (text, error_string_or_None).
    """
    if not api_key:
        return None, "no_key"

    cfg = NINO_MODEL_ROUTER.get(route_type)
    if not cfg:
        return None, f"unknown_route: {route_type}"

    # Try primary model
    text, err = _call_model(cfg["model"], contents, system_instruction, api_key, cfg, overrides, json_schema)
    if text:
        return text, None

    if err and _is_fatal(err):
        return None, err

    # Try fallback
    fb = cfg.get("fallback")
    if fb and err and _should_fallback(err):
        fb_model = fb.get("model", cfg["model"])
        print(f"[router] {cfg['model']} → {fb_model} fallback  ({err[:100]})")
        fb_cfg = {**cfg, **fb}
        text2, err2 = _call_model(fb_model, contents, system_instruction, api_key, fb_cfg, overrides, json_schema)
        if text2:
            return text2, None
        return None, err2 or err

    return None, err


def friendly_error(err: str) -> str:
    if err == "no_key":
        return "Nino isn't configured yet. Please add a Gemini key in Settings or try later."
    if err and _is_fatal(err):
        return "That Gemini API key looks invalid or lacks permission. Please check it in Settings."
    if err and ("429" in err or "quota" in err.lower()):
        return "Nino is very busy right now (AI quota reached). Please try again later, or add your own Gemini key in Settings."
    return "Nino couldn't answer that just now. Please try again in a moment."


# ═══════════════════════════════════════════════════════════════════════
# 9.  INTENT CLASSIFIER  (2-stage: deterministic → AI enum)
# ═══════════════════════════════════════════════════════════════════════
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank\s*you|ok|okay|hii+|hlo|yo|sup|gm|gn|bye|good\s*(morning|night|evening))[\s!.?]*$",
    re.I,
)
_DOUBT_RE = re.compile(
    r"solve|evaluate|integrat|differentiat|derivative|integral|\blimits?\b|equation|find the|calculate|compute|prove|"
    r"value of|how many|which of the following|numerical|mechanism|reaction|oxidation|"
    r"\bph\b|mole|\bforce\b|voltage|probability|matrix|\b(?:sin|cos|tan|log)\b|=|\^|\\|∫|√|\d\s*[+\-*/×]\s*\d",
    re.I,
)
_ADVANCED_RE = re.compile(
    r"irodov|pathfinder|hcv|dc\s*pandey|olympiad|jee\s*advanced|prove\s*rigorously|multiple\s*methods|"
    r"rigorous|edge[- ]?case|competition",
    re.I,
)
_DISTRESS_RE = re.compile(
    r"(suicid|kill\s+my\s*self|end\s+(?:my\s+life|it\s+all)|want\s+to\s+die|self[\s-]?harm|"
    r"hurt\s+my\s*self|hopeless|worthless|depress|panic|lonely|\bcry(?:ing)?\b|burn(?:t|ed)?\s*out|"
    r"mar\s+jau|marna\s+chah|jeena\s+nahi|zindagi\s+khatam)",
    re.I,
)
_SHORT_FOLLOWUP_RE = re.compile(
    r"^(why|how|what|explain|step|but|and|also|wait|huh|so|haan|tell me|show)\b.{0,60}$",
    re.I,
)
_VALID_ROUTES = frozenset(("concept_chat", "doubt_solve", "jee_advanced"))


def classify_intent(
    text: str,
    has_image: bool = False,
    is_test_endpoint: bool = False,
    deep_solve: bool = False,
    previous_route: str | None = None,
    api_key: str | None = None,
) -> str:
    """Return one of: utility, concept_chat, doubt_solve, test_generate, jee_advanced."""
    text = (text or "").strip()

    # ── Stage 1: Deterministic (0 ms, $0) ───────────────────────────
    if is_test_endpoint:
        return "test_generate"
    if deep_solve:
        return "jee_advanced"
    if has_image:
        return "doubt_solve"
    if not text:
        return "concept_chat"
    if _GREETING_RE.match(text):
        return "concept_chat"
    if _DISTRESS_RE.search(text):
        return "concept_chat"          # safety → full personality prompt
    if _ADVANCED_RE.search(text):
        return "jee_advanced"
    if _DOUBT_RE.search(text):
        return "doubt_solve"

    # ── Sticky multi-turn routing ────────────────────────────────────
    if previous_route in ("doubt_solve", "jee_advanced") and _SHORT_FOLLOWUP_RE.match(text):
        return previous_route

    # ── Stage 2: AI enum classifier (~150-250 ms) ───────────────────
    if api_key and len(text) > 5:
        try:
            classified = _ai_classify(text, api_key)
            if classified:
                return classified
        except Exception as exc:
            print(f"[classifier ERROR]: {exc}")

    # ── Fallback: accuracy-biased ────────────────────────────────────
    return "doubt_solve"


def _ai_classify(text: str, api_key: str) -> str | None:
    """Fast 1-token enum classification via flash-lite."""
    try:
        client = _get_client(api_key, 5000)
    except Exception:
        return None

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=text)])]

    # ── Try text/x.enum response format ──────────────────────────────
    for mime, schema in [
        ("text/x.enum",       ["concept_chat", "doubt_solve", "jee_advanced"]),
        ("application/json",  {"type": "OBJECT",
                               "properties": {"route": {"type": "STRING", "enum": ["concept_chat", "doubt_solve", "jee_advanced"]}},
                               "required": ["route"]}),
    ]:
        try:
            kw: dict = {
                "system_instruction":  INTENT_CLASSIFIER_PROMPT,
                "temperature":         0.0,
                "max_output_tokens":   30,
                "response_mime_type":  mime,
                "response_schema":    schema,
            }
            tc = _make_thinking_config("minimal")
            if tc:
                kw["thinking_config"] = tc

            config = types.GenerateContentConfig(**kw)
            resp = client.models.generate_content(model="gemini-3.5-flash-lite", contents=contents, config=config)
            raw = (getattr(resp, "text", "") or "").strip().strip('"\'').lower()

            if mime == "application/json":
                try:
                    raw = json.loads(raw).get("route", "").strip().lower()
                except (json.JSONDecodeError, AttributeError):
                    pass

            if raw in _VALID_ROUTES:
                return raw
        except Exception:
            continue              # try next format

    return None
