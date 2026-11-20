"""
router.py — Multi-Model Router with Fast Failover & Code-Execution Safe Extraction.
"""

import re
import threading
import time
from google import genai
from google.genai import types

from prompts import INTENT_CLASSIFIER_PROMPT

# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL CASCADES (Valid Gemini API models only)
# ═══════════════════════════════════════════════════════════════════════
MODEL_CASCADES = {
    "utility": [
        {"model": "gemini-2.5-flash-lite", "thinking": None,   "code_exec": False},
        {"model": "gemini-2.0-flash",      "thinking": None,   "code_exec": False},
    ],
    "concept_chat": [
        {"model": "gemini-2.5-flash",      "thinking": "low",  "code_exec": False},
        {"model": "gemini-2.0-flash",      "thinking": None,   "code_exec": False},
    ],
    "doubt_solve": [
        {"model": "gemini-2.5-flash",      "thinking": "medium", "code_exec": True},
        {"model": "gemini-2.5-pro",        "thinking": "medium", "code_exec": True},
        {"model": "gemini-2.0-flash",      "thinking": None,     "code_exec": False},
    ],
    "test_generate": [
        {"model": "gemini-2.5-flash",      "thinking": "low",  "code_exec": False},
        {"model": "gemini-2.0-flash",      "thinking": None,   "code_exec": False},
    ],
    "test_verify": [
        {"model": "gemini-2.5-flash",      "thinking": "low",  "code_exec": True},
        {"model": "gemini-2.5-pro",        "thinking": "low",  "code_exec": True},
    ],
    "jee_advanced": [
        {"model": "gemini-2.5-pro",        "thinking": "high",   "code_exec": True},
        {"model": "gemini-2.5-flash",      "thinking": "high",   "code_exec": True},
        {"model": "gemini-2.0-flash",      "thinking": None,     "code_exec": False},
    ],
}

DEFAULT_PARAMS = {
    "utility":        {"max_output_tokens": 64,   "temperature": 0.0, "timeout_ms": 6_000},
    "concept_chat":   {"max_output_tokens": 1800, "temperature": 0.7, "timeout_ms": 20_000},
    "doubt_solve":    {"max_output_tokens": 4096, "temperature": 0.3, "timeout_ms": 30_000},
    "test_generate":  {"max_output_tokens": 6000, "temperature": 0.4, "timeout_ms": 28_000},
    "test_verify":    {"max_output_tokens": 2048, "temperature": 0.1, "timeout_ms": 20_000},
    "jee_advanced":   {"max_output_tokens": 6000, "temperature": 0.2, "timeout_ms": 38_000},
}

_LEVEL_BUDGETS = {
    "minimal": 0,
    "low":     512,
    "medium":  1536,
    "high":    3072,
}

# ═══════════════════════════════════════════════════════════════════════
# 2. CLIENT CACHE
# ═══════════════════════════════════════════════════════════════════════
_clients = {}
_clients_lock = threading.Lock()
DEFAULT_TIMEOUT_MS = 28_000


def _get_client(api_key: str, timeout_ms: int = DEFAULT_TIMEOUT_MS):
    cache_key = (api_key, timeout_ms)
    with _clients_lock:
        client = _clients.get(cache_key)
        if client is None:
            if len(_clients) >= 50:
                _clients.pop(next(iter(_clients)))
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=timeout_ms),
                )
            except Exception:
                client = genai.Client(api_key=api_key)
            _clients[cache_key] = client
        return client


def _build_thinking(level: str | None, model_id: str):
    if not level or "2.0" in model_id or "lite" in model_id:
        return None
    budget = _LEVEL_BUDGETS.get(level.lower(), 1024)
    try:
        return types.ThinkingConfig(thinking_budget=budget)
    except Exception:
        return None


def _get_tools(code_exec: bool):
    if not code_exec:
        return None
    for cls_name in ("ToolCodeExecution", "CodeExecution"):
        cls = getattr(types, cls_name, None)
        if cls:
            try:
                return [types.Tool(code_execution=cls())]
            except Exception:
                pass
    return None


def _extract_text(resp) -> str | None:
    """Safely extract non-thought text even when code_execution parts are present."""
    try:
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            chunks = []
            for p in parts:
                if getattr(p, "thought", False):
                    continue
                t = getattr(p, "text", None)
                if t:
                    chunks.append(t)
            joined = "".join(chunks).strip()
            if joined:
                return joined
    except Exception:
        pass
    try:
        text = getattr(resp, "text", None)
        return text.strip() if text else None
    except Exception:
        return None


def _is_fatal(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in ("401", "403", "api_key_invalid", "permission_denied", "unauthenticated"))


def _is_rate_limit(err: str) -> bool:
    low = (err or "").lower()
    return "429" in low or "quota" in low or "resource_exhausted" in low


def _is_unavailable_503(err: str) -> bool:
    low = (err or "").lower()
    return "503" in low or "unavailable" in low or "high demand" in low or "overloaded" in low


def _is_timeout(err: str) -> bool:
    low = (err or "").lower()
    return "timeout" in low or "timed out" in low or "deadline" in low


# ═══════════════════════════════════════════════════════════════════════
# 3. SINGLE MODEL INVOCATION (Max 2 attempts: full config -> plain fallback)
# ═══════════════════════════════════════════════════════════════════════
def _call_single_model(
    client,
    model_id: str,
    contents,
    system_instruction: str,
    thinking_level: str | None,
    code_exec: bool,
    max_tokens: int,
    temp: float,
    json_schema=None,
    json_mode=False,
):
    attempts = [(thinking_level, code_exec)]
    if thinking_level or code_exec:
        attempts.append((None, False))

    last_err = "empty_response"
    for try_thinking, try_tools in attempts:
        kw = {
            "temperature": temp,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            kw["system_instruction"] = system_instruction

        tc = _build_thinking(try_thinking, model_id)
        if tc:
            kw["thinking_config"] = tc

        tools = _get_tools(try_tools)
        if tools and json_schema is None:
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
                contents=contents,
                config=config,
            )
            text = _extract_text(resp)
            if text:
                return text, None
            last_err = "empty_response"
        except Exception as e:
            err = str(e)
            last_err = err
            low = err.lower()

            if _is_fatal(err) or _is_unavailable_503(err) or _is_rate_limit(err) or _is_timeout(err):
                return None, err
            if "404" in low or "not_found" in low:
                return None, err
            # Otherwise fall through to the plain (no thinking / no tools) attempt

    return None, last_err


# ═══════════════════════════════════════════════════════════════════════
# 4. PUBLIC: generate()
# ═══════════════════════════════════════════════════════════════════════
def generate(
    route_type: str,
    contents,
    system_instruction: str,
    api_key: str,
    json_schema=None,
    **overrides,
):
    if not api_key:
        return None, "no_key"

    cascade = MODEL_CASCADES.get(route_type)
    if not cascade:
        return None, f"unknown_route: {route_type}"

    params = DEFAULT_PARAMS.get(
        route_type, {"max_output_tokens": 2000, "temperature": 0.4, "timeout_ms": DEFAULT_TIMEOUT_MS}
    )
    max_tokens = overrides.get("max_output_tokens", params["max_output_tokens"])
    temp = overrides.get("temperature", params["temperature"])
    timeout_ms = overrides.get("timeout_ms", params["timeout_ms"])
    json_mode = overrides.get("json_mode", False)

    try:
        client = _get_client(api_key, timeout_ms)
    except Exception as exc:
        return None, f"client_init: {exc}"

    last_error = None
    for step in cascade:
        model_id = step["model"]
        thinking = overrides.get("thinking_level", step["thinking"])
        code_exec = overrides.get("code_execution", step["code_exec"])

        text, err = _call_single_model(
            client=client,
            model_id=model_id,
            contents=contents,
            system_instruction=system_instruction,
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

        print(f"[router] {model_id} failed ({str(last_error)[:120]}). Cascading...")
        if _is_rate_limit(err):
            time.sleep(0.4)

    return None, last_error


def friendly_error(err: str) -> str:
    if err == "no_key":
        return "Nino isn't configured yet. Please add a Gemini key in Settings or try later."
    if err and _is_fatal(err):
        return "That Gemini API key looks invalid or lacks permission. Please check it in Settings."
    if err and _is_rate_limit(err):
        return "Nino is receiving heavy traffic (AI quota reached). Wait a few seconds or add your own key in Settings."
    if err and _is_unavailable_503(err):
        return "Google's AI servers are temporarily overloaded. Please tap send again in a moment."
    if err and _is_timeout(err):
        return "That problem took too long to solve. Try breaking it into smaller parts or turning off Deep Solve."
    return "Nino hit a temporary snag solving that. Please try sending your message again."


# ═══════════════════════════════════════════════════════════════════════
# 5. INTENT CLASSIFIER (Fast Deterministic -> Lightweight AI Fallback)
# ═══════════════════════════════════════════════════════════════════════
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank\s*you|ok|okay|hii+|hlo|yo|sup|gm|gn|bye|good\s*(morning|night|evening))[\s!.?]*$",
    re.I,
)
_CONCEPT_RE = re.compile(
    r"^(what\s+is|define|explain\s+the\s+concept|difference\s+between|how\s+to\s+prepare|"
    r"study\s+plan|timetable|strategy|tips\s+for|weightage|syllabus)\b",
    re.I,
)
_DOUBT_RE = re.compile(
    r"solve|evaluate|integrat|differentiat|derivative|integral|\blimits?\b|equation|find\s+the|calculate|compute|prove|"
    r"value\s+of|how\s+many|which\s+of\s+the\s+following|numerical|mechanism|reaction|oxidation|"
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
    r"^(why|how|what|explain|step|but|and|also|wait|huh|so|haan|tell\s+me|show|full\s+solution|complete\s+solution)\b.{0,60}$",
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
    text = (text or "").strip()

    if is_test_endpoint:
        return "test_generate"
    if deep_solve:
        return "jee_advanced"
    if has_image:
        return "doubt_solve"
    if not text or _GREETING_RE.match(text) or _DISTRESS_RE.search(text):
        return "concept_chat"
    if _ADVANCED_RE.search(text):
        return "jee_advanced"
    if _DOUBT_RE.search(text):
        return "doubt_solve"
    if previous_route in ("doubt_solve", "jee_advanced") and _SHORT_FOLLOWUP_RE.match(text):
        return previous_route
    if _CONCEPT_RE.match(text):
        return "concept_chat"

    if api_key and len(text) > 10:
        try:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=text)])]
            res, _ = generate("utility", contents, INTENT_CLASSIFIER_PROMPT, api_key)
            if res:
                cleaned = res.strip().strip('"\'').lower()
                for r in _VALID_ROUTES:
                    if r in cleaned:
                        return r
        except Exception as e:
            print(f"[intent classifier error]: {e}")

    return "doubt_solve"
