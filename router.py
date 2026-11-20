"""
router.py — Multi-Model Router with Developer Instruction & 503/429 Auto-Healing.
Fixes:
  - 400 Developer Instruction invalid format
  - 503 Model High Demand / Unavailable fallback
  - 429 Rate Limit exponential backoff
"""

import json
import random
import re
import threading
import time
from google import genai
from google.genai import types

from prompts import INTENT_CLASSIFIER_PROMPT

# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL CASCADES
# ═══════════════════════════════════════════════════════════════════════
MODEL_CASCADES = {
    "utility": [
        {"model": "gemini-3.5-flash-lite", "thinking": "minimal", "code_exec": False},
        {"model": "gemini-3.1-flash-lite", "thinking": "minimal", "code_exec": False},
        {"model": "gemini-2.5-flash",      "thinking": "low",     "code_exec": False},
        {"model": "gemini-2.0-flash",      "thinking": None,      "code_exec": False},
    ],
    "concept_chat": [
        {"model": "gemini-3.8-flash",      "thinking": "low",     "code_exec": False},
        {"model": "gemini-2.5-flash",      "thinking": "low",     "code_exec": False},
        {"model": "gemini-2.0-flash",      "thinking": None,      "code_exec": False},
    ],
    "doubt_solve": [
        {"model": "gemini-4-hydrogen",     "thinking": "high",    "code_exec": True},
        {"model": "gemini-3.8-flash",      "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.5-pro",        "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.5-flash",      "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.0-flash",      "thinking": None,      "code_exec": False},
    ],
    "test_generate": [
        {"model": "gemini-4-hydrogen",     "thinking": "high",    "code_exec": False},
        {"model": "gemini-3.8-flash",      "thinking": "high",    "code_exec": False},
        {"model": "gemini-2.5-pro",        "thinking": "high",    "code_exec": False},
        {"model": "gemini-2.5-flash",      "thinking": "high",    "code_exec": False},
        {"model": "gemini-2.0-flash",      "thinking": None,      "code_exec": False},
    ],
    "test_verify": [
        {"model": "gemini-3.8-flash",      "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.5-pro",        "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.5-flash",      "thinking": "high",    "code_exec": True},
    ],
    "jee_advanced": [
        {"model": "gemini-4-hydrogen",     "thinking": "xhigh",   "code_exec": True},
        {"model": "gemini-3.8-flash",      "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.5-pro",        "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.5-flash",      "thinking": "high",    "code_exec": True},
    ],
}

DEFAULT_PARAMS = {
    "utility":        {"max_output_tokens": 120,  "temperature": 0.0},
    "concept_chat":   {"max_output_tokens": 1500, "temperature": 0.7},
    "doubt_solve":    {"max_output_tokens": 4000, "temperature": 0.3},
    "test_generate":  {"max_output_tokens": 3000, "temperature": 0.4},
    "test_verify":    {"max_output_tokens": 2000, "temperature": 0.1},
    "jee_advanced":   {"max_output_tokens": 6000, "temperature": 0.2},
}

_LEVEL_BUDGETS = {
    "minimal": 128,
    "low":     512,
    "medium":  1024,
    "high":    2048,
    "xhigh":   4096,
}

# ═══════════════════════════════════════════════════════════════════════
# 2. CLIENT CACHE
# ═══════════════════════════════════════════════════════════════════════
_clients = {}
_clients_lock = threading.Lock()
DEFAULT_TIMEOUT_MS = 25_000

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

def _build_thinking(level: str | None, model_id: str):
    if not level:
        return None
    if "2.0" in model_id:
        return None
    if level == "minimal" and "flash-lite" not in model_id:
        level = "low"
    try:
        return types.ThinkingConfig(thinking_level=level.upper())
    except Exception:
        pass
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

def _is_fatal(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in ("401", "403", "api_key_invalid", "permission_denied", "unauthenticated"))

def _is_rate_limit(err: str) -> bool:
    low = (err or "").lower()
    return "429" in low or "quota" in low or "resource_exhausted" in low

def _is_unavailable_503(err: str) -> bool:
    low = (err or "").lower()
    return "503" in low or "unavailable" in low or "high demand" in low or "overloaded" in low

# ═══════════════════════════════════════════════════════════════════════
# 3. ROBUST INVOCATION (Handles developer_instruction format & 503/429)
# ═══════════════════════════════════════════════════════════════════════
def _call_single_model(client, model_id: str, contents, system_instruction: str,
                       thinking_level: str | None, code_exec: bool,
                       max_tokens: int, temp: float, json_schema=None, json_mode=False):
    """
    Attempts calling the model. Tries both string and types.Content for system_instruction
    to prevent '400 Developer instruction' errors.
    """
    # Try different representations of system_instruction:
    # 1) Plain string
    # 2) Structured types.Content
    # 3) None (prepend to first user message if model rejects system instruction altogether)
    sys_options = [
        system_instruction,
        types.Content(parts=[types.Part.from_text(text=system_instruction)]) if system_instruction else None,
        None
    ]

    for sys_inst in sys_options:
        effective_contents = contents
        if sys_inst is None and system_instruction:
            # Prepend system prompt to the user contents as fallback
            prefixed = [types.Content(role="user", parts=[types.Part.from_text(text=f"[System Instructions: {system_instruction}]\n\n")])]
            effective_contents = prefixed + list(contents)

        for try_thinking in (thinking_level, None):
            for try_tools in (code_exec, False):
                kw = {
                    "temperature": temp,
                    "max_output_tokens": max_tokens,
                }
                if sys_inst is not None:
                    kw["system_instruction"] = sys_inst

                tc = _build_thinking(try_thinking, model_id)
                if tc:
                    kw["thinking_config"] = tc
                tools = _get_tools(try_tools)
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
                        contents=effective_contents,
                        config=config
                    )
                    text = getattr(resp, "text", None)
                    if text:
                        return text, None
                except Exception as e:
                    err = str(e)
                    low = err.lower()

                    if _is_fatal(err):
                        return None, err
                    if _is_unavailable_503(err) or _is_rate_limit(err) or "404" in low or "not_found" in low:
                        return None, err

                    # If error is about developer instruction, break to next sys_inst variant
                    if "developer instruction" in low:
                        break

                    # If error is about thinking or tools, loop will retry with None
                    if "thinking" in low or "tool" in low or "invalid_argument" in low:
                        continue

                    return None, err

    return None, "empty_response"

# ═══════════════════════════════════════════════════════════════════════
# 4. PUBLIC: generate()
# ═══════════════════════════════════════════════════════════════════════
def generate(route_type: str, contents, system_instruction: str, api_key: str,
             json_schema=None, **overrides):
    """
    Cascades through models for this route_type.
    If gemini-3.8-flash returns 503 UNAVAILABLE, automatically cascades to 2.5-flash / 2.0-flash!
    """
    if not api_key:
        return None, "no_key"

    cascade = MODEL_CASCADES.get(route_type)
    if not cascade:
        return None, f"unknown_route: {route_type}"

    params = DEFAULT_PARAMS.get(route_type, {"max_output_tokens": 2000, "temperature": 0.4})
    max_tokens = overrides.get("max_output_tokens", params["max_output_tokens"])
    temp = overrides.get("temperature", params["temperature"])
    json_mode = overrides.get("json_mode", False)

    try:
        client = _get_client(api_key, overrides.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    except Exception as exc:
        return None, f"client_init: {exc}"

    last_error = None
    for step in cascade:
        model_id = step["model"]
        thinking = overrides.get("thinking_level", step["thinking"])
        code_exec = overrides.get("code_execution", step["code_exec"])

        # Try up to 2 attempts for 429 rate limit
        for retry in range(2):
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
                json_mode=json_mode
            )
            if text:
                return text, None

            last_error = err
            if _is_fatal(err):
                return None, err

            if _is_rate_limit(err) and retry == 0:
                time.sleep(1.5)
                continue
            break

        print(f"[router] {model_id} failed ({str(last_error)[:100]}). Cascading to next model...")

    return None, last_error

def friendly_error(err: str) -> str:
    if err == "no_key":
        return "Nino isn't configured yet. Please add a Gemini key in Settings or try later."
    if err and _is_fatal(err):
        return "That Gemini API key looks invalid or lacks permission. Please check it in Settings."
    if err and _is_rate_limit(err):
        return "Nino is receiving heavy traffic (daily AI quota reached). Please wait a moment or add your own key in Settings."
    if err and _is_unavailable_503(err):
        return "Google's preview model is currently overloaded. Nino is rerouting your request, please tap send again!"
    return f"Nino encountered an error: {str(err)[:80]}. Please try sending your message again."

# ═══════════════════════════════════════════════════════════════════════
# 5. INTENT CLASSIFIER (Stage 1 Deterministic -> Stage 2 AI Enum)
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

    if api_key and len(text) > 8:
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
