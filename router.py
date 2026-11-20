"""
router.py — Centralised multi-model router with Rate-Limit Backoff (429 Retries).
Exact models from AI Studio:
  - gemini-3.5-flash-lite
  - gemini-3.8-flash
  - gemini-4-hydrogen
  - gemini-3.1-flash-lite / gemini-2.5-flash
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
# 1. EXACT MODEL ROUTING TABLE
# ═══════════════════════════════════════════════════════════════════════
NINO_MODEL_ROUTER = {
    "utility": {
        "primary":  "gemini-3.5-flash-lite",
        "fallback": "gemini-3.1-flash-lite",
        "thinking": "minimal",
        "code_exec": False,
        "max_tokens": 120,
        "temperature": 0.0,
    },
    "concept_chat": {
        "primary":  "gemini-3.8-flash",
        "fallback": "gemini-2.5-flash",
        "thinking": "low",
        "code_exec": False,
        "max_tokens": 1500,
        "temperature": 0.7,
    },
    "doubt_solve": {
        "primary":  "gemini-4-hydrogen",
        "fallback": "gemini-3.8-flash",
        "thinking": "high",
        "code_exec": True,
        "max_tokens": 4000,
        "temperature": 0.3,
    },
    "test_generate": {
        "primary":  "gemini-4-hydrogen",
        "fallback": "gemini-3.8-flash",
        "thinking": "high",
        "code_exec": False,
        "max_tokens": 3000,
        "temperature": 0.4,
    },
    "test_verify": {
        "primary":  "gemini-3.8-flash",
        "fallback": "gemini-2.5-flash",
        "thinking": "high",
        "code_exec": True,
        "max_tokens": 2000,
        "temperature": 0.1,
    },
    "jee_advanced": {
        "primary":  "gemini-4-hydrogen",
        "fallback": "gemini-3.8-flash",
        "thinking": "xhigh",
        "code_exec": True,
        "max_tokens": 6000,
        "temperature": 0.2,
    },
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
            except Exception:
                client = genai.Client(api_key=api_key)
            _clients[cache_key] = client
        return client

def _build_thinking_config(level: str, model_id: str):
    if not level:
        return None
    # Never pass 'minimal' to 3.8-flash or 4-hydrogen (causes API 400 validation error)
    if level == "minimal" and "flash-lite" not in model_id:
        level = "low"
    try:
        return types.ThinkingConfig(thinking_level=level.upper())
    except (TypeError, ValueError, AttributeError):
        pass
    budget = _LEVEL_BUDGETS.get(level.lower(), 1024)
    try:
        return types.ThinkingConfig(thinking_budget=budget)
    except (TypeError, ValueError, AttributeError):
        return None

def _get_tools(code_exec: bool):
    if not code_exec:
        return None
    # Valid Python SDK tool invocation
    for cls_name in ("ToolCodeExecution", "CodeExecution"):
        cls = getattr(types, cls_name, None)
        if cls:
            try:
                return [types.Tool(code_execution=cls())]
            except Exception:
                pass
    return [{"code_execution": {}}]

# ═══════════════════════════════════════════════════════════════════════
# 3. SINGLE MODEL EXECUTION WITH AUTO 429 BACKOFF
# ═══════════════════════════════════════════════════════════════════════
def _is_rate_limit(err: str) -> bool:
    low = (err or "").lower()
    return "429" in low or "quota" in low or "resource_exhausted" in low

def _is_fatal(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in ("401", "403", "api_key_invalid", "permission_denied", "unauthenticated"))

def _invoke_with_backoff(client, model_id: str, contents, system_instruction: str,
                         thinking_level: str | None, code_exec: bool,
                         max_tokens: int, temp: float, json_schema=None, json_mode=False):
    """
    Calls the model. If 429 occurs, backs off briefly (up to 2 retries) to respect 5 RPM.
    If thinking or tools trigger a 400 validation error, strips them safely and retries.
    """
    for attempt in range(3):
        # 1. Build config
        kw = {
            "system_instruction": system_instruction,
            "temperature": temp,
            "max_output_tokens": max_tokens,
        }
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

        try:
            config = types.GenerateContentConfig(**kw)
            resp = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )
            text = getattr(resp, "text", None)
            if text:
                return text, None
        except Exception as e:
            err = str(e)
            low = err.lower()

            # Fatal auth error -> stop immediately
            if _is_fatal(err):
                return None, err

            # Rate limit -> back off 1.5s - 3s
            if _is_rate_limit(err):
                if attempt < 2:
                    wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    print(f"[{model_id} 429 Rate Limit]: waiting {wait_time:.1f}s (attempt {attempt+1}/3)...")
                    time.sleep(wait_time)
                    continue
                return None, err

            # Validation error with thinking or tools -> retry without them
            if ("thinking" in low or "tool" in low or "invalid_argument" in low) and (thinking_level or code_exec):
                thinking_level = None
                code_exec = False
                continue

            return None, err

    return None, "empty_response"

# ═══════════════════════════════════════════════════════════════════════
# 4. PUBLIC: generate()
# ═══════════════════════════════════════════════════════════════════════
def generate(route_type: str, contents, system_instruction: str, api_key: str,
             json_schema=None, **overrides):
    """
    Route to the primary model. If primary model fails or is rate-limited beyond backoff,
    falls back cleanly to the backup model.
    """
    if not api_key:
        return None, "no_key"

    cfg = NINO_MODEL_ROUTER.get(route_type)
    if not cfg:
        return None, f"unknown_route: {route_type}"

    try:
        client = _get_client(api_key, overrides.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    except Exception as exc:
        return None, f"client_init: {exc}"

    primary_model = cfg["primary"]
    fallback_model = cfg["fallback"]
    thinking = overrides.get("thinking_level", cfg["thinking"])
    code_exec = overrides.get("code_execution", cfg["code_exec"])
    max_tokens = overrides.get("max_output_tokens", cfg["max_tokens"])
    temp = overrides.get("temperature", cfg["temperature"])
    json_mode = overrides.get("json_mode", False)

    # 1. Try Primary Model (with rate-limit backoff)
    text, err = _invoke_with_backoff(
        client=client,
        model_id=primary_model,
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

    if err and _is_fatal(err):
        return None, err

    # 2. Try Fallback Model
    print(f"[router] {primary_model} hit issue ({str(err)[:100]}). Falling back to {fallback_model}...")
    text2, err2 = _invoke_with_backoff(
        client=client,
        model_id=fallback_model,
        contents=contents,
        system_instruction=system_instruction,
        thinking_level="low" if thinking in ("high", "xhigh") else thinking,
        code_exec=code_exec,
        max_tokens=max_tokens,
        temp=temp,
        json_schema=json_schema,
        json_mode=json_mode
    )
    if text2:
        return text2, None

    return None, err2 or err

def friendly_error(err: str) -> str:
    if err == "no_key":
        return "Nino isn't configured yet. Please add a Gemini key in Settings or try later."
    if err and _is_fatal(err):
        return "That Gemini API key looks invalid or lacks permission. Please check it in Settings."
    if err and _is_rate_limit(err):
        return "Nino is receiving heavy traffic (AI quota limit reached on this model). Please wait ~15-30 seconds and send again, or paste your personal Gemini key in Settings."
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

    # Stage 1: Deterministic rules (instant, saves tokens and quota)
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

    # Sticky follow-up check
    if previous_route in ("doubt_solve", "jee_advanced") and _SHORT_FOLLOWUP_RE.match(text):
        return previous_route

    # Stage 2: AI Enum Classifier for ambiguous queries
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
