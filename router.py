"""
router.py — Multi-Model Router with Developer Instruction & 503/429 Auto-Healing for Nino.

Handles:
  - Multi-tier cascading across 6 routes
  - Developer instruction format auto-healing
  - 503 / 504 / 404 instant cascading
  - 429 rate limit exponential backoff with jitter
  - Thinking level safety (minimal only for flash-lite, fallback to budget)
  - Code execution tool fallback
  - Fast 2-stage Intent Classification with sticky multi-turn context
"""

import json
import random
import re
import threading
import time
from google import genai
from google.genai import types

try:
    from prompts import INTENT_CLASSIFIER_PROMPT
except ImportError:
    INTENT_CLASSIFIER_PROMPT = r"""Classify this student message into exactly one category.

Categories:
- concept_chat: Pure theory questions, definitions, conceptual explanations, study tips, comparisons between concepts, motivation, greetings, casual talk. No numbers to crunch.
- doubt_solve: Any message containing a specific problem to solve, numerical calculation, equation, derivation, reaction mechanism, graph interpretation, or "solve / evaluate / find / calculate / prove" intent. Also: homework help, specific formula application, error analysis of a specific problem.
- jee_advanced: Explicitly mentions Irodov, Pathfinder, HCV, Olympiad, JEE Advanced, "prove rigorously", multi-concept problems, or uses language suggesting competition-level depth. Also: requests for multiple solution methods, formal proofs, or edge-case analysis.

Rules:
- If the message contains ANY equation, integral, derivative, specific numbers to compute, or asks to "solve" / "find" / "calculate", classify as doubt_solve (not concept_chat).
- If unsure between concept_chat and doubt_solve, choose doubt_solve (accuracy over speed).
- Short follow-ups like "why?", "explain step 3", "I don't get it" should match the PREVIOUS turn's category."""

# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL CASCADES
# ═══════════════════════════════════════════════════════════════════════
MODEL_CASCADES = {
    "utility": [
        {"model": "gemini-3.5-flash-lite", "thinking": "minimal", "code_exec": False},
        {"model": "gemini-3.1-flash-lite", "thinking": "minimal", "code_exec": False},
        {"model": "gemini-2.5-flash",      "thinking": "low",     "code_exec": False},
    ],
    "concept_chat": [
        {"model": "gemini-3.8-flash",      "thinking": "low",     "code_exec": False},
        {"model": "gemini-2.5-flash",      "thinking": "low",     "code_exec": False},
    ],
    "doubt_solve": [
        {"model": "gemini-4-hydrogen",     "thinking": "high",    "code_exec": True},
        {"model": "gemini-3.8-flash",      "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.5-pro",        "thinking": "high",    "code_exec": True},
        {"model": "gemini-2.5-flash",      "thinking": "high",    "code_exec": True},
    ],
    "test_generate": [
        {"model": "gemini-3.8-flash",      "thinking": "high",    "code_exec": False},
        {"model": "gemini-4-hydrogen",     "thinking": "high",    "code_exec": False},
        {"model": "gemini-2.5-flash",      "thinking": "low",     "code_exec": False},
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
# 2. CLIENT CACHING & CONFIG BUILDERS
# ═══════════════════════════════════════════════════════════════════════
_clients = {}
_clients_lock = threading.Lock()
DEFAULT_TIMEOUT_MS = 50_000

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
    """
    Builds ThinkingConfig safely.
    'minimal' is ONLY passed to models with 'flash-lite' in their name.
    If passed to other models, map it to 'low' to prevent API 400 validation errors.
    Falls back to thinking_budget if thinking_level is unsupported.
    """
    if not level:
        return None
    low_model = (model_id or "").lower()
    if "2.0" in low_model:
        return None

    clean_level = level.lower().strip()
    if clean_level == "minimal" and "flash-lite" not in low_model:
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
    """
    Safely construct types.Tool(code_execution=types.ToolCodeExecution())
    or types.Tool(code_execution=types.CodeExecution()).
    Falls back to None if not supported.
    """
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

def _prepare_contents_with_system(contents, system_instruction: str):
    """
    Prepend system prompt into user contents as fallback when system_instruction
    is rejected by the API.
    """
    if not system_instruction:
        return contents
    prefix_text = f"[System Instructions: {system_instruction}]\n\n"
    if isinstance(contents, str):
        return prefix_text + contents
    if isinstance(contents, list):
        if not contents:
            return [types.Content(role="user", parts=[types.Part.from_text(text=prefix_text)])]
        first = contents[0]
        if isinstance(first, str):
            return [prefix_text + first] + contents[1:]
        elif isinstance(first, types.Content):
            return [types.Content(role="user", parts=[types.Part.from_text(text=prefix_text)])] + list(contents)
        else:
            return [types.Content(role="user", parts=[types.Part.from_text(text=prefix_text)])] + list(contents)
    return contents

# ═══════════════════════════════════════════════════════════════════════
# 3. ERROR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════
def _is_fatal(err: str) -> bool:
    low = (err or "").lower()
    return any(t in low for t in ("401", "403", "api_key_invalid", "permission_denied", "unauthenticated", "invalid api key"))

def _is_rate_limit(err: str) -> bool:
    low = (err or "").lower()
    return "429" in low or "quota" in low or "resource_exhausted" in low or "rate limit" in low

def _is_unavailable_503(err: str) -> bool:
    low = (err or "").lower()
    return "503" in low or "unavailable" in low or "high demand" in low or "overloaded" in low

def _is_timeout_504(err: str) -> bool:
    low = (err or "").lower()
    return "504" in low or "timeout" in low or "deadline_exceeded" in low or "timed out" in low

def _is_not_found_404(err: str) -> bool:
    low = (err or "").lower()
    return "404" in low or "not_found" in low or "not found" in low

def _is_immediate_cascade(err: str) -> bool:
    return _is_unavailable_503(err) or _is_timeout_504(err) or _is_not_found_404(err)

# ═══════════════════════════════════════════════════════════════════════
# 4. SINGLE MODEL INVOCATION
# ═══════════════════════════════════════════════════════════════════════
def _call_single_model(client, model_id: str, contents, system_instruction: str,
                       thinking_level: str | None, code_exec: bool,
                       max_tokens: int, temp: float, json_schema=None, json_mode=False):
    """
    Attempts calling the model. Tries:
      1) Plain string system_instruction
      2) Structured types.Content(parts=[types.Part.from_text(text=system_instruction)])
      3) Prepending system_instruction to user content
    Handles thinking config and code execution fallbacks cleanly.
    """
    modes_to_try = []
    if system_instruction:
        modes_to_try = [
            ("string", system_instruction),
            ("content", types.Content(parts=[types.Part.from_text(text=system_instruction)])),
            ("prefixed", None)
        ]
    else:
        modes_to_try = [("none", None)]

    for mode, sys_inst in modes_to_try:
        effective_contents = contents
        if mode == "prefixed" and system_instruction:
            effective_contents = _prepare_contents_with_system(contents, system_instruction)

        dev_inst_error = False
        thinking_candidates = (thinking_level, None) if thinking_level else (None,)
        tools_candidates = (code_exec, False) if code_exec else (False,)

        for try_thinking in thinking_candidates:
            for try_tools in tools_candidates:
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

                    if _is_fatal(err) or _is_immediate_cascade(err) or _is_rate_limit(err):
                        return None, err

                    # Developer instruction format rejection: break to try next representation
                    if "developer instruction" in low or "developer_instruction" in low:
                        dev_inst_error = True
                        break

                    # Thinking or Tool argument mismatch: continue inner loop without them
                    if "thinking" in low or "tool" in low or "invalid_argument" in low:
                        continue

                    return None, err

            if dev_inst_error:
                break

    return None, "empty_response"

# ═══════════════════════════════════════════════════════════════════════
# 5. PUBLIC: generate()
# ═══════════════════════════════════════════════════════════════════════
def generate(route_type: str, contents, system_instruction: str, api_key: str,
             json_schema=None, **overrides):
    """
    Cascades through models for this route_type.
    Applies auto-healing:
      - 429: exponential backoff with jitter (1.5s - 3s) for up to 2 retries
      - 503 / 504 / 404: immediate cascade to the next model in the list
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

        max_retries = 2
        for attempt in range(max_retries + 1):
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

            if _is_immediate_cascade(err):
                # 503 / 504 / 404: cascade immediately
                break

            if _is_rate_limit(err):
                if attempt < max_retries:
                    # Exponential backoff with jitter (1.5s - 3s)
                    delay = 1.5 * (1.3 ** attempt) + random.uniform(0.0, 1.5)
                    time.sleep(delay)
                    continue
                else:
                    break

            # Other errors: break and try next model
            break

        print(f"[router] {model_id} failed ({str(last_error)[:100]}). Cascading to next model...")

    return None, last_error

# ═══════════════════════════════════════════════════════════════════════
# 6. FRIENDLY ERROR FORMATTER
# ═══════════════════════════════════════════════════════════════════════
def friendly_error(err: str) -> str:
    if not err:
        return "An unknown error occurred. Please try sending your message again."
    err_str = str(err)
    if err_str == "no_key":
        return "Nino isn't configured yet. Please add a Gemini key in Settings or try later."
    if _is_fatal(err_str):
        return "That Gemini API key looks invalid or lacks permission. Please check it in Settings."
    if _is_rate_limit(err_str):
        return "Nino is receiving heavy traffic (daily AI quota reached). Please wait a moment or add your own key in Settings."
    if _is_unavailable_503(err_str) or _is_timeout_504(err_str):
        return "Google's preview model is currently overloaded. Nino is rerouting your request, please tap send again!"
    return f"Nino encountered an error: {err_str[:80]}. Please try sending your message again."

# ═══════════════════════════════════════════════════════════════════════
# 7. INTENT CLASSIFIER (Stage 1 Deterministic -> Stage 2 AI Enum)
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
    """
    Two-stage intent classification:
      Stage 1: Deterministic rules (instant, 0ms)
      Sticky multi-turn: Preserves doubt_solve / jee_advanced for short follow-ups
      Stage 2: Fast classification via utility route
    """
    text = (text or "").strip()

    if is_test_endpoint:
        return "test_generate"
    if deep_solve:
        return "jee_advanced"
    if has_image:
        return "doubt_solve"
    if not text or _GREETING_RE.match(text) or _DISTRESS_RE.search(text):
        return "concept_chat"

    # Sticky multi-turn: Preserve category for short follow-ups like "why?", "step 3"
    if previous_route in ("doubt_solve", "jee_advanced") and _SHORT_FOLLOWUP_RE.match(text):
        return previous_route

    if _ADVANCED_RE.search(text):
        return "jee_advanced"
    if _DOUBT_RE.search(text):
        return "doubt_solve"

    # Stage 2: Fast classification via utility route
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
