"""
test_engine.py — Zero-Error Practice Test Generation Pipeline for Nino.

Features:
  - Small parallel batches (BATCH_SIZE = 3) for fast generation (4-8s) without Vercel timeouts
  - Solve-first structured schema (derivation and final verified answer before options)
  - Zero phantom diagram enforcement (strict text-solvable validation)
  - Option shuffling & correct_index realignment for uniform distractor distribution
  - Parse test request, build chapter specs across Tier 1 & Tier 2, and parallel batch executor
  - Optional 2nd-pass verifier via route 'test_verify'
"""

from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait
import json
import math
import random
import re

from prompts import (
    CHAPTER_FOCUS,
    TEST_GENERATE_PROMPT,
    TEST_VERIFY_PROMPT,
)
from router import generate

# ═══════════════════════════════════════════════════════════════════════
# 1. CHAPTER TIERS, TWISTS, AND FORMAT SPECIFICATIONS
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

# Batch sizing: 2-3 questions per call completes in ~4-8s, preventing Vercel 504 timeouts
BATCH_SIZE = 3
TEST_DEADLINE_S = 60
_test_pool = ThreadPoolExecutor(max_workers=4)

# ═══════════════════════════════════════════════════════════════════════
# 2. SOLVE-FIRST STRUCTURED SCHEMA
# ═══════════════════════════════════════════════════════════════════════
# Property order forces the model to derive BEFORE emitting options:
# subject -> chapter -> question_latex -> step_by_step_derivation
# -> final_verified_answer -> options -> correct_index -> mistakeType
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
# 3. ZERO PHANTOM DIAGRAM RULES & OPTION REALIGNMENT
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
    """
    Validates and cleans the question LaTeX string.
    Ensures zero phantom diagrams: returns None if the question relies on an external visual diagram
    without self-contained textual description.
    """
    if not text:
        return None
    cleaned = text.strip()
    if len(cleaned) < 25:
        return None
    for pat in _PHANTOM_DIAGRAM_PATTERNS:
        if pat.search(cleaned):
            # Rejects question relying on phantom diagrams
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

    # Don't shuffle if options are statement combinations or giveaways
    if any(_GIVEAWAY.search(o) for o in clean_opts) or sum(bool(_STATEMENT_OPT.match(o)) for o in clean_opts) >= 3:
        order = list(range(4))
    elif all(v is not None for v in vals):
        # Numerical options sorted ascending for clean presentation
        order = sorted(range(4), key=lambda i: vals[i])
    else:
        order = list(range(4))
        random.shuffle(order)

    shuffled = [clean_opts[i] for i in order]
    new_idx = order.index(correct_idx)
    return shuffled, new_idx

def _scoped_focus_block(specs) -> str:
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
# 4. PARSER & SPEC GENERATOR
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
please pls paper and class level today tomorrow now quick short full another new some can you do me a favor""".split()) | set(NUMBER_WORDS)
_NOT_A_COUNT = {"class", "grade", "std", "chapter", "ch", "unit", "paper", "set", "shift", "day", "year", "jee", "session"}

def parse_test_request(text):
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

def build_specs(count, subject, hint):
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
        t1, t2 = TIER1[subject][:], TIER2[subject][:]
        random.shuffle(t1)
        random.shuffle(t2)
        n2 = min(len(t2), round(count * 0.35))
        topics = [t1[i % len(t1)] for i in range(count - n2)] + [t2[i % len(t2)] for i in range(n2)]
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

def chunk_specs(specs, size=BATCH_SIZE):
    k = max(1, math.ceil(len(specs) / size))
    base, extra = divmod(len(specs), k)
    out, i = [], 0
    for b in range(k):
        n = base + (1 if b < extra else 0)
        out.append(specs[i:i + n])
        i += n
    return out

# ═══════════════════════════════════════════════════════════════════════
# 5. 2ND-PASS VERIFIER (Optional, modular auditor)
# ═══════════════════════════════════════════════════════════════════════
def verify_question_key(cand: dict, api_key: str) -> bool:
    """Independent solve-and-match check on gemini-3.8-flash with code execution."""
    if not api_key:
        return True
    try:
        prompt = (
            f"Question:\n{cand['question']}\n\n"
            f"Options:\n" + "\n".join(f"({i}) {opt}" for i, opt in enumerate(cand["options"])) +
            "\n\nSolve this question completely using Python sympy/math to verify the correct option."
        )
        contents = [prompt]
        text, err = generate("test_verify", contents, TEST_VERIFY_PROMPT, api_key, json_mode=True)
        if not text:
            return True
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            res = json.loads(match.group(0))
            results = res.get("results") or []
            if results and isinstance(results, list):
                item = results[0]
                ans = item.get("verified_answer")
                conf = str(item.get("confidence", "")).lower()
                if ans == cand["correct"] and conf == "high":
                    return True
                if ans != -1 and ans != cand["correct"]:
                    return False
        return True
    except Exception:
        return True

# ═══════════════════════════════════════════════════════════════════════
# 6. BATCH GENERATION & PARALLEL TEST BUILDER
# ═══════════════════════════════════════════════════════════════════════
def generate_batch(specs, api_key, run_verifier=False):
    lines = "\n".join(
        f"{i}. Subject: {s['subject']} | Topic: {s['topic']} | Level: {s['level']} | "
        f"Format: {s['format']} | Required Twist: {s['twist']}"
        for i, s in enumerate(specs, 1)
    )
    prompt = (
        f"Generate exactly {len(specs)} original JEE MCQs matching these specifications:\n{lines}\n"
        "Remember: Derivation must come first. Text-only solvable. Output strict JSON conforming to schema."
    )
    system_inst = TEST_GENERATE_PROMPT + _scoped_focus_block(specs)
    contents = [prompt]

    text, err = generate(
        "test_generate", contents, system_inst, api_key,
        json_schema=TEST_BATCH_RESPONSE_SCHEMA
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

        # Option realigning / shuffling
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

        if run_verifier:
            if not verify_question_key(item, api_key):
                continue
        valid_qs.append(item)

    return valid_qs, None

def build_test_parallel(count, subject, hint, api_key, run_verifier=False):
    specs = build_specs(count, subject, hint)
    batches = chunk_specs(specs, size=BATCH_SIZE)

    futures = [_test_pool.submit(generate_batch, b, api_key, run_verifier) for b in batches]
    done, _ = _futures_wait(futures, timeout=TEST_DEADLINE_S)

    all_questions = []
    err = None
    for f in futures:
        if f in done:
            try:
                qs, e = f.result()
                all_questions.extend(qs)
                err = err or e
            except Exception as exc:
                err = err or str(exc)
        else:
            err = err or "batch_timeout"

    # Deduplication
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
