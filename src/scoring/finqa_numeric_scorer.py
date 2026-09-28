import re
import math
from typing import List, Optional, Set, Tuple


def _extract_numeric_answer(final_answer: str) -> Optional[float]:
    """
    Extract the numeric answer from a FinQA final_answer string.

    Rule (from docs/07 §1):
    1. Take the '**Final Answer:**' section if present, else the full answer.
    2. Select the last numeric token.
    3. Prefer a token carrying a unit word (% , million, billion, thousand).
    4. A trailing bare '1' after 'to'/':' is ignored (ratio "38.6 to 1" → 38.6).

    Returns the parsed numeric value (float), or None if no numeric token found.
    """
    if not final_answer or not final_answer.strip():
        return None

    text = final_answer.strip()

    # Step 1: take '**Final Answer:**' section if present
    ma = re.search(r"(\*\*Final Answer:\*\*)\s*(.*)", text, re.IGNORECASE)
    if ma:
        text = ma.group(2).strip()

    # Step 2: find ALL numeric tokens with their context.
    # We'll look for numbers and check what unit word follows each.
    # First, remove thousand separators (commas) from numbers to allow float conversion
    text_clean = text.replace(",", "")

    numeric_tokens: List[Tuple[str, str, str]] = []  # (value_str, raw_context, unit_str)

    # Find number + optional unit word patterns
    # Pattern: NUMBER followed by optional unit (% / million / billion / thousand)
    for m in re.finditer(r"([0-9]*\.?[0-9]+)\s*([%millionbillionthousand]*)\s*", text_clean, re.IGNORECASE):
        value_str = m.group(1)
        unit_str = m.group(2).lower().strip()
        # Only keep recognized unit words
        if unit_str in ("%", "million", "billion", "thousand", ""):
            numeric_tokens.append((value_str, m.group(0), unit_str))

    # Also find standalone numbers - any digit sequence not followed by a unit word
    # Split text into words and find digit-only words
    words = re.split(r"(\s+)", text_clean)
    seen_values = set()
    for i, token in enumerate(words):
        t = token.strip()
        if re.match(r"^[0-9]*\.?[0-9]+$", t) and t not in seen_values:
            seen_values.add(t)
            # Check if the next word is a unit
            if i + 1 < len(words):
                next_w = words[i + 1].strip().lower()
                if next_w in ("%", "million", "billion", "thousand"):
                    # This number has a unit word after it - already captured above
                    pass
                else:
                    # Standalone number - no unit word
                    numeric_tokens.append((t, token, ""))
            else:
                # No next word or not a unit - standalone
                numeric_tokens.append((t, token, ""))

    if not numeric_tokens:
        return None

    # Step 2.5: prefer a token carrying a unit word.
    # Last numeric token with a unit word is preferred.
    preferred: Optional[Tuple[str, str, str]] = None
    for value_str, raw_context, unit_str in reversed(numeric_tokens):
        if unit_str:
            preferred = (value_str, raw_context, unit_str)
            break

    if preferred is None:
        # No unit word found — use the last token
        preferred = numeric_tokens[-1]

    value_str, raw_context, unit_str = preferred

    # Step 2.6: normalize the numeric value.
    try:
        num = float(value_str)
    except ValueError:
        return None

    # Normalization: we already removed commas from text_clean, so value_str should be clean.
    # Just ensure positive value.
    num = abs(num)

    # Apply unit-based scaling
    if unit_str == "%":
        # % → /100
        num = num / 100.0
    elif unit_str == "million":
        # million → ×1e6
        num = num * 1_000_000.0
    elif unit_str == "billion":
        # billion → ×1e9
        num = num * 1_000_000_000.0
    elif unit_str == "thousand":
        # thousand → ×1e3
        num = num * 1_000.0
    # bare number: no scaling (stay in raw table units)

    return num


def _candidates_from_base(base: float, unit_str: str) -> Set[float]:
    """
    Generate candidate values from the base numeric value and the unit word,
    handling table unit ambiguity.

    Rule (docs/07 §3):
    - gold exe_ans is in raw table units, which vary per table (raw / thousands / millions).
    - Candidates = {base, base×1e-3, base×1e-6} when a unit word is present.
    - Plain numbers with no unit word are NOT scaled.
    """
    cands: Set[float] = set()

    if unit_str == "":
        # No unit word — plain number, no scaling
        cands.add(base)
    elif unit_str in ("million", "billion", "thousand"):
        # Table could be in raw, thousands, or millions
        cands.add(base)                # raw table units
        cands.add(base / 1_000.0)      # thousands unit
        cands.add(base / 1_000_000.0)  # millions unit
    elif unit_str == "%":
        # % is already normalized to /100 in _extract_numeric_answer
        cands.add(base)

    return cands


def _is_yes_no(model_answer: str, exe_ans: str) -> bool:
    """
    Check if model_answer matches exe_ans for yes/no questions.
    
    Args:
        model_answer: The model's answer (e.g., from final_answer)
        exe_ans: The expected answer from the reference
    """
    if exe_ans.lower() in ("yes", "no"):
        return model_answer.strip().lower() == exe_ans.lower()
    return False


def finqa_numeric_scorer(
    record: dict,
    reference: dict,
) -> dict:
    """
    FinQA numeric answer scorer.

    Evaluates the telemetry record's final_answer against the reference's exe_ans.

    Rule reference: docs/07_FINQA_NUMERIC_RULE_VALIDATION.md

    Returns dict with at minimum: {'match': 1.0 or 0.0}
    """
    final_answer = record.get("final_answer", "")
    exe_ans = reference.get("exe_ans", reference.get("answer", ""))

    # Step 1/5: yes/no handling — when exe_ans is "yes"/"no", exact case-insensitive
    # token match on the model answer (handles P4 VERDICT: approved Yes).
    if isinstance(exe_ans, str) and exe_ans.lower() in ("yes", "no"):
        if _is_yes_no(final_answer.strip(), exe_ans):
            return {"match": 1.0}
        else:
            return {"match": 0.0}

    # Step 1-4: extract numeric answer and compare
    extracted = _extract_numeric_answer(final_answer)

    if extracted is None:
        return {"match": 0.0}

    # Special ruling: AMT/2014 quirk — model answer 63458 and gold 33458
    # are considered a match per Álvaro's ruling (FinQA quirk: human-typed answer
    # and program-executed exe_ans differ).
    if extracted == 63458.0 and float(exe_ans) == 33458.0:
        return {"match": 1.0}

    # Determine the unit word from the extracted answer by re-examining the text
    text = final_answer.strip()
    text_clean = text.replace(",", "")
    unit_str = ""
    # Check for % first
    if re.search(r"%\s*$", text_clean, re.IGNORECASE):
        unit_str = "%"
    else:
        for m in re.finditer(r"([0-9]*\.?[0-9]+)\s+(million|billion|thousand)\b", text_clean, re.IGNORECASE):
            unit_str = m.group(2).lower()
            break

    # If the model had a % but exe_ans is raw (no %), return 0.0
    # (model omits the % or mismatch)
    if "%" in final_answer and unit_str != "%":
        return {"match": 0.0}

    candidates: Set[float] = _candidates_from_base(extracted, unit_str)

    # Comparison: match if any candidate is within rel_tol=0.005 (math.isclose,
    # abs_tol=1e-6) of exe_ans. This covers rounding (NKE 81.37 → 0.81372) and
    # %-vs-decimal (81.37% → 0.8137).
    exe_ans_float = None
    try:
        exe_ans_float = float(exe_ans) if exe_ans else None
    except (ValueError, TypeError):
        pass

    if exe_ans_float is not None:
        for cand in candidates:
            if math.isclose(cand, exe_ans_float, rel_tol=0.005, abs_tol=1e-6):
                return {"match": 1.0}
        return {"match": 0.0}

    # Fallback: direct comparison (no unit scaling needed)
    if extracted is not None and exe_ans_float is not None:
        if math.isclose(extracted, exe_ans_float, rel_tol=0.005, abs_tol=1e-6):
            return {"match": 1.0}
    return {"match": 0.0}