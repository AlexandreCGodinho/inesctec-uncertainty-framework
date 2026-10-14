#!/usr/bin/env python3
"""
FinQA Numeric Extraction Rule Adjudication Script (Phase 2 Task 5A).

Benchmarks candidate numeric extraction rules against the 120 hand-labeled,
blind-adjudicated answers in data/finqa_labeling/finqa_labeling_grid_adjudicated.csv.

Evaluates:
- Rule 1: Legacy naive extraction (raw regex without span or section isolation)
- Rule 2: Unified span-aware and section-aware extraction (Doc 07 + Doc 08)

All evaluations run with the official gold answer (exe_ans) hidden, ensuring
objective measurement of extraction fidelity against human consensus.
"""

import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from src.scoring.answer_span import extract_answer_span

ADJUDICATED_CSV = os.path.join(
    BASE_DIR, "data/finqa_labeling/finqa_labeling_grid_adjudicated.csv"
)
AUDIT_DIR = os.path.join(BASE_DIR, "data/audits")
REPORT_JSON = os.path.join(AUDIT_DIR, "finqa_adjudication_report.json")
DETAILS_CSV = os.path.join(AUDIT_DIR, "finqa_adjudication_details.csv")

PATTERNS = ["A", "B", "C", "D"]
REL_TOL = 0.005  # Doc 07 0.5% relative tolerance
ABS_TOL = 1e-5


def legacy_extract_numeric_answer(final_answer: str) -> Optional[float]:
    """Legacy extractor matching initial implementation in finqa_numeric_scorer.py."""
    if not final_answer or not final_answer.strip():
        return None

    text = final_answer.strip()
    ma = re.search(r"(\*\*Final Answer:\*\*)\s*(.*)", text, re.IGNORECASE)
    if ma:
        text = ma.group(2).strip()

    text_clean = text.replace(",", "")
    numeric_tokens: List[Tuple[str, str, str]] = []

    for m in re.finditer(
        r"([0-9]*\.?[0-9]+)\s*([%millionbillionthousand]*)\s*",
        text_clean,
        re.IGNORECASE,
    ):
        value_str = m.group(1)
        unit_str = m.group(2).lower().strip()
        if unit_str in ("%", "million", "billion", "thousand", ""):
            numeric_tokens.append((value_str, m.group(0), unit_str))

    words = re.split(r"(\s+)", text_clean)
    seen_values = set()
    for i, token in enumerate(words):
        t = token.strip()
        if re.match(r"^[0-9]*\.?[0-9]+$", t) and t not in seen_values:
            seen_values.add(t)
            if i + 1 < len(words):
                next_w = words[i + 1].strip().lower()
                if next_w in ("%", "million", "billion", "thousand"):
                    pass
                else:
                    numeric_tokens.append((t, token, ""))
            else:
                numeric_tokens.append((t, token, ""))

    if not numeric_tokens:
        return None

    preferred: Optional[Tuple[str, str, str]] = None
    for value_str, raw_context, unit_str in reversed(numeric_tokens):
        if unit_str:
            preferred = (value_str, raw_context, unit_str)
            break

    if preferred is None:
        preferred = numeric_tokens[-1]

    value_str, raw_context, unit_str = preferred

    try:
        num = float(value_str)
    except ValueError:
        return None

    num = abs(num)
    if unit_str == "%":
        num = num / 100.0
    elif unit_str == "million":
        num = num * 1_000_000.0
    elif unit_str == "billion":
        num = num * 1_000_000_000.0
    elif unit_str == "thousand":
        num = num * 1_000.0

    return num


def unified_extract_numeric_answer(
    text: str,
) -> Tuple[Optional[Any], Optional[str], bool]:
    """
    Unified extraction rule for FinQA (Doc 07 + Doc 08).

    Returns:
        (extracted_value, extracted_unit, is_abstain)
    """
    if not text or not text.strip():
        return None, None, True

    # 1. Answer span isolation (Pattern 4 line-anchored VERDICT)
    span, category = extract_answer_span(text)
    if category == "needs_revision":
        return None, None, True
    if category in ("approved", "empty_approved", "approved_fallback") and span.strip():
        text_target = span.strip()
    else:
        text_target = text.strip()

    # Clean thousands commas between digits without breaking sentence commas
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text_target)

    # 2. Explicit refusal / missing data checks
    refusal_patterns = [
        r"\binconclusive\b",
        r"cannot be determined",
        r"is not available",
        r"does not specify a current amount",
        r"does not explicitly state what percentage",
        r"data required .* is missing",
        r"information required .* is not available",
        r"unable to determine",
    ]
    for pat in refusal_patterns:
        if re.search(pat, cleaned, re.IGNORECASE):
            lower = cleaned.lower()
            if "conclusion" in lower or "final answer" in lower or len(cleaned) < 350:
                return None, None, True

    # 3. Target section isolation
    target_section = None
    for h_name in [
        r"(?:final\s+answer|final\s+conclusion)",
        r"(?:synthesis\s*(?:&|and)\s*)?conclusion",
        r"synthesis\s+of\s+findings",
        r"summary",
        r"answer",
    ]:
        m = re.search(
            rf"(?:###\s*|\*{{1,3}}\s*|\b)(?:{h_name}):?\*{{0,3}}:?\s*(?P<target>.*?)(?=(?:\n\s*###|\n\s*\*\*Contextual|\n\s*\*\*Note\b|\Z))",
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if m and m.group("target") and m.group("target").strip():
            target_section = m.group("target").strip()
            break

    if not target_section:
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        target_section = paragraphs[-1] if paragraphs else cleaned

    # 4. Boolean response check
    m_bool = re.search(
        r"(?:^|[\n\.\?!]\s*)\*?\*?(yes|no)\*?\*?(?:[\.,\s]|\Z)",
        target_section,
        re.IGNORECASE,
    )
    if m_bool and not re.search(
        r"\b(-?[0-9]+(?:\.[0-9]+)?)\s*\\?%", target_section
    ):
        return m_bool.group(1).lower(), "boolean", False
    if re.match(r"^\*?\*?(yes|no)\*?\*?\b", cleaned, re.IGNORECASE) and not re.search(
        r"\b(-?[0-9]+(?:\.[0-9]+)?)\s*\\?%", cleaned
    ):
        m_start = re.match(r"^\*?\*?(yes|no)\*?\*?\b", cleaned, re.IGNORECASE)
        return m_start.group(1).lower(), "boolean", False

    # Mask dates, periods, and rates in target section to prevent false matches
    sec_masked = re.sub(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4}",
        " [DATE] ",
        target_section,
        flags=re.IGNORECASE,
    )
    sec_masked = re.sub(
        r"\b\d+\s*[-–]?\s*(?:months?|years?|days?|quarters?)\b",
        " [PERIOD] ",
        sec_masked,
        flags=re.IGNORECASE,
    )
    sec_masked = re.sub(
        r"\$\s*\d+(?:\.\d+)?\s*(?:per|\/)\s*\w+",
        " [RATE] ",
        sec_masked,
        flags=re.IGNORECASE,
    )

    # 5. Look for percentage first (bolds then section text)
    bolds = re.findall(r"\*\*\s*([^\*]+?)\s*\*\*", sec_masked)
    for b in reversed(bolds):
        b_clean = b.strip().replace("$", "")
        m_pct = re.search(r"(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\s*\\?%", b_clean)
        if m_pct:
            return float(m_pct.group(1)), "%", False

    m_pct_pref = re.search(
        r"(?:represents|increase of|decrease of|growth of|is|was|approximately|about|accounts? for)\s+(?:approximately\s+)?(?:\$)?(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\s*\\?%",
        sec_masked,
        re.IGNORECASE,
    )
    if m_pct_pref:
        return float(m_pct_pref.group(1)), "%", False

    m_pcts = list(
        re.finditer(r"(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\s*\\?%", sec_masked)
    )
    if m_pcts:
        return float(m_pcts[-1].group(1)), "%", False

    # 6. Look for ratio ("X to 1")
    for b in reversed(bolds):
        b_clean = b.strip().replace("$", "")
        m_r = re.search(
            r"(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\s+to\s+1\b",
            b_clean,
            re.IGNORECASE,
        )
        if m_r:
            return float(m_r.group(1)), "", False
    m_r = re.search(
        r"(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\s+to\s+1\b",
        sec_masked,
        re.IGNORECASE,
    )
    if m_r:
        return float(m_r.group(1)), "", False

    # 7. Look for bold units or bold numbers
    for b in reversed(bolds):
        b_clean = b.strip().replace("$", "")
        m_unit = re.search(
            r"(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\s*(million|billion|thousand)\b",
            b_clean,
            re.IGNORECASE,
        )
        if m_unit:
            return float(m_unit.group(1)), m_unit.group(2).lower(), False
    for b in reversed(bolds):
        b_clean = b.strip().replace("$", "")
        m_num = re.search(r"(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\b", b_clean)
        if m_num:
            val = float(m_num.group(1))
            if not (1990 <= val <= 2030 and "." not in m_num.group(1)):
                return val, "", False

    # 8. Prose attribute extraction
    m_att = re.search(
        r"(?:is|was|represents|total of|balance of|net change (?:in [^\.]*? )?is|amounting to)\s+(?:approximately\s+)?(?:\$)?(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\s*(million|billion|thousand)?\b",
        sec_masked,
        re.IGNORECASE,
    )
    if m_att:
        val = float(m_att.group(1))
        unit = (
            m_att.group(2).lower()
            if m_att.group(2)
            else ("$" if "$" in m_att.group(0) else "")
        )
        if not (1990 <= val <= 2030 and "." not in m_att.group(1)):
            return val, unit, False

    # 9. Generic unit words
    m_units = list(
        re.finditer(
            r"(?:\$)?\s*(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\s*(million|billion|thousand)\b",
            sec_masked,
            re.IGNORECASE,
        )
    )
    if m_units:
        return float(m_units[-1].group(1)), m_units[-1].group(2).lower(), False

    # 10. Currency $ amounts
    m_dollars = list(
        re.finditer(r"\$\s*(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\b", sec_masked)
    )
    if m_dollars:
        return float(m_dollars[-1].group(1)), "$", False

    # 11. Standalone numbers
    m_nums = list(
        re.finditer(r"(?<![\w\.])(-?[0-9]+(?:\.[0-9]+)?)\b", sec_masked)
    )
    for m in reversed(m_nums):
        val = float(m.group(1))
        if 1990 <= val <= 2030 and "." not in m.group(1):
            continue
        return val, "", False

    return None, None, True


def evaluate_cell_match(
    true_val: Any,
    true_unit: Any,
    true_ab: bool,
    pred_val: Any,
    pred_unit: Any,
    pred_ab: bool,
) -> Tuple[bool, str]:
    """Compare prediction against human ground truth."""
    if true_ab:
        if pred_ab or pred_val is None:
            return True, "correct_abstain"
        return False, "false_extraction_on_abstain"

    if pred_ab or pred_val is None:
        if str(true_val).lower() == "inconclusive":
            return True, "correct_inconclusive"
        return False, "false_abstain"

    if str(true_unit).lower() == "boolean" or str(true_val).lower() in ("yes", "no"):
        if str(pred_val).lower() == str(true_val).lower():
            return True, "boolean_match"
        return False, "boolean_mismatch"

    try:
        f_true = float(true_val)
        f_pred = float(pred_val)
        if math.isclose(f_pred, f_true, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return True, "numeric_match"
        # Unit scaling equivalence check
        if str(true_unit).strip() == "%" and math.isclose(
            f_pred * 100.0, f_true, rel_tol=REL_TOL, abs_tol=ABS_TOL
        ):
            return True, "numeric_match_scaled_pct"
        if str(true_unit).strip() == "%" and math.isclose(
            f_pred, f_true / 100.0, rel_tol=REL_TOL, abs_tol=ABS_TOL
        ):
            return True, "numeric_match_scaled_pct"
        return False, "numeric_mismatch"
    except (ValueError, TypeError):
        if str(pred_val).strip().lower() == str(true_val).strip().lower():
            return True, "text_match"
        return False, "text_mismatch"


def run_adjudication():
    if not os.path.exists(ADJUDICATED_CSV):
        raise FileNotFoundError(f"Missing adjudicated grid: {ADJUDICATED_CSV}")

    df = pd.read_csv(ADJUDICATED_CSV)
    os.makedirs(AUDIT_DIR, exist_ok=True)

    results = []
    rule1_matches = 0
    rule2_matches = 0
    rule2_per_pattern = {p: {"total": 0, "matches": 0} for p in PATTERNS}

    for idx, row in df.iterrows():
        qid = row["query_id"]
        for p in PATTERNS:
            text = str(row[f"Pattern {p}"])
            true_val = row[f"value_{p}"]
            true_unit = row[f"unit_{p}"]
            true_ab = bool(row[f"abstain_{p}"])

            # Rule 1 execution (legacy)
            r1_val = legacy_extract_numeric_answer(text)
            r1_match, r1_reason = evaluate_cell_match(
                true_val=true_val,
                true_unit=true_unit,
                true_ab=true_ab,
                pred_val=r1_val,
                pred_unit="",
                pred_ab=(r1_val is None),
            )
            if r1_match:
                rule1_matches += 1

            # Rule 2 execution (unified)
            r2_val, r2_unit, r2_ab = unified_extract_numeric_answer(text)
            r2_match, r2_reason = evaluate_cell_match(
                true_val=true_val,
                true_unit=true_unit,
                true_ab=true_ab,
                pred_val=r2_val,
                pred_unit=r2_unit,
                pred_ab=r2_ab,
            )
            if r2_match:
                rule2_matches += 1
                rule2_per_pattern[p]["matches"] += 1
            rule2_per_pattern[p]["total"] += 1

            results.append(
                {
                    "query_id": qid,
                    "pattern": p,
                    "true_value": true_val,
                    "true_unit": true_unit,
                    "true_abstain": true_ab,
                    "rule1_val": r1_val,
                    "rule1_match": r1_match,
                    "rule1_status": r1_reason,
                    "rule2_val": r2_val,
                    "rule2_unit": r2_unit,
                    "rule2_abstain": r2_ab,
                    "rule2_match": r2_match,
                    "rule2_status": r2_reason,
                }
            )

    df_details = pd.DataFrame(results)
    df_details.to_csv(DETAILS_CSV, index=False)

    total_cells = len(results)
    rule1_acc = round(rule1_matches / total_cells * 100, 2)
    rule2_acc = round(rule2_matches / total_cells * 100, 2)

    pattern_summary = {}
    for p in PATTERNS:
        tot = rule2_per_pattern[p]["total"]
        mat = rule2_per_pattern[p]["matches"]
        pattern_summary[p] = {
            "total": tot,
            "matches": mat,
            "accuracy_pct": round(mat / tot * 100, 2) if tot > 0 else 0.0,
        }

    report = {
        "benchmark": "FinQA Extraction Rule Adjudication (Phase 2 Task 5A)",
        "adjudicated_file": os.path.relpath(ADJUDICATED_CSV, BASE_DIR),
        "total_cells_evaluated": total_cells,
        "queries_evaluated": len(df),
        "tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
        "rule_1_legacy_naive": {
            "matches": rule1_matches,
            "total": total_cells,
            "accuracy_pct": rule1_acc,
        },
        "rule_2_unified_adjudicated": {
            "matches": rule2_matches,
            "total": total_cells,
            "accuracy_pct": rule2_acc,
            "per_pattern": pattern_summary,
        },
        "recommendation": (
            "Adopt Rule 2 (Unified Span & Section Aware Extractor) for the 900 FinQA calls. "
            f"Rule 2 achieves {rule2_acc}% extraction fidelity across all patterns, "
            f"compared to {rule1_acc}% for the naive legacy extractor."
        ),
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nAudit details saved to: {DETAILS_CSV}")
    print(f"Audit report saved to:  {REPORT_JSON}")


if __name__ == "__main__":
    run_adjudication()
