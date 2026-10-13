import re
from src.scoring.answer_span import extract_answer_span
from src.scoring.ifeval_scorer import _extract_constraints


def compute_u_L(
    final_answer: str,
    constraints: list[dict] | None = None,
    dataset: str | None = None,
    query: str | None = None,
) -> float | None:
    # 1. FinQA and PopQA: formatting schema constraints do not apply (N/A)
    if dataset and dataset.lower() in ("finqa", "popqa"):
        return None

    # 2. LiveBench: verify exclusively the expected answer slot
    if dataset and dataset.lower() == "livebench":
        return _check_livebench_answer_slot(final_answer, query, constraints)

    # 3. IFEval: compute constraints on the extracted answer span
    if dataset and dataset.lower() == "ifeval":
        return _compute_ifeval_u_L(final_answer, query, constraints)

    # 4. Explicit constraints passed (e.g., legacy unit tests)
    if constraints is not None:
        if not constraints:
            return 0.0
        passed = sum(1 for c in constraints if _check_constraint(final_answer, c))
        total = len(constraints)
        return 1.0 - (passed / total) if total > 0 else 0.0

    return 0.0


_HEADER_LABELS = {
    "conclusion",
    "answer",
    "final answer",
    "note",
    "step",
    "summary",
    "solution",
    "result",
    "verdict",
    "explanation",
    "analysis",
}


def _is_header_label(text: str) -> bool:
    cleaned = text.strip().lower().rstrip(":")
    return (
        cleaned in _HEADER_LABELS
        or cleaned.startswith("step ")
        or cleaned.startswith("case ")
    )


def _is_terminal_slot(span: str, match: re.Match) -> bool:
    """Check if a matched span is at the terminal answer slot position."""
    trailing = span[match.end():].strip()
    if not trailing or bool(re.fullmatch(r"^[.\s!,;:)\]]+$", trailing)):
        return True

    lines = [line.strip() for line in span.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if match.group(0) in last_line:
            last_line_tail = last_line.split(match.group(0))[-1].strip()
            if not last_line_tail or bool(re.fullmatch(r"^[.\s!,;:)\]]+$", last_line_tail)):
                return True

    return False


def _extract_livebench_answer_slot(final_answer: str, query: str | None = None) -> str | None:
    """Isolate the terminal answer slot from final_answer using extract_answer_span."""
    if not final_answer or not final_answer.strip():
        return None

    span, _ = extract_answer_span(final_answer)
    span = span.strip()
    if not span:
        return None

    # 1. Format specified by query
    if query:
        if "***" in query:
            matches = list(re.finditer(r"\*\*\*([^*]+?)\*\*\*", span))
            if matches and _is_terminal_slot(span, matches[-1]):
                return matches[-1].group(1).strip()
            return None

        if "<solution>" in query:
            matches = list(re.finditer(r"<solution>(.*?)</solution>", span, re.DOTALL))
            if matches and _is_terminal_slot(span, matches[-1]):
                return matches[-1].group(1).strip()
            return None

        if "**bold**" in query or "in **bold**" in query or "bold" in query.lower():
            matches = list(re.finditer(r"\*\*([^*]+?)\*\*", span))
            for m in reversed(matches):
                if _is_terminal_slot(span, m) and not _is_header_label(m.group(1)):
                    return m.group(1).strip()
            return None

    # 2. General fallback: inspect markers in order of specificity
    star_matches = list(re.finditer(r"\*\*\*([^*]+?)\*\*\*", span))
    if star_matches and _is_terminal_slot(span, star_matches[-1]):
        return star_matches[-1].group(1).strip()

    sol_matches = list(re.finditer(r"<solution>(.*?)</solution>", span, re.DOTALL))
    if sol_matches and _is_terminal_slot(span, sol_matches[-1]):
        return sol_matches[-1].group(1).strip()

    bold_matches = list(re.finditer(r"\*\*([^*]+?)\*\*", span))
    for m in reversed(bold_matches):
        if _is_terminal_slot(span, m) and not _is_header_label(m.group(1)):
            return m.group(1).strip()

    return None


extract_livebench_answer_slot = _extract_livebench_answer_slot


def _check_livebench_answer_slot(
    final_answer: str,
    query: str | None = None,
    constraints: list[dict] | None = None,
) -> float:
    slot = _extract_livebench_answer_slot(final_answer, query)
    if slot is None or not slot.strip():
        return 1.0

    if constraints is not None:
        if not constraints:
            return 0.0
        passed = sum(1 for c in constraints if _check_constraint(slot, c))
        total = len(constraints)
        return 1.0 - (passed / total) if total > 0 else 0.0

    return 0.0


def _compute_ifeval_u_L(
    final_answer: str,
    query: str | None = None,
    constraints: list[dict] | None = None,
) -> float:
    extracted_span, _ = extract_answer_span(final_answer)

    if constraints is not None:
        active_constraints = constraints
    elif query and query.strip():
        active_constraints = _extract_constraints(query)
    else:
        active_constraints = []

    if not active_constraints:
        return 0.0

    passed = 0
    for c in active_constraints:
        if "strict" in c and callable(c["strict"]):
            if c["strict"](extracted_span, c["value"]):
                passed += 1
        elif _check_constraint(extracted_span, c):
            passed += 1

    total = len(active_constraints)
    return 1.0 - (passed / total) if total > 0 else 0.0


def _check_constraint(answer: str, constraint: dict) -> bool:
    ctype = constraint.get("type", "")
    value = constraint.get("value", "")
    if ctype == "min_length":
        return len(answer) >= value
    if ctype == "max_length":
        return len(answer) <= value
    if ctype == "contains":
        return value.lower() in answer.lower()
    if ctype == "starts_with":
        return answer.strip().startswith(value)
    if ctype == "ends_with":
        return answer.strip().endswith(value)
    return True
