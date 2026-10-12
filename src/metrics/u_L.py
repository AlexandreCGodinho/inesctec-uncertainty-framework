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
        return _check_livebench_answer_slot(final_answer, query)

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


def _check_livebench_answer_slot(final_answer: str, query: str | None = None) -> float:
    if not final_answer or not final_answer.strip():
        return 1.0

    # Determine required format from query if present
    if query:
        if "***" in query:
            passed = bool(re.search(r"\*\*\*[^*]+?\*\*\*", final_answer))
            return 0.0 if passed else 1.0
        if "<solution>" in query:
            passed = bool(re.search(r"<solution>.*?</solution>", final_answer, re.DOTALL))
            return 0.0 if passed else 1.0
        if "**bold**" in query or "in **bold**" in query or "bold" in query.lower():
            passed = bool(re.search(r"\*\*[^*]+?\*\*", final_answer))
            return 0.0 if passed else 1.0

    # General fallback: check if any standard LiveBench answer slot is present
    has_slot = (
        bool(re.search(r"\*\*\*[^*]+?\*\*\*", final_answer))
        or bool(re.search(r"<solution>.*?</solution>", final_answer, re.DOTALL))
        or bool(re.search(r"\*\*[^*]+?\*\*", final_answer))
    )
    return 0.0 if has_slot else 1.0


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
