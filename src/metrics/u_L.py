def compute_u_L(final_answer: str, constraints: list[dict] | None = None) -> float:
    if not constraints:
        return 0.0
    passed = sum(
        1 for c in constraints
        if _check_constraint(final_answer, c)
    )
    total = len(constraints)
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
