def compute_u_F(
    final_answer: str,
    reference: str,
    llm_judge_callable=None,
) -> float:
    if not llm_judge_callable:
        return 0.0
    claims = _split_claims(final_answer)
    if not claims:
        return 0.5
    scores = []
    weights = []
    for claim in claims:
        severity = _estimate_severity(claim)
        s = _judge_claim(claim, reference, llm_judge_callable)
        scores.append(s)
        weights.append(severity)
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5
    mean_s = sum(w * s for w, s in zip(weights, scores)) / total_weight
    return (1.0 - mean_s) / 2.0


def _split_claims(text: str) -> list[str]:
    sentences = text.replace("\n", " ").split(".")
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _estimate_severity(claim: str) -> float:
    high_severity_keywords = ["always", "never", "all", "none", "must", "impossible"]
    count = sum(1 for kw in high_severity_keywords if kw in claim.lower())
    return 1.0 + 0.5 * count


def _judge_claim(
    claim: str, reference: str, llm_judge_callable
) -> float:
    prompt = (
        f"Claim: {claim}\n"
        f"Reference: {reference}\n\n"
        f"Score the claim against the reference:\n"
        f"  -1 = contradicted\n"
        f"   0 = not enough information\n"
        f"  +0.5 = partially supported\n"
        f"  +1 = fully supported\n\n"
        f"Score:"
    )
    response = llm_judge_callable("judge", prompt)
    try:
        score = float(response.response_text.strip()[:4])
        if score not in [-1, 0, 0.5, 1]:
            return 0.0
        return score
    except (ValueError, TypeError):
        return 0.0
