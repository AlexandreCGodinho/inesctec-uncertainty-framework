import json

class JudgeParseError(Exception):
    pass

def compute_u_F(
    final_answer: str,
    reference: str,
    llm_judge_callable=None,
) -> float:
    if not llm_judge_callable:
        raise ValueError("LLM Judge is missing. Factual evaluation aborted.")
    claims = _split_claims(final_answer)
    if not claims:
        return 0.5
    
    weights = [_estimate_severity(claim) for claim in claims]
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5

    scores_dict = _judge_claims_batch(claims, reference, llm_judge_callable)
    
    scores = []
    for i in range(len(claims)):
        score = scores_dict.get(str(i), 0.0)
        try:
            score = float(score)
            if score not in [-1, 0, 0.5, 1]:
                score = 0.0
        except (ValueError, TypeError):
            score = 0.0
        scores.append(score)

    mean_s = sum(w * s for w, s in zip(weights, scores)) / total_weight
    return (1.0 - mean_s) / 2.0


def _split_claims(text: str) -> list[str]:
    sentences = text.replace("\n", " ").split(".")
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _estimate_severity(claim: str) -> float:
    high_severity_keywords = ["always", "never", "all", "none", "must", "impossible"]
    count = sum(1 for kw in high_severity_keywords if kw in claim.lower())
    return 1.0 + 0.5 * count


def _judge_claims_batch(
    claims: list[str], reference: str, llm_judge_callable
) -> dict:
    claims_text = "\n".join(f"{i}: {c}" for i, c in enumerate(claims))
    if reference.strip():
        prompt = (
            f"Reference: {reference}\n\n"
            f"Claims:\n{claims_text}\n\n"
            f"Score each claim against the reference:\n"
        )
    else:
        prompt = (
            f"Claims:\n{claims_text}\n\n"
            f"Assess each claim's factual accuracy using your own knowledge:\n"
        )
        
    prompt += (
        f"  -1 = contradicted\n"
        f"   0 = not enough information\n"
        f"  +0.5 = partially supported\n"
        f"  +1 = fully supported\n\n"
        f"You MUST return a valid JSON object mapping the claim index (as string) to its numeric score.\n"
        f"Example: {{\"0\": 1, \"1\": -1, \"2\": 0.5}}\n\n"
        f"JSON:"
    )
    for attempt in range(2):
        response = llm_judge_callable.call("judge", prompt)
        try:
            raw = response.response_text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            return json.loads(raw.strip())
        except Exception:
            if attempt == 1:
                raise JudgeParseError("Failed to parse JSON after retry")
    return {}
