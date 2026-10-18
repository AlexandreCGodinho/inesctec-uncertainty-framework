import json
import os
from pathlib import Path
import yaml


class JudgeParseError(Exception):
    pass


def _get_min_claim_chars(explicit_min: int | None = None) -> int:
    if explicit_min is not None:
        return explicit_min
    config_path = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and "uf_min_claim_chars" in cfg:
                return int(cfg["uf_min_claim_chars"])
        except Exception:
            pass
    return 20


def compute_u_F(
    final_answer: str,
    reference: str,
    llm_judge_callable=None,
    min_claim_chars: int | None = None,
) -> float | None:
    if not llm_judge_callable:
        raise ValueError("LLM Judge is missing. Factual evaluation aborted.")
    claims = _split_claims(final_answer, min_chars=min_claim_chars)
    if not claims:
        return None

    weights = [_estimate_severity(claim) for claim in claims]
    total_weight = sum(weights)
    if total_weight == 0:
        return None

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


def _split_claims(text: str, min_chars: int | None = None) -> list[str]:
    threshold = _get_min_claim_chars(min_chars)
    sentences = text.replace("\n", " ").split(".")
    return [s.strip() for s in sentences if len(s.strip()) > threshold]


def _estimate_severity(claim: str) -> float:
    high_severity_keywords = ["always", "never", "all", "none", "must", "impossible"]
    count = sum(1 for kw in high_severity_keywords if kw in claim.lower())
    return 1.0 + 0.5 * count


def format_judge_prompt(claims: list[str], reference: str = "") -> str:
    """Construct the exact factual judge prompt for claims, conditioned on reference if non-empty."""
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
    return prompt


def _judge_claims_batch(
    claims: list[str], reference: str, llm_judge_callable
) -> dict:
    prompt = format_judge_prompt(claims, reference)
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
