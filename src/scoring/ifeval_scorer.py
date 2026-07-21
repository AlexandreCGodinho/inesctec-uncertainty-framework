import re

from src.scoring.base import Scorer


class IfevalScorer(Scorer):
    def score(self, record: dict, reference: dict) -> dict:
        response = record.get("final_answer", "")
        prompt = reference.get("prompt", reference.get("query", ""))
        return _score_ifeval_inner(prompt, response)


def _score_ifeval_inner(prompt: str, response: str) -> dict[str, float]:
    constraints = _extract_constraints(prompt)
    if not constraints:
        return {
            "strict_prompt": 0.0,
            "loose_prompt": 0.0,
            "strict_instruction": 0.0,
            "loose_instruction": 0.0,
        }

    strict_results = [c["strict"](response, c["value"]) for c in constraints]
    loose_results = [c["loose"](response, c["value"]) for c in constraints]

    n = len(constraints)
    return {
        "strict_prompt": 1.0 if strict_results and all(strict_results) else 0.0,
        "loose_prompt": 1.0 if loose_results and any(loose_results) else 0.0,
        "strict_instruction": sum(strict_results) / n,
        "loose_instruction": sum(loose_results) / n,
    }


def _extract_constraints(prompt: str) -> list[dict]:
    constraints = []
    flags = re.IGNORECASE

    m = re.search(r"(?:at least|minimum of|no fewer than)\s+(\d+)\s*characters?", prompt, flags)
    if m:
        n = int(m.group(1))
        constraints.append({
            "type": "min_length",
            "value": n,
            "strict": lambda r, v: len(r) >= v,
            "loose": lambda r, v: len(r) >= v - 2,
        })

    m = re.search(r"(?:no more than|at most|maximum of|fewer than)\s+(\d+)\s*characters?", prompt, flags)
    if m:
        n = int(m.group(1))
        constraints.append({
            "type": "max_length",
            "value": n,
            "strict": lambda r, v: len(r) <= v,
            "loose": lambda r, v: len(r) <= v + 5,
        })

    m = re.search(r"(?:at least|minimum of|no fewer than)\s+(\d+)\s*words?", prompt, flags)
    if m:
        n = int(m.group(1))
        constraints.append({
            "type": "min_words",
            "value": n,
            "strict": lambda r, v: len(r.split()) >= v,
            "loose": lambda r, v: len(r.split()) >= v - 2,
        })

    m = re.search(r"(?:no more than|at most|maximum of|fewer than)\s+(\d+)\s*words?", prompt, flags)
    if m:
        n = int(m.group(1))
        constraints.append({
            "type": "max_words",
            "value": n,
            "strict": lambda r, v: len(r.split()) <= v,
            "loose": lambda r, v: len(r.split()) <= v + 5,
        })

    m = re.search(r"(?:at least|minimum of|no fewer than)\s+(\d+)\s*sentences?", prompt, flags)
    if m:
        n = int(m.group(1))
        constraints.append({
            "type": "min_sentences",
            "value": n,
            "strict": lambda r, v: _count_sentences(r) >= v,
            "loose": lambda r, v: _count_sentences(r) >= v - 1,
        })

    m = re.search(r"(?:no more than|at most|maximum of|fewer than)\s+(\d+)\s*sentences?", prompt, flags)
    if m:
        n = int(m.group(1))
        constraints.append({
            "type": "max_sentences",
            "value": n,
            "strict": lambda r, v: _count_sentences(r) <= v,
            "loose": lambda r, v: _count_sentences(r) <= v + 1,
        })

    m = re.search(r"(?:include|contain|using|with|use)\s+the\s+word\s+['\"]?(\w+)['\"]?", prompt, flags)
    if m:
        word = m.group(1).lower()
        constraints.append({
            "type": "contains_word",
            "value": word,
            "strict": lambda r, v: v in r.lower().split(),
            "loose": lambda r, v: v in r.lower(),
        })

    m = re.search(r"exactly\s+(\d+)\s*bullets?", prompt, flags)
    if m:
        n = int(m.group(1))
        constraints.append({
            "type": "exact_bullets",
            "value": n,
            "strict": lambda r, v: _count_bullets(r) == v,
            "loose": lambda r, v: abs(_count_bullets(r) - v) <= 1,
        })

    m = re.search(r"start(?:s|ing)?\s+with\s+['\"]?(.+?)['\"]?\.?$", prompt, flags)
    if m:
        prefix = m.group(1).strip()
        constraints.append({
            "type": "starts_with",
            "value": prefix,
            "strict": lambda r, v: r.strip().lower().startswith(v.lower()),
            "loose": lambda r, v: r.strip().lower().startswith(v.lower()),
        })

    m = re.search(r"end(?:s|ing)?\s+with\s+['\"]?(.+?)['\"]?\.?$", prompt, flags)
    if m:
        suffix = m.group(1).strip()
        constraints.append({
            "type": "ends_with",
            "value": suffix,
            "strict": lambda r, v: r.strip().lower().endswith(v.lower()),
            "loose": lambda r, v: r.strip().lower().endswith(v.lower()),
        })

    return constraints


def _count_sentences(text: str) -> int:
    if not text or not text.strip():
        return 0
    parts = re.split(r"[.!?]+(?:\s|$)", text.strip())
    return len([p for p in parts if p.strip()])


def _count_bullets(text: str) -> int:
    return text.count("- ") + text.count("* ") + text.count("•")
