import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scoring.answer_span import extract_answer_span
from src.scoring.base import Scorer

logger = logging.getLogger(__name__)

# Ensure Google Research's instruction_following_eval vendored harness is imported
try:
    from src.scoring.vendor.google_ifeval import evaluation_lib as EL
    from src.scoring.vendor.google_ifeval import instructions_registry
    _HAS_OFFICIAL_HARNESS = True
except ImportError:
    try:
        from instruction_following_eval import evaluation_lib as EL
        from instruction_following_eval import instructions_registry
        _HAS_OFFICIAL_HARNESS = True
    except ImportError:
        _HAS_OFFICIAL_HARNESS = False
        logger.warning("instruction_following_eval vendored harness not available, falling back to regex parser.")

_ALIAS_MAP = {
    "detectable_format:json": "detectable_format:json_format",
}


class IfevalScorer(Scorer):
    """IFEval Scorer implementing Google Research's official evaluation harness.

    Supports:
    - Official strict prompt-level evaluation (all instructions followed byte-for-byte).
    - Official loose prompt-level evaluation (heuristic transformations removing greetings/markdown).
    - Instruction-level metrics (strict and loose).
    - P4 dual-evaluation: span after VERDICT (primary, main paper table) and full response (appendix sensitivity).
    - Backwards-compatible regex fallback when instruction_id_list is not provided in references.
    """

    def score(
        self,
        record: dict,
        reference: dict,
        use_span: Optional[bool] = None,
    ) -> dict:
        """
        Score an IFEval record against reference instructions.
        
        If use_span is True, or if pattern == 4 and use_span is not explicitly False,
        evaluates the text extracted after the VERDICT line (extracted_answer_span per Doc 09).
        """
        pattern = record.get("pattern")
        raw_final_answer = record.get("final_answer", "")

        # Determine which text is primary based on pattern and flags
        should_use_span = use_span if use_span is not None else (pattern == 4)

        if should_use_span:
            extracted_text, span_cat = extract_answer_span(raw_final_answer)
            eval_text = extracted_text if extracted_text and extracted_text.strip() else raw_final_answer
        else:
            eval_text = raw_final_answer
            extracted_text, span_cat = extract_answer_span(raw_final_answer)

        prompt = reference.get("prompt", reference.get("query", ""))
        instruction_id_list = reference.get("instruction_id_list")
        key = reference.get("key", 0)

        # If official instruction list is available and harness is imported, use official evaluator
        if instruction_id_list and _HAS_OFFICIAL_HARNESS:
            kwargs = reference.get("kwargs", [])
            primary_res = _score_official_google(prompt, eval_text, instruction_id_list, kwargs, key=key)

            # If evaluating span, also evaluate full response for sensitivity analysis (appendix)
            if should_use_span and raw_final_answer != eval_text:
                full_res = _score_official_google(prompt, raw_final_answer, instruction_id_list, kwargs, key=key)
                primary_res["strict_prompt_full_response"] = full_res.get("strict_prompt", 0.0)
                primary_res["loose_prompt_full_response"] = full_res.get("loose_prompt", 0.0)
                primary_res["strict_instruction_full_response"] = full_res.get("strict_instruction", 0.0)
                primary_res["loose_instruction_full_response"] = full_res.get("loose_instruction", 0.0)
            else:
                primary_res["strict_prompt_full_response"] = primary_res["strict_prompt"]
                primary_res["loose_prompt_full_response"] = primary_res["loose_prompt"]
                primary_res["strict_instruction_full_response"] = primary_res["strict_instruction"]
                primary_res["loose_instruction_full_response"] = primary_res["loose_instruction"]

            primary_res["span_category"] = span_cat
            primary_res["evaluated_text_type"] = "span_after_verdict" if should_use_span else "final_answer"
            return primary_res

        # Fallback to regex evaluation when instruction IDs are not supplied (e.g. lightweight tests)
        res = _score_ifeval_inner(prompt, eval_text)
        res["strict_prompt_full_response"] = res["strict_prompt"]
        res["loose_prompt_full_response"] = res["loose_prompt"]
        res["span_category"] = span_cat
        res["evaluated_text_type"] = "span_after_verdict" if should_use_span else "final_answer"
        return res


def _score_official_google(
    prompt: str,
    response: str,
    instruction_id_list: List[str],
    kwargs: List[Dict[str, Any]],
    key: int = 0,
) -> Dict[str, Any]:
    """Execute official Google Research strict and loose evaluation."""
    # Normalize instruction IDs (e.g. alias mapping)
    normalized_inst_ids = [_ALIAS_MAP.get(iid, iid) for iid in instruction_id_list]

    cleaned_kwargs = []
    for i in range(len(normalized_inst_ids)):
        if i < len(kwargs) and isinstance(kwargs[i], dict):
            cleaned_kwargs.append({k: v for k, v in kwargs[i].items() if v is not None})
        else:
            cleaned_kwargs.append({})

    inp = EL.InputExample(
        key=key,
        instruction_id_list=normalized_inst_ids,
        prompt=prompt,
        kwargs=cleaned_kwargs,
    )

    resp_dict = {prompt: response or ""}
    res_strict = EL.test_instruction_following_strict(inp, resp_dict)
    res_loose = EL.test_instruction_following_loose(inp, resp_dict)

    strict_list = res_strict.follow_instruction_list
    loose_list = res_loose.follow_instruction_list
    n = len(strict_list) if strict_list else 1

    strict_prompt_pass = 1.0 if res_strict.follow_all_instructions else 0.0
    loose_prompt_pass = 1.0 if res_loose.follow_all_instructions else 0.0
    at_least_one = 1.0 if any(strict_list) else 0.0

    return {
        "match": strict_prompt_pass,
        "strict_prompt": strict_prompt_pass,
        "loose_prompt": loose_prompt_pass,
        "strict_instruction": sum(strict_list) / n if strict_list else 0.0,
        "loose_instruction": sum(loose_list) / n if loose_list else 0.0,
        "at_least_one_instruction": at_least_one,
        "instruction_id_list": instruction_id_list,
        "follow_instruction_list_strict": strict_list,
        "follow_instruction_list_loose": loose_list,
    }



def _score_ifeval_inner(prompt: str, response: str) -> dict[str, Any]:
    constraints = _extract_constraints(prompt)
    if not constraints:
        return {
            "match": 0.0,
            "strict_prompt": 0.0,
            "loose_prompt": 0.0,
            "strict_instruction": 0.0,
            "loose_instruction": 0.0,
            "at_least_one_instruction": 0.0,
        }

    strict_results = [c["strict"](response, c["value"]) for c in constraints]
    loose_results = [c["loose"](response, c["value"]) for c in constraints]

    n = len(constraints)
    strict_prompt = 1.0 if strict_results and all(strict_results) else 0.0
    loose_prompt = 1.0 if loose_results and any(loose_results) else 0.0
    return {
        "match": strict_prompt,
        "strict_prompt": strict_prompt,
        "loose_prompt": loose_prompt,
        "strict_instruction": sum(strict_results) / n,
        "loose_instruction": sum(loose_results) / n,
        "at_least_one_instruction": 1.0 if any(strict_results) else 0.0,
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
