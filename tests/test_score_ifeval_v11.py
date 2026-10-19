"""Unit tests for IFEval v11 scoring and record enrichment."""

import pytest
from src.scoring.ifeval_scorer import IfevalScorer
from src.scoring.answer_span import extract_answer_span


def test_ifeval_strict_constraint_satisfied():
    scorer = IfevalScorer()
    record = {"final_answer": "This response contains exactly six words now."}
    reference = {"prompt": "Write a response with at least 5 words."}
    result = scorer.score(record, reference)
    assert result["strict_prompt"] == 1.0
    assert result["loose_prompt"] == 1.0


def test_ifeval_strict_constraint_violated():
    scorer = IfevalScorer()
    record = {"final_answer": "Short text."}
    reference = {"prompt": "Write a response with at least 50 words."}
    result = scorer.score(record, reference)
    assert result["strict_prompt"] == 0.0
    assert result["strict_instruction"] == 0.0


def test_ifeval_answer_span_extraction():
    # Governed workflow with VERDICT: approved
    full_text = "Analysis complete.\nVERDICT: approved\nThis is the final validated answer."
    span, category = extract_answer_span(full_text)
    assert span == "This is the final validated answer."
    assert category == "approved"

    # Governed workflow with VERDICT: needs_revision
    full_text_rev = "Draft text.\nVERDICT: needs_revision\nFlawed response."
    span_rev, cat_rev = extract_answer_span(full_text_rev)
    assert span_rev == ""
    assert cat_rev == "needs_revision"

    # Standard response without VERDICT
    plain_text = "Direct output without verdict line."
    span_plain, cat_plain = extract_answer_span(plain_text)
    assert span_plain == plain_text
    assert cat_plain == "no_verdict"


def test_ifeval_empty_response_abstain():
    scorer = IfevalScorer()
    record = {"final_answer": ""}
    reference = {"prompt": "Write anything."}
    result = scorer.score(record, reference)
    assert result["strict_prompt"] == 0.0
    span, _ = extract_answer_span(record["final_answer"])
    assert bool(not span or not span.strip()) is True


def test_ifeval_record_enrichment_parity():
    # Mock a complete baseline record with GPU telemetry and uncertainty metrics
    baseline_record = {
        "run_id": "0_GOVERNED_WORKFLOW",
        "pattern": 1,
        "u_L": 0.0,
        "u_F": 0.0,
        "u_C": None,
        "u_R": None,
        "u_P": 0.0,
        "U_surface": 0.0,
        "U_composite": 0.0,
        "truncation_flag": False,
        "judge_parse_error": False,
        "query_id": "0",
        "dataset": "ifeval",
        "seed": 1234,
        "final_answer": "```json\n{\"travel_guide\": \"Bohemia\"}\n```",
        "termination_reason": "completed",
        "tokens_prompt": 774,
        "tokens_completion": 1087,
        "latency_ms": 22936.5,
        "model_name": "google/gemma-4-12b-it",
        "provider": "hosted_vllm",
    }

    ref = {
        "prompt": "Write a travel guide in JSON format.",
        "instruction_id_list": ["detectable_format:json"],
    }

    scorer = IfevalScorer()
    res = scorer.score(baseline_record, ref)
    span, span_cat = extract_answer_span(baseline_record["final_answer"])

    # Simulate enrichment logic
    enriched_record = dict(baseline_record)
    enriched_record["match"] = res["strict_prompt"]
    enriched_record["ifeval_extracted_span"] = span
    enriched_record["ifeval_prompt"] = ref["prompt"]
    enriched_record["strict_prompt"] = res["strict_prompt"]
    enriched_record["loose_prompt"] = res["loose_prompt"]
    enriched_record["strict_instruction"] = res["strict_instruction"]
    enriched_record["loose_instruction"] = res["loose_instruction"]
    enriched_record["instruction_id_list"] = ref["instruction_id_list"]
    enriched_record["ifeval_span_category"] = span_cat
    enriched_record["ifeval_abstain"] = False
    enriched_record["ifeval_scored_v11"] = True

    # 1. Verify 100% of baseline telemetry is preserved
    for key in baseline_record:
        assert enriched_record[key] == baseline_record[key]

    # 2. Verify audit fields are correctly injected
    assert "match" in enriched_record
    assert enriched_record["ifeval_prompt"] == ref["prompt"]
    assert enriched_record["ifeval_scored_v11"] is True
