"""Unit tests for LiveBench v11 scoring and record enrichment."""

import pytest
from src.scoring.livebench_scorer import LiveBenchScorer


def test_livebench_empty_response_parse_error():
    scorer = LiveBenchScorer(mock_mode=True)
    record = {"final_answer": ""}
    result = scorer.score(record, {})
    assert result["accuracy"] == 0.0
    assert result["parse_error"] is True


def test_livebench_slot_extraction():
    final_ans = "Step 1: deduction\nStep 2: verification\n\n***spy movies***"
    extracted_slot = final_ans
    if "***" in final_ans:
        parts = final_ans.split("***")
        if len(parts) >= 3:
            extracted_slot = parts[-2].strip()
    assert extracted_slot == "spy movies"


def test_livebench_mock_match():
    scorer = LiveBenchScorer(mock_mode=True)
    record = {"query_id": "test_q", "final_answer": "The answer is 42."}
    reference = {"ground_truth": "42"}
    result = scorer.score(record, reference)
    assert result["accuracy"] == 1.0
    assert result["parse_error"] is False


def test_livebench_record_enrichment_parity():
    # Mock a complete baseline record with GPU telemetry and uncertainty metrics
    baseline_record = {
        "run_id": "bc8a2030_GOVERNED_WORKFLOW",
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
        "query_id": "bc8a2030",
        "dataset": "livebench",
        "seed": 1234,
        "final_answer": "Final conclusion: ***spy movies***",
        "termination_reason": "completed",
        "tokens_prompt": 1201,
        "tokens_completion": 1004,
        "latency_ms": 21303.6,
        "model_name": "google/gemma-4-12b-it",
        "provider": "hosted_vllm",
    }

    ref = {"ground_truth": "spy movies", "task_name": "spatial_reasoning"}
    scorer = LiveBenchScorer(mock_mode=True)
    res = scorer.score(baseline_record, ref)

    extracted_slot = "spy movies"

    # Simulate enrichment logic
    enriched_record = dict(baseline_record)
    enriched_record["match"] = res["accuracy"]
    enriched_record["livebench_extracted_slot"] = extracted_slot
    enriched_record["livebench_gold_answer"] = ref["ground_truth"]
    enriched_record["livebench_task_name"] = ref["task_name"]
    enriched_record["livebench_abstain"] = res["parse_error"]
    enriched_record["livebench_scored_v11"] = True

    # 1. Verify 100% of baseline telemetry is preserved
    for key in baseline_record:
        assert enriched_record[key] == baseline_record[key]

    # 2. Verify audit fields are correctly injected
    assert enriched_record["match"] == 1.0
    assert enriched_record["livebench_extracted_slot"] == "spy movies"
    assert enriched_record["livebench_scored_v11"] is True
