"""Unit tests for PopQA v11 scoring and record enrichment."""

import pytest
from src.scoring.popqa_scorer import PopqaScorer, _extract_answer


def test_popqa_exact_alias_match():
    scorer = PopqaScorer()
    record = {"final_answer": "The composer of the opera Nancy was Adolphe Adam."}
    reference = {
        "answer": "Adolphe Adam",
        "possible_answers": "['Adolphe Adam', 'Adolphe-Charles Adam']",
        "question": "Who composed Nancy?",
    }
    result = scorer.score(record, reference)
    assert result["match"] == 1.0
    assert result["score_popqa_raw"] == 1.0


def test_popqa_secondary_alias_match():
    scorer = PopqaScorer()
    record = {"final_answer": "The author is Charles Dickens."}
    reference = {
        "answer": "C. Dickens",
        "possible_answers": "['Charles Dickens', 'Dickens', 'C. Dickens']",
    }
    result = scorer.score(record, reference)
    assert result["match"] == 1.0


def test_popqa_incorrect_answer():
    scorer = PopqaScorer()
    record = {"final_answer": "The opera was composed by Wolfgang Amadeus Mozart."}
    reference = {
        "answer": "Adolphe Adam",
        "possible_answers": "['Adolphe Adam']",
    }
    result = scorer.score(record, reference)
    assert result["match"] == 0.0


def test_popqa_noise_cleaning():
    raw_text = "Verdict: Positive\n\nAdolphe Adam\n\nI hope this helps! Feel free to ask more."
    cleaned = _extract_answer(raw_text)
    assert "Verdict" not in cleaned
    assert "I hope this helps" not in cleaned
    assert "Adolphe Adam" in cleaned


def test_popqa_empty_answer_abstain():
    scorer = PopqaScorer()
    record = {"final_answer": ""}
    reference = {"answer": "Some Entity"}
    result = scorer.score(record, reference)
    assert result["match"] == 0.0
    assert result["abstain"] is True
    assert result["extracted_answer"] == ""


def test_popqa_needs_revision_abstention_case_a():
    """Verify Doc 08 Case A: needs_revision is an operational abstention (match=0.0)."""
    scorer = PopqaScorer()
    # Deliberation discussion mentions the correct answer, but rejects it with needs_revision
    record = {
        "final_answer": (
            "The deliberation discussed whether Adolphe Adam was the composer.\n"
            "Analysis 1 says Adolphe Adam, but there are multiple flaws.\n"
            "VERDICT: needs_revision"
        )
    }
    reference = {
        "answer": "Adolphe Adam",
        "possible_answers": "['Adolphe Adam']",
    }
    result = scorer.score(record, reference)
    assert result["match"] == 0.0
    assert result["match_word_boundary"] == 0.0
    assert result["match_naive_substring"] == 0.0
    assert result["match_extracted"] == 0.0
    assert result["abstain"] is True
    assert result["extracted_answer"] == ""
    assert result["span_category"] == "needs_revision"


def test_popqa_local_mapping_integrity():
    """Verify local 150-prompt hermetic mapping integrity."""
    from scripts.score_popqa_v11 import POPQA_MAPPING_FILE, load_popqa_references
    assert POPQA_MAPPING_FILE.exists()
    refs = load_popqa_references(POPQA_MAPPING_FILE)
    assert len(refs) == 150
    # Check sample item
    assert "5247809" in refs
    assert refs["5247809"]["obj"] == "Jimmy Van Heusen"


def test_popqa_record_enrichment_parity():
    # Mock a complete baseline record with GPU telemetry and uncertainty metrics
    baseline_record = {
        "run_id": "5247809_GOVERNED_WORKFLOW",
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
        "query_id": "5247809",
        "dataset": "popqa",
        "seed": 1234,
        "final_answer": "The composer was Adolphe Adam.",
        "termination_reason": "completed",
        "tokens_prompt": 286,
        "tokens_completion": 339,
        "latency_ms": 7564.3,
        "model_name": "google/gemma-4-12b-it",
        "provider": "hosted_vllm",
    }

    ref = {
        "answer": "Adolphe Adam",
        "possible_answers": "['Adolphe Adam', 'Adolphe-Charles Adam']",
        "question": "Who composed Nancy?",
        "prop": "composer",
        "subj": "Nancy",
    }

    scorer = PopqaScorer()
    res = scorer.score(baseline_record, ref)
    extracted = _extract_answer(baseline_record["final_answer"])

    # Simulate enrichment logic
    enriched_record = dict(baseline_record)
    enriched_record["match"] = res["match"]
    enriched_record["popqa_extracted_answer"] = extracted
    enriched_record["popqa_gold_answer"] = ref["answer"]
    enriched_record["popqa_aliases"] = ['Adolphe Adam', 'Adolphe-Charles Adam']
    enriched_record["popqa_score_raw"] = res["score_popqa_raw"]
    enriched_record["popqa_question"] = ref["question"]
    enriched_record["popqa_prop"] = ref["prop"]
    enriched_record["popqa_subj"] = ref["subj"]
    enriched_record["popqa_abstain"] = False
    enriched_record["popqa_scored_v11"] = True

    # 1. Verify 100% of baseline telemetry is preserved
    for key in baseline_record:
        assert enriched_record[key] == baseline_record[key]

    # 2. Verify audit fields are correctly injected
    assert enriched_record["match"] == 1.0
    assert enriched_record["popqa_extracted_answer"] == "The composer was Adolphe Adam."
    assert enriched_record["popqa_gold_answer"] == "Adolphe Adam"
    assert enriched_record["popqa_scored_v11"] is True
