import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.metrics.u_F import format_judge_prompt
from src.scoring.recompute_p4_uf import (
    recompute_single_record,
    recompute_file,
    get_seed_dir_name,
    load_dataset_references,
    run_recompute_batch,
)


class MockJudgeResponse:
    def __init__(self, response_text: str):
        self.response_text = response_text


class MockJudgeClient:
    def __init__(self, score_mapping: str = '{"0": 1, "1": 1}'):
        self.score_mapping = score_mapping
        self.call_count = 0
        self.last_prompt = ""
        self.last_role = ""

    def call(self, role: str, prompt: str):
        self.call_count += 1
        self.last_role = role
        self.last_prompt = prompt
        return MockJudgeResponse(self.score_mapping)


def test_get_seed_dir_name():
    assert get_seed_dir_name(1234) == "report_run_a100"
    assert get_seed_dir_name(2345) == "report_run_a100_seed2345"
    assert get_seed_dir_name(3456) == "report_run_a100_seed3456"


def test_format_judge_prompt_parity_against_july_baseline():
    """Verify that format_judge_prompt produces character-identical templates to July baseline."""
    claims = ["The capital of France is Paris."]
    reference = "Paris"

    # Reference-conditioned prompt (July baseline commit bc7e61d)
    prompt_cond = format_judge_prompt(claims, reference=reference)
    expected_cond_prefix = (
        "Reference: Paris\n\n"
        "Claims:\n0: The capital of France is Paris.\n\n"
        "Score each claim against the reference:\n"
    )
    assert prompt_cond.startswith(expected_cond_prefix)
    assert "  -1 = contradicted\n" in prompt_cond
    assert "  +1 = fully supported\n" in prompt_cond

    # Reference-free prompt (July baseline commit bc7e61d)
    prompt_free = format_judge_prompt(claims, reference="")
    expected_free_prefix = (
        "Claims:\n0: The capital of France is Paris.\n\n"
        "Assess each claim's factual accuracy using your own knowledge:\n"
    )
    assert prompt_free.startswith(expected_free_prefix)
    assert "Reference:" not in prompt_free


def test_recompute_needs_revision_abstains_without_judge_call():
    client = MockJudgeClient()
    record = {
        "query_id": "test_1",
        "final_answer": "Deliberation text.\nVERDICT: needs_revision\nDisagreement between analysts.",
        "u_F": 0.45,
    }
    updated = recompute_single_record(record, client)

    assert updated["u_F"] is None
    assert updated["u_F_original"] == 0.45
    assert updated["span_category"] == "needs_revision"
    assert updated["span_claims_count"] == 0
    assert client.call_count == 0  # Zero calls to judge


def test_recompute_approved_reference_conditioned_passes_reference():
    client = MockJudgeClient(score_mapping='{"0": 1, "1": 1}')
    record = {
        "query_id": "test_2",
        "final_answer": (
            "Deliberation commentary with long sentences.\n"
            "VERDICT: approved\n"
            "The capital of France is Paris. It is located in Europe."
        ),
        "u_F": 0.35,
    }
    updated = recompute_single_record(
        record,
        client,
        reference="Paris is the capital of France.",
        min_claim_chars=10,
    )

    assert updated["u_F"] == 0.0  # (1 - 1.0)/2 = 0.0 (perfect factual support)
    assert updated["u_F_original"] == 0.35
    assert updated["span_category"] == "approved"
    assert updated["reference_conditioned"] is True
    assert "The capital of France is Paris" in updated["extracted_answer_span"]
    assert client.call_count == 1
    assert "Reference: Paris is the capital of France." in client.last_prompt
    assert "Score each claim against the reference:" in client.last_prompt


def test_recompute_approved_reference_free_omits_reference():
    client = MockJudgeClient(score_mapping='{"0": 1, "1": 1}')
    record = {
        "query_id": "test_ifeval",
        "final_answer": (
            "Deliberation commentary.\n"
            "VERDICT: approved\n"
            "Follow all instructions precisely and format cleanly."
        ),
        "u_F": 0.35,
    }
    updated = recompute_single_record(
        record,
        client,
        reference="",
        min_claim_chars=10,
    )

    assert updated["u_F"] == 0.0
    assert updated["reference_conditioned"] is False
    assert client.call_count == 1
    assert "Reference:" not in client.last_prompt
    assert "Assess each claim's factual accuracy using your own knowledge:" in client.last_prompt


def test_recompute_empty_span_returns_none():
    client = MockJudgeClient()
    record = {
        "query_id": "test_3",
        "final_answer": "VERDICT: approved\n",
        "u_F": 0.2,
    }
    updated = recompute_single_record(record, client)

    assert updated["u_F"] is None
    assert client.call_count == 0


def test_recompute_file_end_to_end_with_references(tmp_path: Path):
    in_file = tmp_path / "input.jsonl"
    out_file = tmp_path / "output.jsonl"

    recs = [
        {"query_id": "q1", "final_answer": "VERDICT: needs_revision\n", "u_F": 0.5},
        {
            "query_id": "q2",
            "final_answer": "VERDICT: approved\nThis is a long sentence that forms a valid claim.",
            "u_F": 0.5,
        },
    ]
    with open(in_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    client = MockJudgeClient(score_mapping='{"0": 1}')
    refs = {"q1": "Reference 1", "q2": "Reference 2"}
    summary = recompute_file(
        in_file,
        out_file,
        client,
        dataset_name="finqa",
        references=refs,
        sample_size=None,
    )

    assert summary["processed"] == 2
    assert summary["reference_conditioned"] is True
    assert summary["needs_revision"] == 1
    assert summary["approved"] == 1
    assert summary["u_f_valid_count"] == 1
    assert summary["u_f_none_count"] == 1
    assert summary["u_f_none_reasons"]["needs_revision"] == 1
    assert out_file.exists()

    # Read output file and assert reference_conditioned flag on each record
    with open(out_file, "r", encoding="utf-8") as f:
        out_records = [json.loads(line) for line in f]
    assert out_records[0]["reference_conditioned"] is True   # FinQA is reference-conditioned
    assert out_records[1]["reference_conditioned"] is True


def test_recompute_file_loud_failure_on_missing_reference(tmp_path: Path):
    in_file = tmp_path / "input_missing.jsonl"
    out_file = tmp_path / "output_missing.jsonl"

    recs = [
        {"query_id": "unmatched_qid", "final_answer": "VERDICT: approved\nA valid claim.", "u_F": 0.5},
    ]
    with open(in_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    client = MockJudgeClient(score_mapping='{"0": 1}')
    # Provided references do NOT contain 'unmatched_qid'
    refs = {"other_qid": "Some Reference"}

    with pytest.raises(KeyError, match="not found in references for dataset 'popqa'"):
        recompute_file(
            in_file,
            out_file,
            client,
            dataset_name="popqa",
            references=refs,
        )


def test_recompute_file_ifeval_runs_reference_free(tmp_path: Path):
    in_file = tmp_path / "ifeval_input.jsonl"
    out_file = tmp_path / "ifeval_output.jsonl"

    recs = [
        {"query_id": "ifeval_1", "final_answer": "VERDICT: approved\nThis is a long sentence claim.", "u_F": 0.5},
    ]
    with open(in_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    client = MockJudgeClient(score_mapping='{"0": 1}')
    # For IFEval, references dict can be empty and must NOT raise KeyError
    summary = recompute_file(
        in_file,
        out_file,
        client,
        dataset_name="ifeval",
        references={},
    )

    assert summary["processed"] == 1
    assert summary["reference_conditioned"] is False
    assert "Assess each claim's factual accuracy using your own knowledge:" in client.last_prompt


def test_run_recompute_batch_records_manifest(tmp_path: Path, monkeypatch):
    base_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    seed_dir = base_dir / "report_run_a100"
    seed_dir.mkdir(parents=True)
    in_file = seed_dir / "finqa_pattern4.jsonl"
    with open(in_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "q1", "final_answer": "VERDICT: approved\n90%", "u_F": 0.5}) + "\n")
        f.write(json.dumps({"query_id": "q2", "final_answer": "VERDICT: approved\nThis is a full sentence claim about finances.", "u_F": 0.5}) + "\n")

    # Mock load_dataset_references so unit test does not touch network
    def mock_load_refs(dataset_name: str, base_dir: str = "data"):
        return {"q1": "90%", "q2": "100 million"}

    monkeypatch.setattr("src.scoring.recompute_p4_uf.load_dataset_references", mock_load_refs)

    client = MockJudgeClient(score_mapping='{"0": 1}')
    summary = run_recompute_batch(
        base_dir=str(base_dir),
        output_dir=str(out_dir),
        client=client,
        datasets=["finqa"],
        seeds=[1234],
    )

    assert summary["overall_records"] == 2
    assert summary["overall_u_f_valid"] == 1
    assert summary["overall_u_f_none"] == 1
    assert "finqa_seed1234" in summary["cell_uf_stats"]
    assert summary["cell_uf_stats"]["finqa_seed1234"]["reference_conditioned"] is True
    assert summary["cell_uf_stats"]["finqa_seed1234"]["u_f_none_reasons"]["zero_claims"] == 1

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["overall_records"] == 2
    assert "finqa_seed1234" in m["cell_uf_stats"]
    assert m["dataset_reference_status"]["finqa"]["reference_conditioned"] is True
    assert "judge_prompt_templates" in m
    assert "reference_conditioned" in m["judge_prompt_templates"]
    assert "reference_free" in m["judge_prompt_templates"]
