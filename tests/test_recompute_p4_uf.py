import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.scoring.recompute_p4_uf import (
    recompute_single_record,
    recompute_file,
    get_seed_dir_name,
)


class MockJudgeResponse:
    def __init__(self, response_text: str):
        self.response_text = response_text


class MockJudgeClient:
    def __init__(self, score_mapping: str = '{"0": 1, "1": 1}'):
        self.score_mapping = score_mapping
        self.call_count = 0

    def call(self, role: str, prompt: str):
        self.call_count += 1
        return MockJudgeResponse(self.score_mapping)


def test_get_seed_dir_name():
    assert get_seed_dir_name(1234) == "report_run_a100"
    assert get_seed_dir_name(2345) == "report_run_a100_seed2345"
    assert get_seed_dir_name(3456) == "report_run_a100_seed3456"


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


def test_recompute_approved_calls_judge_on_span():
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
    updated = recompute_single_record(record, client, min_claim_chars=10)

    assert updated["u_F"] == 0.0  # (1 - 1.0)/2 = 0.0 (perfect factual support)
    assert updated["u_F_original"] == 0.35
    assert updated["span_category"] == "approved"
    assert "The capital of France is Paris" in updated["extracted_answer_span"]
    assert client.call_count == 1


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


def test_recompute_file_end_to_end(tmp_path: Path):
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
    summary = recompute_file(in_file, out_file, client, sample_size=None)

    assert summary["processed"] == 2
    assert summary["needs_revision"] == 1
    assert summary["approved"] == 1
    assert summary["u_f_valid_count"] == 1
    assert summary["u_f_none_count"] == 1
    assert summary["u_f_none_reasons"]["needs_revision"] == 1
    assert out_file.exists()


def test_run_recompute_batch_records_manifest(tmp_path: Path):
    from src.scoring.recompute_p4_uf import run_recompute_batch
    base_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    seed_dir = base_dir / "report_run_a100"
    seed_dir.mkdir(parents=True)
    in_file = seed_dir / "finqa_pattern4.jsonl"
    with open(in_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "q1", "final_answer": "VERDICT: approved\n90%", "u_F": 0.5}) + "\n")
        f.write(json.dumps({"query_id": "q2", "final_answer": "VERDICT: approved\nThis is a full sentence claim about finances.", "u_F": 0.5}) + "\n")

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
    assert summary["cell_uf_stats"]["finqa_seed1234"]["u_f_none_reasons"]["zero_claims"] == 1

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["overall_records"] == 2
    assert "finqa_seed1234" in m["cell_uf_stats"]
