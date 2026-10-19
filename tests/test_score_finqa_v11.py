import json
import pytest
from pathlib import Path
from scripts.score_finqa_v11 import score_single_file, _generate_markdown_report


def test_score_single_file_numeric_and_abstain(tmp_path: Path):
    in_file = tmp_path / "finqa_pattern1.jsonl"
    out_file = tmp_path / "out" / "finqa_pattern1.jsonl"

    recs = [
        {"query_id": "q1", "final_answer": "**Final Answer:** 25.5%"},
        {"query_id": "q2", "final_answer": "The answer is 100 million dollars"},
        {"query_id": "q3", "final_answer": "I cannot answer based on context"},
    ]
    with open(in_file, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    refs = {
        "q1": {"exe_ans": 0.255, "answer": "25.5%"},
        "q2": {"exe_ans": 100000.0, "answer": "100 million"},  # In thousands: 100m = 100,000 thousand
        "q3": {"exe_ans": 50.0, "answer": "50"},
    }

    res = score_single_file(in_file, out_file, refs)

    assert res["total"] == 3
    assert res["matches"] == 2
    assert res["abstentions"] == 1
    assert round(res["accuracy_pct"], 1) == 66.7

    # Verify output file
    with open(out_file, "r", encoding="utf-8") as f:
        out_records = [json.loads(line) for line in f if line.strip()]

    assert len(out_records) == 3
    assert out_records[0]["match"] == 1.0
    assert out_records[0]["finqa_extracted_value"] == 25.5
    assert out_records[0]["finqa_extracted_unit"] == "%"
    assert out_records[1]["match"] == 1.0
    assert out_records[2]["match"] == 0.0
    assert out_records[2]["finqa_abstain"] is True


def test_generate_markdown_report(tmp_path: Path):
    md_file = tmp_path / "report.md"
    summary = {
        "timestamp": "2026-10-15T12:00:00Z",
        "pinned_finqa_commit": "0f16e28",
        "tolerance": {"relative": 0.005, "absolute": 1e-06},
        "overall_total": 900,
        "overall_matches": 550,
        "overall_accuracy_pct": 61.11,
        "pattern_summary": {
            "Pattern_1": {"total": 150, "matches": 90, "accuracy_pct": 60.0},
            "Pattern_4": {"total": 450, "matches": 300, "accuracy_pct": 66.67},
        },
        "files_evaluated": [
            {
                "input_path": "data/report_run_a100/finqa_pattern1.jsonl",
                "pattern": 1,
                "seed": 1234,
                "total": 150,
                "matches": 90,
                "abstentions": 5,
                "accuracy_pct": 60.0,
            }
        ],
    }
    _generate_markdown_report(summary, md_file)
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "FinQA 900 Automated Scoring Report" in content
    assert "61.11%" in content
    assert "Summary by Pattern" in content
    assert "Methodological Note on Sample Asymmetry" in content
    assert "P1 (Governed Workflow)" in content


def test_load_finqa_references_fails_offline_without_cache(tmp_path: Path):
    from scripts.score_finqa_v11 import load_finqa_references

    missing_cache = tmp_path / "nonexistent_cache.json"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_finqa_references(cache_path=missing_cache, allow_fetch=False)

    msg = str(exc_info.value)
    assert "Scoring is offline by default" in msg
    assert "--allow-fetch" in msg


def test_score_single_file_raises_on_missing_query_id(tmp_path: Path):
    in_file = tmp_path / "test_missing.jsonl"
    out_file = tmp_path / "out_missing.jsonl"
    with open(in_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "unknown_query_999", "final_answer": "42"}) + "\n")

    with pytest.raises(KeyError) as exc_info:
        score_single_file(in_file, out_file, references={})

    assert "unknown_query_999" in str(exc_info.value)

