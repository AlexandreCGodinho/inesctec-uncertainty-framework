import pytest
import tempfile
import json
from pathlib import Path
from src.scoring.score_telemetry import main

def test_finqa_scoring(monkeypatch):
    test_dir = Path("tests/data")
    telemetry_path = test_dir / "mock_telemetry.jsonl"
    reference_path = test_dir / "mock_references.jsonl"
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as out_file:
        out_path = out_file.name
        
    # Mock CLI arguments for FinQA
    args = ["score_telemetry.py", "--input", str(telemetry_path), "--output", out_path, "--dataset", "finqa", "--reference", str(reference_path)]
    monkeypatch.setattr("sys.argv", args)
    
    main()
    
    with open(out_path, "r") as f:
        lines = f.readlines()
        
    assert len(lines) == 5
    
    results = [json.loads(line) for line in lines]
    
    # f1: correct execution, program not exact match
    f1 = next(r for r in results if r["query_id"] == "f1")
    assert f1["scores"]["execution_accuracy"] == 1.0
    assert f1["scores"]["program_accuracy"] == 0.0
    assert not f1["scores"]["parse_error"]
    
    # f2: correct execution, program exact match
    f2 = next(r for r in results if r["query_id"] == "f2")
    assert f2["scores"]["execution_accuracy"] == 1.0
    assert f2["scores"]["program_accuracy"] == 1.0
    
    # f3: unparseable math output
    f3 = next(r for r in results if r["query_id"] == "f3")
    assert f3["scores"]["parse_error"] == True
    assert f3["scores"]["execution_accuracy"] == 0.0

def test_livebench_scoring(monkeypatch):
    test_dir = Path("tests/data")
    telemetry_path = test_dir / "mock_telemetry.jsonl"
    reference_path = test_dir / "mock_references.jsonl"
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as out_file:
        out_path = out_file.name
        
    # Mock CLI arguments for LiveBench
    args = ["score_telemetry.py", "--input", str(telemetry_path), "--output", out_path, "--dataset", "livebench", "--reference", str(reference_path)]
    monkeypatch.setattr("sys.argv", args)
    
    main()
    
    with open(out_path, "r") as f:
        lines = f.readlines()
        
    results = [json.loads(line) for line in lines]
    
    # l1: "true" matches "It is true"
    l1 = next(r for r in results if r["query_id"] == "l1")
    assert l1["scores"]["accuracy"] == 1.0
    
    # l2: "true" does not match "False."
    l2 = next(r for r in results if r["query_id"] == "l2")
    assert l2["scores"]["accuracy"] == 0.0
