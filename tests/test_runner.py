import json
import logging
from pathlib import Path

import pytest

from src.schemas import PatternID, Telemetry
from src.llm_client import MockLLMClient
from src.patterns.p1_governed import GovernedWorkflow
from src.patterns.p2_role_pipeline import RoleSeparatedPipeline
from src.patterns.p3_reflexive import ReflexiveSingleAgent
from src.patterns.p4_deliberation import MultiPerspectiveDeliberation
from src.runner import load_dataset_queries, write_telemetry_jsonl, run_pilot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_queries():
    return [
        {"query_id": "q1", "query": "What is AI?", "reference": "AI is artificial intelligence."},
        {"query_id": "q2", "query": "What is ML?", "reference": "ML is a subset of AI."},
        {"query_id": "q3", "query": "What is RL?", "reference": "RL is reinforcement learning."},
    ]


# ---------------------------------------------------------------------------
# Task 1 — Dataset loading (mocked, no network)
# ---------------------------------------------------------------------------

def _fake_hf_dataset(data):
    """Return an object that behaves enough like a Dataset for list() + iteration."""
    class _Fake:
        def __init__(self, rows):
            self._rows = rows
        def __iter__(self):
            return iter(self._rows)
        def __len__(self):
            return len(self._rows)
    return _Fake(data)


class TestLoadDatasetQueries:
    def test_popqa_with_sample_size(self, monkeypatch):
        import datasets as hf_datasets

        data = [{"id": "p1", "question": "What is AI?", "answer": "AI is..."}]

        def mock_load(*args, **kwargs):
            return {"train": _fake_hf_dataset(data)}

        monkeypatch.setattr(hf_datasets, "load_dataset", mock_load)

        queries = load_dataset_queries("popqa", sample_size=1)
        assert len(queries) == 1
        assert queries[0]["query_id"] == "p1"
        assert queries[0]["query"] == "What is AI?"
        assert queries[0]["reference"] == "AI is..."

    def test_popqa_full_dataset_when_size_none(self, monkeypatch, caplog):
        import datasets as hf_datasets

        data = [
            {"id": "p1", "question": "Q1?", "answer": "A1"},
            {"id": "p2", "question": "Q2?", "answer": "A2"},
        ]

        def mock_load(*args, **kwargs):
            return {"train": _fake_hf_dataset(data)}

        monkeypatch.setattr(hf_datasets, "load_dataset", mock_load)

        with caplog.at_level(logging.WARNING):
            queries = load_dataset_queries("popqa", sample_size=None)

        assert len(queries) == 2
        assert "No sample_size limit" in caplog.text

    # test_livebench_returns_empty removed because it's no longer skippedvebench" in caplog.text

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_queries("nonexistent", sample_size=10)

    def test_malformed_row_is_skipped(self, monkeypatch):
        import datasets as hf_datasets

        data = [
            {"id": "p1", "question": "Q1?", "answer": "A1"},
            "not_a_dict",  # malformed
        ]

        def mock_load(*args, **kwargs):
            return {"train": _fake_hf_dataset(data)}

        monkeypatch.setattr(hf_datasets, "load_dataset", mock_load)

        queries = load_dataset_queries("popqa", sample_size=None)
        assert len(queries) == 1
        assert queries[0]["query_id"] == "p1"


# ---------------------------------------------------------------------------
# Task 2 — Telemetry JSONL writer
# ---------------------------------------------------------------------------

class TestWriteTelemetryJsonl:
    def test_writes_to_correct_file(self, tmp_path):
        t = Telemetry(run_id="r1", pattern=PatternID.GOVERNED_WORKFLOW)
        write_telemetry_jsonl(t, "popqa", output_dir=str(tmp_path))

        expected = tmp_path / "popqa_pattern1.jsonl"
        assert expected.exists()

    def test_append_not_overwrite(self, tmp_path):
        t1 = Telemetry(run_id="r1", pattern=PatternID.GOVERNED_WORKFLOW)
        t2 = Telemetry(run_id="r2", pattern=PatternID.GOVERNED_WORKFLOW)

        write_telemetry_jsonl(t1, "popqa", output_dir=str(tmp_path))
        write_telemetry_jsonl(t2, "popqa", output_dir=str(tmp_path))

        lines = (tmp_path / "popqa_pattern1.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_third_write_appends(self, tmp_path):
        t1 = Telemetry(run_id="r1", pattern=PatternID.GOVERNED_WORKFLOW)
        t2 = Telemetry(run_id="r2", pattern=PatternID.GOVERNED_WORKFLOW)
        t3 = Telemetry(run_id="r3", pattern=PatternID.GOVERNED_WORKFLOW)

        write_telemetry_jsonl(t1, "popqa", output_dir=str(tmp_path))
        write_telemetry_jsonl(t2, "popqa", output_dir=str(tmp_path))
        write_telemetry_jsonl(t3, "popqa", output_dir=str(tmp_path))

        lines = (tmp_path / "popqa_pattern1.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_lines_are_valid_json(self, tmp_path):
        t = Telemetry(
            run_id="r1",
            pattern=PatternID.ROLE_SEPARATED_PIPELINE,
            u_L=0.0, u_F=0.0, u_P=0.0,
            U_surface=0.0, U_composite=0.0,
            query_id="q1", seed=42,
            model_name="test-model",
            provider="test-provider",
            final_answer="test answer",
            termination_reason="completed",
            tokens_prompt=10, tokens_completion=5,
            latency_ms=12.3,
        )
        write_telemetry_jsonl(t, "ifeval", output_dir=str(tmp_path))

        line = (tmp_path / "ifeval_pattern2.jsonl").read_text().strip()
        obj = json.loads(line)
        assert obj["run_id"] == "r1"
        assert obj["pattern"] == 2
        assert obj["u_L"] == 0.0
        assert obj["query_id"] == "q1"
        assert obj["seed"] == 42
        assert obj["model_name"] == "test-model"
        assert obj["provider"] == "test-provider"
        assert obj["final_answer"] == "test answer"
        assert obj["termination_reason"] == "completed"
        assert obj["tokens_prompt"] == 10
        assert obj["tokens_completion"] == 5
        assert obj["latency_ms"] == 12.3

    def test_file_per_dataset_pattern_combination(self, tmp_path):
        t1 = Telemetry(run_id="r1", pattern=PatternID.GOVERNED_WORKFLOW)
        t2 = Telemetry(run_id="r2", pattern=PatternID.ROLE_SEPARATED_PIPELINE)

        write_telemetry_jsonl(t1, "popqa", output_dir=str(tmp_path))
        write_telemetry_jsonl(t2, "popqa", output_dir=str(tmp_path))

        assert (tmp_path / "popqa_pattern1.jsonl").exists()
        assert (tmp_path / "popqa_pattern2.jsonl").exists()


# ---------------------------------------------------------------------------
# Task 3 — End-to-end pilot with MockLLMClient
# ---------------------------------------------------------------------------

class TestRunPilot:
    def test_end_to_end_with_mock_client(self, synthetic_queries, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.runner.load_dataset_queries",
            lambda name, size, seed=None, **kw: synthetic_queries,
        )

        config = {"samples_per_dataset": {"test": 3}, "base_seed": 1234}
        patterns = [
            (PatternID.GOVERNED_WORKFLOW, GovernedWorkflow),
            (PatternID.ROLE_SEPARATED_PIPELINE, RoleSeparatedPipeline),
        ]
        client = MockLLMClient()

        run_pilot(["test"], patterns, client, config, output_dir=str(tmp_path))

        f1 = tmp_path / "test_pattern1.jsonl"
        f2 = tmp_path / "test_pattern2.jsonl"
        assert f1.exists()
        assert f2.exists()

        lines1 = f1.read_text().strip().split("\n")
        lines2 = f2.read_text().strip().split("\n")
        assert len(lines1) == 3
        assert len(lines2) == 3

        for line in lines1 + lines2:
            obj = json.loads(line)
            telemetry = Telemetry(**obj)
            assert telemetry.u_L is not None
            assert telemetry.u_F is not None
            assert telemetry.U_surface is not None
            assert telemetry.U_composite is not None
            assert telemetry.query_id in ("q1", "q2", "q3")
            assert telemetry.seed == 0  # MockLLMClient.seed == 0
            assert telemetry.final_answer is not None
            assert telemetry.termination_reason == "completed"
            assert telemetry.tokens_prompt >= 0
            assert telemetry.tokens_completion >= 0
            assert telemetry.latency_ms >= 0

    def test_end_to_end_with_all_patterns(self, synthetic_queries, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.runner.load_dataset_queries",
            lambda name, size, seed=None, **kw: synthetic_queries[:2],
        )

        config = {"samples_per_dataset": {"test": 2}, "base_seed": 1234}
        patterns = [
            (PatternID.GOVERNED_WORKFLOW, GovernedWorkflow),
            (PatternID.ROLE_SEPARATED_PIPELINE, RoleSeparatedPipeline),
            (PatternID.REFLEXIVE_SINGLE_AGENT, ReflexiveSingleAgent),
            (PatternID.MULTI_PERSPECTIVE_DELIBERATION, MultiPerspectiveDeliberation),
        ]
        client = MockLLMClient()

        run_pilot(["test"], patterns, client, config, output_dir=str(tmp_path))

        for pid in range(1, 5):
            f = tmp_path / f"test_pattern{pid}.jsonl"
            assert f.exists(), f"Missing {f}"
            lines = f.read_text().strip().split("\n")
            assert len(lines) == 2, f"{f} expected 2 lines, got {len(lines)}"
            for line in lines:
                obj = json.loads(line)
                telemetry = Telemetry(**obj)
                assert telemetry.u_L is not None
                assert telemetry.u_F is not None
                assert telemetry.U_surface is not None
                assert telemetry.U_composite is not None
                assert telemetry.query_id in ("q1", "q2")
                assert telemetry.seed == 0
                assert telemetry.final_answer is not None
                assert telemetry.termination_reason == "completed"
                assert telemetry.tokens_prompt >= 0
                assert telemetry.tokens_completion >= 0
                assert telemetry.latency_ms >= 0

    # test_no_query_skips_gracefully removed because empty query now triggers fail-fast

    def test_missing_dataset_in_config_raises_and_is_logged(self, monkeypatch, tmp_path, caplog):
        def mock_load(name, size, seed=None, **kw):
            return [{"query_id": "q1", "query": "Q?", "reference": "Ref."}]

        monkeypatch.setattr("src.runner.load_dataset_queries", mock_load)

        config = {"samples_per_dataset": {}, "base_seed": 1234}
        patterns = [(PatternID.GOVERNED_WORKFLOW, GovernedWorkflow)]

        with caplog.at_level(logging.ERROR):
            run_pilot(["missing"], patterns, MockLLMClient(), config, output_dir=str(tmp_path))

        assert "not listed in samples_per_dataset" in caplog.text
        # Iterdir will contain manifest.json
        files = [p.name for p in tmp_path.iterdir()]
        assert files == ["manifest.json"]

    def test_null_sample_size_raises_and_is_logged(self, monkeypatch, tmp_path, caplog):
        def mock_load(name, size, seed=None, **kw):
            return [{"query_id": "q1", "query": "Q?", "reference": "Ref."}]

        monkeypatch.setattr("src.runner.load_dataset_queries", mock_load)

        config = {"samples_per_dataset": {"popqa": None}, "base_seed": 1234}
        patterns = [(PatternID.GOVERNED_WORKFLOW, GovernedWorkflow)]

        with caplog.at_level(logging.ERROR):
            run_pilot(["popqa"], patterns, MockLLMClient(), config, output_dir=str(tmp_path))

        assert "is null in experiment.yaml" in caplog.text
        files = [p.name for p in tmp_path.iterdir()]
        assert files == ["manifest.json"]

    def test_configured_dataset_still_works(self, monkeypatch, tmp_path):
        def mock_load(name, size, seed=None, **kw):
            assert size == 5
            return [{"query_id": "q1", "query": "Q?", "reference": "Ref."}]

        monkeypatch.setattr("src.runner.load_dataset_queries", mock_load)

        config = {"samples_per_dataset": {"popqa": 5}, "base_seed": 1234}
        patterns = [(PatternID.GOVERNED_WORKFLOW, GovernedWorkflow)]

        run_pilot(["popqa"], patterns, MockLLMClient(), config, output_dir=str(tmp_path))

        assert (tmp_path / "popqa_pattern1.jsonl").exists()


# ---------------------------------------------------------------------------
# Task 4 — Resilience
# ---------------------------------------------------------------------------

class TestResilience:
    def test_failing_query_does_not_crash_pipeline(self, tmp_path, monkeypatch):
        synthetic = [
            {"query_id": "ok1", "query": "Good query", "reference": "Good ref."},
            {"query_id": "bad", "query": "Trigger failure", "reference": "Bad ref."},
            {"query_id": "ok2", "query": "Another good", "reference": "Another ref."},
        ]

        monkeypatch.setattr(
            "src.runner.load_dataset_queries",
            lambda name, size, seed=None, **kw: synthetic,
        )

        call_count = [0]

        class FailingOrchestrator:
            def __init__(self, client):
                pass
            def run(self, state):
                call_count[0] += 1
                if state.query_id == "bad":
                    raise RuntimeError("Simulated orchestration failure")
                state.final_answer = "OK"
                return state

        config = {"samples_per_dataset": {"test": 3}, "base_seed": 1234}
        patterns = [(PatternID.GOVERNED_WORKFLOW, FailingOrchestrator)]

        run_pilot(["test"], patterns, MockLLMClient(), config, output_dir=str(tmp_path))

        assert call_count[0] == 3

        lines = (tmp_path / "test_pattern1.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2


class TestTaskB1Requirements:
    def test_run_pilot_fail_fast_on_empty_dataset(self, tmp_path, monkeypatch):
        # Mock load_dataset_queries to return empty list
        monkeypatch.setattr(
            "src.runner.load_dataset_queries",
            lambda name, size, seed=None, **kw: [],
        )

        config = {"samples_per_dataset": {"empty_dataset": 3}, "base_seed": 1234}
        patterns = [(PatternID.GOVERNED_WORKFLOW, GovernedWorkflow)]

        # Should raise ValueError for fail-fast
        with pytest.raises(ValueError, match="returned 0 records. Fail-fast rule triggered."):
            run_pilot(["empty_dataset"], patterns, MockLLMClient(), config, output_dir=str(tmp_path))

    def test_run_pilot_generates_manifest(self, synthetic_queries, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.runner.load_dataset_queries",
            lambda name, size, seed=None, **kw: synthetic_queries,
        )

        config = {"samples_per_dataset": {"test_manifest": 3}, "base_seed": 1234}
        patterns = [(PatternID.GOVERNED_WORKFLOW, GovernedWorkflow)]
        
        run_pilot(["test_manifest"], patterns, MockLLMClient(), config, output_dir=str(tmp_path))
        
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        assert "runs" in manifest_data
        run_key = "test_manifest_pattern1"
        assert run_key in manifest_data["runs"]
        
        # synthetic_queries has 3 items
        assert manifest_data["runs"][run_key]["expected"] == 3
        assert manifest_data["runs"][run_key]["actual"] == 3

    def test_run_pilot_resume_skips_completed_queries(self, synthetic_queries, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.runner.load_dataset_queries",
            lambda name, size, seed=None, **kw: synthetic_queries,
        )
        config = {"samples_per_dataset": {"test_resume": 3}, "base_seed": 1234}
        patterns = [(PatternID.GOVERNED_WORKFLOW, GovernedWorkflow)]

        # 1. Primeira execução: grava 3 queries
        run_pilot(["test_resume"], patterns, MockLLMClient(), config, output_dir=str(tmp_path))
        lines_before = (tmp_path / "test_resume_pattern1.jsonl").read_text().strip().split("\n")
        assert len(lines_before) == 3

        # 2. Segunda execução (retomada): deve saltar as 3 queries existentes sem duplicar linhas
        run_pilot(["test_resume"], patterns, MockLLMClient(), config, output_dir=str(tmp_path))
        lines_after = (tmp_path / "test_resume_pattern1.jsonl").read_text().strip().split("\n")
        assert len(lines_after) == 3

        with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        assert manifest_data["runs"]["test_resume_pattern1"]["actual"] == 3

