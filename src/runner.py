"""Phase 3 — telemetry pipeline runner.

Given dataset queries, runs them through orchestrators, computes
telemetry, and writes results to JSONL. Pattern-agnostic: accepts
orchestrator list as a parameter.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.schemas import RunState, PatternID, Telemetry
from src.metrics.build_telemetry import build_telemetry

logger = logging.getLogger(__name__)

_DATASET_HF_IDS: dict[str, str] = {
    "popqa": "akariasai/PopQA",
    "ifeval": "google/IFEval",
    "finqa": "czyssrs/FinQA",
}

_DATASET_FIELD_MAP: dict[str, dict[str, str | None]] = {
    "popqa": {
        "query_id": "id",
        "query": "question",
        "reference": "answer",
    },
    "ifeval": {
        "query_id": "instruction_id",
        "query": "prompt",
        "reference": None,
    },
    "finqa": {
        "query_id": "id",
        "query": "question",
        "reference": "answer",
    },
}


def load_dataset_queries(
    dataset_name: str, sample_size: int | None, base_seed: int = 1234, sampling_seed: int = 42
) -> list[dict[str, str]]:
    """Load queries for a given dataset name.

    Uses HuggingFace ``datasets.load_dataset`` with the identifiers from
    Doc 02, Table 3.  If *sample_size* is *None* the full dataset is
    returned (with a warning).  If set, the *first N* items are taken
    (deterministic, no shuffle).

    Field mappings (``query_id`` / ``query`` / ``reference``) are
    best-effort assumptions based on the expected schema of each HF
    dataset and have NOT been verified against real data offline — see
    ASSUMPTIONS below.

    **Assumptions / field mapping (unverified)**

    - ``popqa`` (akariasai/PopQA): ``id`` → query_id, ``question`` → query,
      ``answer`` → reference.
    - ``ifeval`` (google/IFEval): ``instruction_id`` → query_id,
      ``prompt`` → query, reference = ``""`` (IFEval is instruction-
      following, not factual recall).
    - ``finqa`` (czyssrs/FinQA): ``id`` → query_id, ``question`` → query,
      ``answer`` → reference.
    - ``livebench``: no ``load_dataset`` handle per Doc 02 — returns
      ``[]`` with a logged warning.  Must be implemented later via
      ``inspect_evals`` or the livebench API.
    """
    import random
    
    dev_split_dir = Path("data/dev_split_groq_llama31")
    excluded_ids = set()
    if dev_split_dir.exists():
        for jsonl_file in dev_split_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "query_id" in record:
                            qid = str(record["query_id"])
                            if qid.isdigit():
                                excluded_ids.add(qid)
                    except Exception:
                        pass

    if dataset_name == "livebench":
        import livebench.common as lb
        # Use the official LiveBench utility to fetch and parse the dataset.
        categories, _ = lb.get_categories_tasks(bench_name="live_bench/reasoning")
        queries = []
        for cat_name, cat_dataset in categories.items():
            # load_questions filters by release date correctly per the official harness
            questions = lb.load_questions(cat_dataset, livebench_releases=lb.LIVE_BENCH_RELEASES, task_name=None)
            for q in questions:
                # LiveBench schema structure
                qid = str(q.get("question_id", ""))
                turns = q.get("turns", [])
                query_text = str(turns[0]) if isinstance(turns, list) and len(turns) > 0 else str(q.get("question", ""))
                ref = str(q.get("ground_truth", q.get("answer", "")))
                
                
                if qid in excluded_ids:
                    continue

                queries.append({
                    "query_id": qid,
                    "query": query_text,
                    "reference": ref,
                })
        
        if sample_size is not None:
            queries = random.Random(sampling_seed).sample(queries, min(sample_size, len(queries)))
        return queries

    import datasets as hf_datasets  # type: ignore[import-untyped]

    hf_id = _DATASET_HF_IDS.get(dataset_name)
    if hf_id is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Supported: {list(_DATASET_HF_IDS)}"
        )

    field_map = _DATASET_FIELD_MAP.get(dataset_name, {})
    logger.info("Loading dataset %s (sample_size=%s)", hf_id, sample_size)

    try:
        if dataset_name == "finqa":
            import urllib.request
            finqa_url = "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/test.json"
            req = urllib.request.Request(finqa_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                hf_dataset = json.loads(resp.read().decode("utf-8"))
        else:
            ds_dict = hf_datasets.load_dataset(hf_id, trust_remote_code=True)
            if "train" in ds_dict:
                hf_dataset = ds_dict["train"]
            elif "test" in ds_dict:
                hf_dataset = ds_dict["test"]
            else:
                hf_dataset = ds_dict[list(ds_dict.keys())[0]]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset '{hf_id}'. "
            "Check network access and Hugging Face token if the dataset is gated."
        ) from exc

    all_items = list(hf_dataset)
    
    filtered_items = []
    for i, raw in enumerate(all_items):
        qid_field = field_map.get("query_id", "id")
        qid = str(raw.get(qid_field, i))
        if qid not in excluded_ids:
            filtered_items.append(raw)
            
    if sample_size is not None:
        filtered_items = random.Random(sampling_seed).sample(filtered_items, min(sample_size, len(filtered_items)))
    else:
        logger.warning("No sample_size limit set — loading full dataset '%s'", hf_id)

    queries = []
    for i, raw in enumerate(filtered_items):
        try:
            qid_field = field_map.get("query_id", "id")
            query_text_field = field_map.get("query", "query")
            ref_field = field_map.get("reference")

            qid = str(raw.get(qid_field, i))
            if dataset_name == "finqa" and isinstance(raw.get("qa"), dict):
                query_text = str(raw["qa"].get("question", ""))
                ref = str(raw["qa"].get("answer", ""))
            else:
                query_text = str(raw.get(query_text_field, ""))
                ref = str(raw[ref_field]) if ref_field and ref_field in raw else ""

            queries.append({
                "query_id": qid,
                "query": query_text,
                "reference": ref,
            })
        except Exception as exc:
            logger.warning(
                "Skipping row %d of '%s': %s", i, hf_id, exc
            )
            continue

    return queries


def write_telemetry_jsonl(
    telemetry: Telemetry, dataset_name: str, output_dir: str = "telemetry"
) -> None:
    """Append one JSON line to a per-(dataset, pattern) file.

    File name:  *output_dir*/``{dataset_name}_pattern{pattern_id}.jsonl``.
    Opens in append mode — never overwrites existing data.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filename = out / f"{dataset_name}_pattern{telemetry.pattern.value}.jsonl"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(telemetry.model_dump_json(exclude_none=False) + "\n")


def _resolve_sample_size(config: dict, dataset_name: str) -> int:
    samples_config = config.get("samples_per_dataset", {})
    if dataset_name not in samples_config:
        raise ValueError(
            f"Dataset '{dataset_name}' is not listed in "
            f"samples_per_dataset in experiment.yaml. Add it explicitly "
            f"(even as null) before running the pilot, to avoid "
            f"accidentally loading an unbounded dataset."
        )
    sample_size = samples_config[dataset_name]
    if sample_size is None:
        raise ValueError(
            f"samples_per_dataset['{dataset_name}'] is null in "
            f"experiment.yaml. Set an explicit integer sample size "
            f"before running the pilot (Doc 04 §4.0: 'set during "
            f"Phase 1')."
        )
    return sample_size


def _load_completed_query_ids(dataset_name: str, pattern_id: PatternID, output_dir: str) -> set[str]:
    out = Path(output_dir)
    filename = out / f"{dataset_name}_pattern{pattern_id.value}.jsonl"
    if not filename.exists():
        return set()
    completed = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "query_id" in data:
                    completed.add(str(data["query_id"]))
            except Exception:
                continue
    return completed


def run_pilot(
    datasets: list[str],
    patterns: list[tuple[PatternID, type]],
    client: Any,
    config: dict,
    output_dir: str = "telemetry",
) -> None:
    """Orchestrate the full pilot: load queries → run each pattern → write telemetry.

    *datasets* — names (e.g. ``["popqa", "ifeval"]``).
    *patterns* — list of ``(PatternID, OrchestratorClass)`` pairs.  The
      orchestrator is instantiated as ``cls(client)`` (no extra kwargs so
      that the pipeline is pattern-agnostic; each pattern reads its own
      config from ``experiment.yaml`` when needed).
    *client* — ``LLMClient`` or ``MockLLMClient`` instance.
    *config* — full ``experiment.yaml`` contents as a dict.
    *output_dir* — where JSONL files are written (default ``"telemetry"``).
    """
    manifest_data = {
        "finqa_commit": "0f16e2867befa6840783e58be38c9efb9229d742",
        "runs": {}
    }

    for dataset_name in datasets:
        try:
            sample_size = _resolve_sample_size(config, dataset_name)
        except ValueError as exc:
            logger.error(
                "Skipping dataset '%s': %s", dataset_name, exc
            )
            continue
        queries = load_dataset_queries(
            dataset_name, sample_size, config.get("base_seed", 1234), config.get("sampling_seed", 42)
        )
        if not queries:
            raise ValueError(f"Dataset '{dataset_name}' returned 0 records. Fail-fast rule triggered.")

        logger.info(
            "Processing %d queries for dataset '%s' across %d pattern(s)",
            len(queries), dataset_name, len(patterns),
        )

        for pattern_id, orchestrator_cls in patterns:
            expected_count = len(queries)
            completed_ids = _load_completed_query_ids(dataset_name, pattern_id, output_dir)
            actual_count = len(completed_ids)
            if completed_ids:
                logger.info(
                    "Resuming dataset '%s' pattern %s: skipping %d already completed query(ies)",
                    dataset_name,
                    pattern_id.name,
                    len(completed_ids),
                )

            for query_data in queries:
                if str(query_data["query_id"]) in completed_ids:
                    continue
                try:
                    state = RunState(
                        run_id=f"{query_data['query_id']}_{pattern_id.name}",
                        pattern=pattern_id,
                        dataset=dataset_name,
                        query_id=query_data["query_id"],
                        query=query_data["query"],
                        reference=query_data["reference"],
                        seed=client.seed,
                        model_name=client.model_name,
                        provider=client.provider,
                    )
                    orchestrator = orchestrator_cls(client)
                    state = orchestrator.run(state)
                    telemetry = build_telemetry(state)
                    write_telemetry_jsonl(telemetry, dataset_name, output_dir)
                    actual_count += 1
                except Exception:
                    logger.exception(
                        "Failed run: dataset=%s pattern=%s query=%s",
                        dataset_name,
                        pattern_id.name,
                        query_data["query_id"],
                    )
            
            # Record manifest entries
            manifest_key = f"{dataset_name}_pattern{pattern_id.value}"
            sampled_ids = [str(q["query_id"]) for q in queries]
            manifest_data["runs"][manifest_key] = {
                "expected": expected_count,
                "actual": actual_count,
                "sampled_ids": sampled_ids
            }

    # Write End-of-run manifest
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("End-of-run manifest written to %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse
    import os
    import yaml
    from src.llm_client import LLMClient, check_server_available
    from src.patterns.p1_governed import GovernedWorkflow
    from src.patterns.p2_role_pipeline import RoleSeparatedPipeline
    from src.patterns.p3_reflexive import ReflexiveSingleAgent
    from src.patterns.p4_deliberation import MultiPerspectiveDeliberation

    cfg_path = Path(__file__).resolve().parent.parent / "config" / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Run uncertainty framework pilot.")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("TELEMETRY_DIR", config.get("output_dir", "telemetry")),
        help="Directory to store telemetry JSONL files.",
    )
    args, _ = parser.parse_known_args()
    output_dir = args.output_dir

    client = LLMClient()

    if not check_server_available(client.api_base):
        logger.critical(
            "vLLM server at %s is not reachable. Aborting pilot.",
            client.api_base,
        )
        sys.exit(1)

    datasets = ["popqa", "ifeval", "finqa", "livebench"]
    patterns = [
        (PatternID.GOVERNED_WORKFLOW, GovernedWorkflow),
        (PatternID.ROLE_SEPARATED_PIPELINE, RoleSeparatedPipeline),
        (PatternID.REFLEXIVE_SINGLE_AGENT, ReflexiveSingleAgent),
        (PatternID.MULTI_PERSPECTIVE_DELIBERATION, MultiPerspectiveDeliberation),
    ]

    run_pilot(datasets, patterns, client, config, output_dir=output_dir)

