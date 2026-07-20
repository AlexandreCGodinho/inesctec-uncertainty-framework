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
    dataset_name: str, sample_size: int | None, base_seed: int = 1234
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
    if dataset_name == "livebench":
        logger.warning(
            "livebench has no direct datasets.load_dataset handle (Doc 02 Table 3). "
            "Skipping — implement via inspect_evals or livebench API before Phase 3."
        )
        return []

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
    if sample_size is not None:
        all_items = all_items[:sample_size]
    else:
        logger.warning("No sample_size limit set — loading full dataset '%s'", hf_id)

    queries = []
    for i, raw in enumerate(all_items):
        try:
            qid_field = field_map.get("query_id", "id")
            query_text_field = field_map.get("query", "query")
            ref_field = field_map.get("reference")

            qid = str(raw.get(qid_field, i))
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
    for dataset_name in datasets:
        try:
            sample_size = _resolve_sample_size(config, dataset_name)
        except ValueError as exc:
            logger.error(
                "Skipping dataset '%s': %s", dataset_name, exc
            )
            continue
        queries = load_dataset_queries(
            dataset_name, sample_size, config.get("base_seed", 1234)
        )
        if not queries:
            logger.info("No queries for dataset '%s' — skipping", dataset_name)
            continue

        logger.info(
            "Processing %d queries for dataset '%s' across %d pattern(s)",
            len(queries), dataset_name, len(patterns),
        )

        for query_data in queries:
            for pattern_id, orchestrator_cls in patterns:
                try:
                    state = RunState(
                        run_id=f"{query_data['query_id']}_{pattern_id.name}",
                        pattern=pattern_id,
                        dataset=dataset_name,
                        query_id=query_data["query_id"],
                        query=query_data["query"],
                        reference=query_data["reference"],
                        seed=client.seed,
                    )
                    orchestrator = orchestrator_cls(client)
                    state = orchestrator.run(state)
                    telemetry = build_telemetry(state)
                    write_telemetry_jsonl(telemetry, dataset_name, output_dir)
                except Exception:
                    logger.exception(
                        "Failed run: dataset=%s pattern=%s query=%s",
                        dataset_name,
                        pattern_id.name,
                        query_data["query_id"],
                    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import yaml
    from src.llm_client import LLMClient, check_server_available
    from src.patterns.p1_governed import GovernedWorkflow
    from src.patterns.p2_role_pipeline import RoleSeparatedPipeline
    from src.patterns.p3_reflexive import ReflexiveSingleAgent
    from src.patterns.p4_deliberation import MultiPerspectiveDeliberation

    cfg_path = Path(__file__).resolve().parent.parent / "config" / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

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

    run_pilot(datasets, patterns, client, config)
