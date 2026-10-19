#!/usr/bin/env python3
"""
Recompute u_F on Extracted Answer Spans for Pattern 4 (Phase 2 Task 4).

Processes:
- Datasets: PopQA, IFEval, LiveBench, FinQA
- Pattern: P4 (Multi-Perspective Deliberation)
- Seeds: 1234, 2345, 3456
- Total items: 4 datasets * 3 seeds * 150 items = 1,800 items

Operational Rules:
1. Extract answer span using extract_answer_span (Doc 08).
2. If category == "needs_revision":
   - u_F = None (strict operational abstention).
   - No judge LLM call is made.
3. If category in ("approved", "empty_approved", "approved_fallback", "no_verdict"):
   - If span is non-empty, evaluate claims strictly on the extracted span.
   - If claims list is empty, u_F = None (Phase 1 zero-claims fallback).
   - If claims exist, call LLM judge (google/gemma-4-12b-it) to score claims.
4. Reference Conditioning (Experimental Parity with P1-P3 Baseline):
   - PopQA, LiveBench, FinQA: reference-conditioned (joined by query_id).
   - IFEval: reference-free (per benchmark instruction-following design).
   - Missing reference on conditioned datasets fails loudly with KeyError.
5. Baseline data files (data/report_run_a100*/) remain frozen and untouched.
6. All recomputed records are saved to an isolated audit directory.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.metrics.u_F import (
    JudgeParseError,
    _split_claims,
    compute_u_F,
    format_judge_prompt,
)
from src.scoring.answer_span import extract_answer_span

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("recompute_p4_uf")

DATASETS = ["popqa", "ifeval", "livebench", "finqa"]
SEEDS = [1234, 2345, 3456]


def get_seed_dir_name(seed: int) -> str:
    """Return the input folder name for a given seed."""
    if seed == 1234:
        return "report_run_a100"
    return f"report_run_a100_seed{seed}"


def load_dataset_references(
    dataset_name: str, base_dir: str = "data"
) -> Dict[str, str]:
    """
    Load query_id -> reference mapping for a dataset matching baseline rules.

    - popqa: akariasai/PopQA (key: id, val: obj canonical answer).
    - finqa: test.json from official pinned commit 0f16e28 (key: id, val: qa.answer).
    - livebench: live_bench/reasoning load_questions (key: question_id, val: ground_truth or answer).
    - ifeval: reference-free by definition, returns empty mapping {}.
    """
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "ifeval":
        # IFEval runs reference-free in baseline pipeline
        return {}

    base_path = Path(base_dir)

    if dataset_name == "finqa":
        # Check local cache first
        cache_candidates = [
            base_path / "finqa_references_cache.json",
            Path("data/finqa_references_cache.json"),
        ]
        raw_data = None
        for cand in cache_candidates:
            if cand.exists():
                logger.info("Loading FinQA references from local cache: %s", cand)
                with open(cand, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                break

        if raw_data is None:
            import urllib.request
            logger.info("Local FinQA cache not found. Fetching official test.json from GitHub (commit 0f16e28)...")
            url = "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/test.json"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                raw_data = json.loads(resp.read().decode("utf-8"))

        ref_map: Dict[str, str] = {}
        for item in raw_data:
            qid = str(item.get("id"))
            ans = str(item.get("qa", {}).get("answer", ""))
            ref_map[qid] = ans
        return ref_map

    if dataset_name == "popqa":
        # Check local cache first
        cache_candidates = [
            base_path / "popqa_references_cache.json",
            Path("data/popqa_references_cache.json"),
        ]
        for cand in cache_candidates:
            if cand.exists():
                logger.info("Loading PopQA references from local cache: %s", cand)
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)

        import datasets as hf_datasets
        logger.info("Loading PopQA dataset from HuggingFace (akariasai/PopQA)...")
        ds = hf_datasets.load_dataset("akariasai/PopQA")
        split = ds["train"] if "train" in ds else ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
        ref_map = {}
        for row in split:
            qid = str(row.get("id"))
            ref = str(row.get("obj") or row.get("answer") or "")
            ref_map[qid] = ref
        return ref_map

    if dataset_name == "livebench":
        # Check local cache first
        cache_candidates = [
            base_path / "livebench_references_cache.json",
            Path("data/livebench_references_cache.json"),
        ]
        for cand in cache_candidates:
            if cand.exists():
                logger.info("Loading LiveBench references from local cache: %s", cand)
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)

        import livebench.common as lb
        logger.info("Loading LiveBench questions via official harness...")
        categories, _ = lb.get_categories_tasks(bench_name="live_bench/reasoning")
        ref_map = {}
        for cat_name, cat_dataset in categories.items():
            questions = lb.load_questions(cat_dataset, livebench_releases=lb.LIVE_BENCH_RELEASES, task_name=None)
            for q in questions:
                qid = str(q.get("question_id", ""))
                ref = str(q.get("ground_truth", q.get("answer", "")))
                ref_map[qid] = ref
        return ref_map

    raise ValueError(f"Unknown dataset: {dataset_name}")


def recompute_single_record(
    record: Dict[str, Any],
    client: Any,
    reference: str = "",
    min_claim_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process a single Pattern 4 telemetry record and recompute u_F on extracted span.
    """
    final_answer = record.get("final_answer", "") or ""
    span, category = extract_answer_span(final_answer)

    raw_claims = _split_claims(final_answer, min_chars=min_claim_chars)
    span_claims = (
        _split_claims(span, min_chars=min_claim_chars) if span.strip() else []
    )

    new_u_F: Optional[float] = None
    judge_parse_error = False

    if category == "needs_revision":
        # Strict abstention: needs_revision means no final consensus was accepted
        new_u_F = None
        judge_parse_error = False
    elif not span.strip() or not span_claims:
        # Empty span or zero claims fallback to None per Phase 1
        new_u_F = None
    else:
        try:
            new_u_F = compute_u_F(
                final_answer=span,
                reference=reference,
                llm_judge_callable=client,
                min_claim_chars=min_claim_chars,
            )
        except JudgeParseError:
            new_u_F = None
            judge_parse_error = True
        except Exception as exc:
            logger.warning(
                "Unexpected error during judge evaluation of query %s: %s",
                record.get("query_id"),
                exc,
            )
            new_u_F = None
            judge_parse_error = True

    # Build updated record with audit trail
    updated = dict(record)
    updated["u_F_original"] = record.get("u_F")
    updated["u_F"] = new_u_F
    updated["judge_parse_error"] = judge_parse_error
    updated["extracted_answer_span"] = span
    updated["span_category"] = category
    updated["raw_claims_count"] = len(raw_claims)
    updated["span_claims_count"] = len(span_claims)
    updated["reference_conditioned"] = bool(reference and reference.strip())
    updated["u_F_recomputed"] = True

    return updated


def recompute_file(
    input_path: Path,
    output_path: Path,
    client: Any,
    dataset_name: str = "",
    references: Optional[Dict[str, str]] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Recompute u_F for all records in a single {dataset}_pattern4.jsonl file.
    """
    if not input_path.exists():
        logger.warning("Input file %s does not exist, skipping.", input_path)
        return {"total": 0, "processed": 0}

    records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if sample_size is not None and sample_size > 0:
        records = records[:sample_size]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_dict = references or {}
    is_conditioned = dataset_name.lower().strip() != "ifeval"

    processed_count = 0
    needs_revision_count = 0
    approved_count = 0
    claims_raw_total = 0
    claims_span_total = 0
    u_f_valid_count = 0
    u_f_none_count = 0
    none_reasons = {
        "needs_revision": 0,
        "zero_claims": 0,
        "empty_span": 0,
        "judge_parse_error": 0,
    }

    recomputed_records: List[Dict[str, Any]] = []

    for rec in records:
        qid = str(rec.get("query_id", ""))
        if is_conditioned:
            if qid not in ref_dict:
                raise KeyError(
                    f"Query ID '{qid}' from {input_path.name} not found in references "
                    f"for dataset '{dataset_name}'. Loud failure required."
                )
            ref = ref_dict[qid]
        else:
            ref = ""

        updated = recompute_single_record(rec, client, reference=ref)
        recomputed_records.append(updated)
        processed_count += 1

        if updated["span_category"] == "needs_revision":
            needs_revision_count += 1
        elif updated["span_category"] == "approved":
            approved_count += 1

        claims_raw_total += updated["raw_claims_count"]
        claims_span_total += updated["span_claims_count"]

        if updated.get("u_F") is not None:
            u_f_valid_count += 1
        else:
            u_f_none_count += 1
            if updated.get("span_category") == "needs_revision":
                none_reasons["needs_revision"] += 1
            elif updated.get("judge_parse_error"):
                none_reasons["judge_parse_error"] += 1
            elif not updated.get("extracted_answer_span", "").strip():
                none_reasons["empty_span"] += 1
            elif updated.get("span_claims_count", 0) == 0:
                none_reasons["zero_claims"] += 1

    u_f_none_pct = round((u_f_none_count / processed_count * 100.0), 2) if processed_count > 0 else 0.0
    u_f_valid_pct = round((u_f_valid_count / processed_count * 100.0), 2) if processed_count > 0 else 0.0

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in recomputed_records:
            f.write(json.dumps(rec) + "\n")

    logger.info(
        "Finished %s -> %s (Processed: %d, Conditioned: %s, Approved: %d, NeedsRevision: %d, RawClaims: %d, SpanClaims: %d, u_F Valid: %d, u_F None: %d [%.1f%%])",
        input_path.name,
        output_path.name,
        processed_count,
        is_conditioned,
        approved_count,
        needs_revision_count,
        claims_raw_total,
        claims_span_total,
        u_f_valid_count,
        u_f_none_count,
        u_f_none_pct,
    )

    return {
        "file": input_path.name,
        "dataset": dataset_name,
        "reference_conditioned": is_conditioned,
        "total": len(records),
        "processed": processed_count,
        "approved": approved_count,
        "needs_revision": needs_revision_count,
        "raw_claims_total": claims_raw_total,
        "span_claims_total": claims_span_total,
        "u_f_valid_count": u_f_valid_count,
        "u_f_valid_pct": u_f_valid_pct,
        "u_f_none_count": u_f_none_count,
        "u_f_none_pct": u_f_none_pct,
        "u_f_none_reasons": none_reasons,
    }


def run_recompute_batch(
    base_dir: str,
    output_dir: str,
    client: Any = None,
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full recomputation batch across datasets and seeds with reference conditioning."""
    target_datasets = datasets or DATASETS
    target_seeds = seeds or SEEDS
    base_path = Path(base_dir)
    out_path = Path(output_dir)

    # 1. Preload reference mappings for each target dataset
    references_by_dataset: Dict[str, Dict[str, str]] = {}
    for ds in target_datasets:
        references_by_dataset[ds] = load_dataset_references(ds, base_dir=base_dir)
        logger.info(
            "Loaded %d reference records for dataset '%s' (conditioned: %s)",
            len(references_by_dataset[ds]),
            ds,
            ds.lower().strip() != "ifeval",
        )

    summary: Dict[str, Any] = {
        "datasets": target_datasets,
        "seeds": target_seeds,
        "files_processed": [],
        "overall_records": 0,
        "overall_raw_claims": 0,
        "overall_span_claims": 0,
        "overall_u_f_valid": 0,
        "overall_u_f_none": 0,
        "overall_u_f_none_pct": 0.0,
        "cell_uf_stats": {},
    }

    for seed in target_seeds:
        seed_folder = get_seed_dir_name(seed)
        input_seed_dir = base_path / seed_folder
        output_seed_dir = out_path / seed_folder

        for ds in target_datasets:
            in_file = input_seed_dir / f"{ds}_pattern4.jsonl"
            out_file = output_seed_dir / f"{ds}_pattern4.jsonl"

            res = recompute_file(
                input_path=in_file,
                output_path=out_file,
                client=client,
                dataset_name=ds,
                references=references_by_dataset.get(ds, {}),
                sample_size=sample_size,
            )
            cell_key = f"{ds}_seed{seed}"
            res["dataset"] = ds
            res["seed"] = seed
            summary["files_processed"].append(res)
            summary["overall_records"] += res.get("processed", 0)
            summary["overall_raw_claims"] += res.get("raw_claims_total", 0)
            summary["overall_span_claims"] += res.get("span_claims_total", 0)
            summary["overall_u_f_valid"] += res.get("u_f_valid_count", 0)
            summary["overall_u_f_none"] += res.get("u_f_none_count", 0)
            summary["cell_uf_stats"][cell_key] = {
                "dataset": ds,
                "seed": seed,
                "folder": seed_folder,
                "file": res.get("file"),
                "reference_conditioned": res.get("reference_conditioned", ds != "ifeval"),
                "total": res.get("total", 0),
                "processed": res.get("processed", 0),
                "u_f_valid_count": res.get("u_f_valid_count", 0),
                "u_f_valid_pct": res.get("u_f_valid_pct", 0.0),
                "u_f_none_count": res.get("u_f_none_count", 0),
                "u_f_none_pct": res.get("u_f_none_pct", 0.0),
                "u_f_none_reasons": res.get("u_f_none_reasons", {}),
            }

    if summary["overall_records"] == 0:
        logger.warning(
            "No records were processed. Verify that base-dir '%s' contains the expected seed directories: %s",
            base_dir,
            [get_seed_dir_name(s) for s in target_seeds],
        )

    if summary["overall_records"] > 0:
        summary["overall_u_f_none_pct"] = round(
            summary["overall_u_f_none"] / summary["overall_records"] * 100.0, 2
        )
        summary["overall_u_f_valid_pct"] = round(
            summary["overall_u_f_valid"] / summary["overall_records"] * 100.0, 2
        )

    import subprocess
    import time

    commit_sha = "unknown"
    git_tag = "v10"
    try:
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        git_tag = (
            subprocess.check_output(["git", "describe", "--tags", "--always"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        pass

    judge_prompt_templates = {
        "reference_conditioned": format_judge_prompt(["{claim_text}"], reference="{reference}"),
        "reference_free": format_judge_prompt(["{claim_text}"], reference=""),
    }

    dataset_reference_status = {
        ds: {
            "reference_conditioned": ds != "ifeval",
            "references_loaded": len(references_by_dataset.get(ds, {})),
        }
        for ds in target_datasets
    }

    manifest_data = {
        "commit": commit_sha,
        "tag": git_tag,
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge_model": "google/gemma-4-12b-it",
        "judge_role": "judge",
        "server_params": {
            "model": "google/gemma-4-12b-it",
            "max-model-len": 8192,
            "gpu-memory-utilization": 0.90,
            "dtype": "bfloat16",
        },
        "dataset_reference_status": dataset_reference_status,
        "judge_prompt_templates": judge_prompt_templates,
        "judge_prompt": judge_prompt_templates["reference_conditioned"],
        "datasets": target_datasets,
        "seeds": target_seeds,
        "overall_records": summary["overall_records"],
        "overall_u_f_valid": summary["overall_u_f_valid"],
        "overall_u_f_none": summary["overall_u_f_none"],
        "overall_u_f_none_pct": summary.get("overall_u_f_none_pct", 0.0),
        "cell_uf_stats": summary["cell_uf_stats"],
    }

    summary["manifest"] = manifest_data

    # Save summary report and manifest.json
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "recompute_uf_summary.json"
    manifest_file = out_path / "manifest.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Summary report saved to %s and manifest to %s", report_file, manifest_file)
    return summary


if __name__ == "__main__":
    from src.llm_client import LLMClient, check_server_available

    parser = argparse.ArgumentParser(
        description="Recompute u_F on extracted answer spans for Pattern 4 (with reference conditioning)."
    )
    parser.add_argument(
        "--base-dir",
        default="data",
        help="Base directory containing the report_run_a100* input directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/audits/audit_recomputed_uf_v11",
        help="Output directory where recomputed files will be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional limit on records per file for dry-run testing.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Datasets to process (default: popqa ifeval livebench finqa).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to process (default: 1234 2345 3456).",
    )
    args = parser.parse_args()

    client = LLMClient()
    if not check_server_available(client.api_base):
        logger.critical(
            "vLLM server at %s is not reachable. Aborting.", client.api_base
        )
        sys.exit(1)

    run_recompute_batch(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        client=client,
        datasets=args.datasets,
        seeds=args.seeds,
        sample_size=args.sample_size,
    )
