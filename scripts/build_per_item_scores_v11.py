"""Consolidation script to build telemetry/per_item_scores_v11.csv.

Integrates:
1. Baseline telemetry across all 4 datasets (PopQA, IFEval, LiveBench, FinQA)
   and seeds (1234, 2345, 3456) from data/report_run_a100*/ (read-only).
2. Phase 2 Task 4 recomputed u_F on extracted answer spans for Pattern 4
   from data/audits/audit_recomputed_uf_v11/ (read-only).
3. Phase 2 Task 5B deterministic scores for FinQA across all 900 records
   from data/audits/finqa_v11_scores/ (read-only).
4. Official evaluation scoring for PopQA (Word-Boundary Doc 09), IFEval (Google Research),
   and LiveBench (Official Reasoning) from data/audits/*_v11_scores/ (read-only).
5. Strict abstention and v11 freeze composite/surface score recomputation per Doc 03.

Output:
- telemetry/per_item_scores_v11.csv (3,600 rows total, 33 columns)
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.composite import compute_composite, compute_surface

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RUNS = [
    # Seed 1234 (all 4 patterns)
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "popqa", "pattern": 1},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "popqa", "pattern": 2},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "popqa", "pattern": 3},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "popqa", "pattern": 4},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "ifeval", "pattern": 1},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "ifeval", "pattern": 2},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "ifeval", "pattern": 3},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "ifeval", "pattern": 4},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "livebench", "pattern": 1},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "livebench", "pattern": 2},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "livebench", "pattern": 3},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "livebench", "pattern": 4},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "finqa", "pattern": 1},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "finqa", "pattern": 2},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "finqa", "pattern": 3},
    {"seed_dir": "report_run_a100", "seed": 1234, "dataset": "finqa", "pattern": 4},
    # Seed 2345 (Pattern 4 only)
    {"seed_dir": "report_run_a100_seed2345", "seed": 2345, "dataset": "popqa", "pattern": 4},
    {"seed_dir": "report_run_a100_seed2345", "seed": 2345, "dataset": "ifeval", "pattern": 4},
    {"seed_dir": "report_run_a100_seed2345", "seed": 2345, "dataset": "livebench", "pattern": 4},
    {"seed_dir": "report_run_a100_seed2345", "seed": 2345, "dataset": "finqa", "pattern": 4},
    # Seed 3456 (Pattern 4 only)
    {"seed_dir": "report_run_a100_seed3456", "seed": 3456, "dataset": "popqa", "pattern": 4},
    {"seed_dir": "report_run_a100_seed3456", "seed": 3456, "dataset": "ifeval", "pattern": 4},
    {"seed_dir": "report_run_a100_seed3456", "seed": 3456, "dataset": "livebench", "pattern": 4},
    {"seed_dir": "report_run_a100_seed3456", "seed": 3456, "dataset": "finqa", "pattern": 4},
]


def build_v11_table(
    base_dir: str = "data",
    recomputed_uf_dir: str = "data/audits/audit_recomputed_uf_v11",
    audits_dir: str = "data/audits",
    output_csv: str = "telemetry/per_item_scores_v11.csv",
) -> List[Dict[str, Any]]:
    """Build unified per-item scores table v11."""
    base_path = Path(base_dir)
    recomp_path = Path(recomputed_uf_dir)
    audits_path = Path(audits_dir)
    out_path = Path(output_csv)

    all_rows: List[Dict[str, Any]] = []

    for run in RUNS:
        s_dir = run["seed_dir"]
        seed = run["seed"]
        dataset = run["dataset"]
        pattern = run["pattern"]

        raw_file = base_path / s_dir / f"{dataset}_pattern{pattern}.jsonl"
        if not raw_file.exists():
            raise FileNotFoundError(f"Missing baseline file: {raw_file}")

        # If Pattern 4, load recomputed u_F mapping
        recomp_map: Dict[str, Dict[str, Any]] = {}
        if pattern == 4:
            recomp_file = recomp_path / s_dir / f"{dataset}_pattern4.jsonl"
            if not recomp_file.exists():
                raise FileNotFoundError(f"Missing recomputed P4 file: {recomp_file}")
            with open(recomp_file, "r", encoding="utf-8") as rf:
                for line in rf:
                    if line.strip():
                        r_obj = json.loads(line)
                        recomp_map[str(r_obj["query_id"])] = r_obj

        # Load audited pre-scored mapping for this dataset and run
        scores_dir = audits_path / f"{dataset}_v11_scores"
        score_file = scores_dir / s_dir / f"{dataset}_pattern{pattern}.jsonl"
        if not score_file.exists():
            raise FileNotFoundError(f"Missing audited score file: {score_file}")

        score_map: Dict[str, Dict[str, Any]] = {}
        with open(score_file, "r", encoding="utf-8") as sf:
            for line in sf:
                if line.strip():
                    s_obj = json.loads(line)
                    score_map[str(s_obj["query_id"])] = s_obj

        logger.info("Processing %s / %s_pattern%d (seed %d)...", s_dir, dataset, pattern, seed)

        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = str(record.get("query_id", ""))

                # Baseline uncertainty metrics
                u_L = record.get("u_L")
                u_C = record.get("u_C")
                u_R = record.get("u_R")

                # Handle Pattern 4 recomputed u_F
                if pattern == 4:
                    recomp_rec = recomp_map.get(qid, {})
                    u_F = recomp_rec.get("u_F")
                    u_F_orig = recomp_rec.get("u_F_original", record.get("u_F"))
                    span_category = recomp_rec.get("span_category", "")
                    raw_claims_cnt = recomp_rec.get("raw_claims_count", 0)
                    span_claims_cnt = recomp_rec.get("span_claims_count", 0)
                else:
                    u_F = record.get("u_F")
                    u_F_orig = u_F
                    span_category = "baseline"
                    raw_claims_cnt = 0
                    span_claims_cnt = 0

                # Standardize u_R for iterative patterns on immediate consensus
                if pattern in (2, 3, 4) and u_R is None:
                    if record.get("termination_reason") == "completed":
                        u_R = 0.0

                # Recompute U_surface and U_composite strictly per Doc 03 v11 freeze
                u_P = None  # Deactivated across all patterns in v11 (w_P = 0.00)
                U_comp = compute_composite(
                    pattern_id=pattern,
                    u_L=u_L,
                    u_F=u_F,
                    u_P=u_P,
                    u_R=u_R,
                    u_C=u_C,
                    dataset=dataset,
                )

                if dataset in ("finqa", "popqa"):
                    U_surf = u_F  # u_L is N/A; surface score is strictly u_F
                else:
                    U_surf = compute_surface(u_L=u_L, u_F=u_F, u_P=None)

                # Ground truth scoring fields from audited records
                accuracy: Optional[float] = None
                headline_bin: str = ""
                score_popqa_raw: Optional[float] = None
                strict_prompt: Optional[float] = None
                loose_prompt: Optional[float] = None
                strict_instruction: Optional[float] = None
                loose_instruction: Optional[float] = None
                finqa_extracted_value: Any = None
                finqa_gold_answer: str = ""
                finqa_abstain: Optional[bool] = None

                scored_rec = score_map.get(qid, {})

                if dataset == "finqa":
                    match_val = float(scored_rec.get("match", 0.0))
                    accuracy = match_val
                    headline_bin = str(int(match_val))
                    finqa_extracted_value = scored_rec.get("finqa_extracted_value")
                    finqa_gold_answer = str(scored_rec.get("finqa_gold_answer", ""))
                    finqa_abstain = scored_rec.get("finqa_abstain", False)

                elif dataset == "popqa":
                    # Primary metric: Word-boundary containment (Doc 09)
                    match_val = float(scored_rec.get("popqa_word_boundary", scored_rec.get("match", 0.0)))
                    accuracy = match_val
                    headline_bin = str(int(match_val))
                    score_popqa_raw = float(scored_rec.get("popqa_score_raw", scored_rec.get("popqa_naive_substring", match_val)))

                elif dataset == "ifeval":
                    # Primary metric: Official Google Research strict prompt evaluation
                    strict_prompt = float(scored_rec.get("strict_prompt", 0.0))
                    loose_prompt = float(scored_rec.get("loose_prompt", 0.0))
                    strict_instruction = float(scored_rec.get("strict_instruction", 0.0))
                    loose_instruction = float(scored_rec.get("loose_instruction", 0.0))
                    accuracy = strict_prompt
                    headline_bin = str(int(strict_prompt))

                elif dataset == "livebench":
                    accuracy = float(scored_rec.get("accuracy", scored_rec.get("match", 0.0)))
                    headline_bin = ""  # LiveBench uses continuous accuracy

                row = {
                    "query_id": qid,
                    "dataset": dataset,
                    "pattern": str(pattern),
                    "seed": seed,
                    "headline_bin": headline_bin,
                    "accuracy": accuracy,
                    "u_F": u_F,
                    "u_F_original": u_F_orig,
                    "u_L": u_L,
                    "u_C": u_C,
                    "u_R": u_R,
                    "u_P": u_P,
                    "U_surface": U_surf,
                    "U_composite": U_comp,
                    "span_category": span_category,
                    "raw_claims_count": raw_claims_cnt,
                    "span_claims_count": span_claims_cnt,
                    "termination_reason": record.get("termination_reason", ""),
                    "tokens_prompt": record.get("tokens_prompt", 0),
                    "tokens_completion": record.get("tokens_completion", 0),
                    "latency_ms": record.get("latency_ms", 0.0),
                    "model_name": record.get("model_name", "google/gemma-4-12b-it"),
                    "provider": record.get("provider", "hosted_vllm"),
                    "truncation_flag": record.get("truncation_flag", False),
                    "score_popqa_raw": score_popqa_raw,
                    "strict_prompt": strict_prompt,
                    "loose_prompt": loose_prompt,
                    "strict_instruction": strict_instruction,
                    "loose_instruction": loose_instruction,
                    "finqa_extracted_value": finqa_extracted_value,
                    "finqa_gold_answer": finqa_gold_answer,
                    "finqa_abstain": finqa_abstain,
                    "final_answer": record.get("final_answer", ""),
                }
                all_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "dataset",
        "pattern",
        "seed",
        "headline_bin",
        "accuracy",
        "u_F",
        "u_F_original",
        "u_L",
        "u_C",
        "u_R",
        "u_P",
        "U_surface",
        "U_composite",
        "span_category",
        "raw_claims_count",
        "span_claims_count",
        "termination_reason",
        "tokens_prompt",
        "tokens_completion",
        "latency_ms",
        "model_name",
        "provider",
        "truncation_flag",
        "score_popqa_raw",
        "strict_prompt",
        "loose_prompt",
        "strict_instruction",
        "loose_instruction",
        "finqa_extracted_value",
        "finqa_gold_answer",
        "finqa_abstain",
        "final_answer",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    logger.info("Successfully generated %s with %d records.", out_path, len(all_rows))
    return all_rows


if __name__ == "__main__":
    build_v11_table()
