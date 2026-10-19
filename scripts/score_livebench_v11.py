"""Score LiveBench baseline runs and produce enriched v11 audit records.

Reads baseline runs in data/report_run_a100*/ (read-only).
Preserves 100% of the baseline telemetry record and appends scoring audit fields.
Outputs enriched audit records to data/audits/livebench_v11_scores/.
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.livebench_scorer import LiveBenchScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RUNS = [
    # Seed 1234 (all 4 patterns)
    {"seed_dir": "report_run_a100", "seed": 1234, "pattern": 1},
    {"seed_dir": "report_run_a100", "seed": 1234, "pattern": 2},
    {"seed_dir": "report_run_a100", "seed": 1234, "pattern": 3},
    {"seed_dir": "report_run_a100", "seed": 1234, "pattern": 4},
    # Seed 2345 (Pattern 4)
    {"seed_dir": "report_run_a100_seed2345", "seed": 2345, "pattern": 4},
    # Seed 3456 (Pattern 4)
    {"seed_dir": "report_run_a100_seed3456", "seed": 3456, "pattern": 4},
]


def load_livebench_scorer() -> LiveBenchScorer:
    logger.info("Initializing official LiveBench reasoning scorer...")
    return LiveBenchScorer(mock_mode=False)


def score_livebench_all(
    base_dir: Path,
    out_dir: Path,
    scorer: LiveBenchScorer,
) -> Dict[str, Any]:
    all_rows = []
    summary_by_cell = {}

    out_dir.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        s_dir = run["seed_dir"]
        seed = run["seed"]
        pattern = run["pattern"]
        cell_id = f"livebench_p{pattern}_s{seed}"

        raw_file = base_dir / s_dir / f"livebench_pattern{pattern}.jsonl"
        if not raw_file.exists():
            raise FileNotFoundError(f"Missing baseline raw file: {raw_file}")

        target_dir = out_dir / s_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_jsonl = target_dir / f"livebench_pattern{pattern}.jsonl"

        cell_rows = []
        cell_matches = []

        with open(raw_file, "r", encoding="utf-8") as f, open(target_jsonl, "w", encoding="utf-8") as out_f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = str(record["query_id"])

                final_ans = record.get("final_answer", "")
                res = scorer.score(record, {})
                acc = float(res.get("accuracy", 0.0))
                parse_err = bool(res.get("parse_error", False))

                q_obj = scorer.questions_map.get(qid, {})
                gold_ans = str(q_obj.get("ground_truth", q_obj.get("answer", "")))
                task_name = str(q_obj.get("task_name", q_obj.get("category", "")))

                extracted_slot = final_ans
                if "***" in final_ans:
                    parts = final_ans.split("***")
                    if len(parts) >= 3:
                        extracted_slot = parts[-2].strip()

                # Enriched record: preserves 100% of baseline telemetry and appends audit fields
                enriched_record = dict(record)
                enriched_record["match"] = acc
                enriched_record["livebench_extracted_slot"] = extracted_slot
                enriched_record["livebench_gold_answer"] = gold_ans
                enriched_record["livebench_task_name"] = task_name
                enriched_record["livebench_abstain"] = parse_err
                enriched_record["livebench_scored_v11"] = True

                out_f.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")
                cell_rows.append(enriched_record)
                cell_matches.append(acc)

        all_rows.extend(cell_rows)
        summary_by_cell[cell_id] = {
            "total": len(cell_rows),
            "correct_equivalent": float(np.sum(cell_matches)),
            "accuracy": float(np.mean(cell_matches)),
            "accuracy_pct": float(np.mean(cell_matches) * 100.0),
        }
        logger.info("Scored %s: N=%d, Acc=%.2f%%", cell_id, len(cell_rows), np.mean(cell_matches) * 100.0)

    # Write consolidated CSV for manual human inspection
    csv_file = out_dir / "livebench_audit_all_900.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "query_id", "pattern", "seed", "accuracy", "livebench_abstain",
            "task_name", "livebench_extracted_slot", "livebench_gold_answer",
            "tokens_prompt", "tokens_completion", "latency_ms", "final_answer_preview"
        ])
        for r in all_rows:
            writer.writerow([
                r["query_id"],
                r["pattern"],
                r["seed"],
                r["match"],
                r["livebench_abstain"],
                r["livebench_task_name"],
                r["livebench_extracted_slot"].replace("\n", " ")[:150],
                r["livebench_gold_answer"].replace("\n", " ")[:150],
                r.get("tokens_prompt", 0),
                r.get("tokens_completion", 0),
                r.get("latency_ms", 0.0),
                r["final_answer"].replace("\n", " ")[:200],
            ])

    manifest = {
        "dataset": "livebench",
        "total_records": len(all_rows),
        "total_files": len(RUNS),
        "schema_pattern": "enriched_baseline_record",
        "cells": summary_by_cell,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    logger.info("LiveBench audit completed: %d enriched records written to %s", len(all_rows), out_dir)
    return manifest


def main():
    base_dir = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "data" / "audits" / "livebench_v11_scores"
    scorer = load_livebench_scorer()
    score_livebench_all(base_dir, out_dir, scorer)


if __name__ == "__main__":
    main()
