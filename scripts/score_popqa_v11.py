"""Score PopQA baseline runs and produce enriched v11 audit records.

Reads baseline runs in data/report_run_a100*/ (read-only).
Preserves 100% of the baseline telemetry record and appends scoring audit fields.
Outputs enriched audit records to data/audits/popqa_v11_scores/.
"""

import ast
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

from src.scoring.popqa_scorer import PopqaScorer, _extract_answer

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


def load_popqa_references() -> Dict[str, Any]:
    from datasets import load_dataset

    logger.info("Loading PopQA canonical references from Hugging Face...")
    popqa_ds = load_dataset("akariasai/PopQA", split="test")
    return {str(item["id"]): item for item in popqa_ds}


def score_popqa_all(
    base_dir: Path,
    out_dir: Path,
    references: Dict[str, Any],
) -> Dict[str, Any]:
    scorer = PopqaScorer()
    all_rows = []
    summary_by_cell = {}

    out_dir.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        s_dir = run["seed_dir"]
        seed = run["seed"]
        pattern = run["pattern"]
        cell_id = f"popqa_p{pattern}_s{seed}"

        raw_file = base_dir / s_dir / f"popqa_pattern{pattern}.jsonl"
        if not raw_file.exists():
            raise FileNotFoundError(f"Missing baseline raw file: {raw_file}")

        target_dir = out_dir / s_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_jsonl = target_dir / f"popqa_pattern{pattern}.jsonl"

        cell_rows = []
        cell_matches = []

        with open(raw_file, "r", encoding="utf-8") as f, open(target_jsonl, "w", encoding="utf-8") as out_f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = str(record["query_id"])
                p_ref = references.get(qid, {})

                final_ans = record.get("final_answer", "")
                extracted = _extract_answer(final_ans)
                ref_ans = p_ref.get("answer", p_ref.get("reference", ""))
                possible = p_ref.get("possible_answers", None)
                if possible is not None:
                    try:
                        aliases = ast.literal_eval(possible)
                    except (ValueError, SyntaxError):
                        aliases = [ref_ans]
                else:
                    aliases = [ref_ans]

                res = scorer.score(record, p_ref)
                match_val = float(res.get("match", 0.0))
                raw_match = float(res.get("score_popqa_raw", 0.0))
                abstain = bool(not extracted or not extracted.strip())

                # Enriched record: preserves 100% of baseline telemetry and appends audit fields
                enriched_record = dict(record)
                enriched_record["match"] = match_val
                enriched_record["popqa_extracted_answer"] = extracted
                enriched_record["popqa_gold_answer"] = ref_ans
                enriched_record["popqa_aliases"] = aliases
                enriched_record["popqa_score_raw"] = raw_match
                enriched_record["popqa_question"] = p_ref.get("question", "")
                enriched_record["popqa_prop"] = p_ref.get("prop", "")
                enriched_record["popqa_subj"] = p_ref.get("subj", "")
                enriched_record["popqa_abstain"] = abstain
                enriched_record["popqa_scored_v11"] = True

                out_f.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")
                cell_rows.append(enriched_record)
                cell_matches.append(match_val)

        all_rows.extend(cell_rows)
        summary_by_cell[cell_id] = {
            "total": len(cell_rows),
            "correct": int(np.sum(cell_matches)),
            "accuracy": float(np.mean(cell_matches)),
            "accuracy_pct": float(np.mean(cell_matches) * 100.0),
        }
        logger.info("Scored %s: %d/%d (%.2f%%)", cell_id, int(np.sum(cell_matches)), len(cell_rows), np.mean(cell_matches) * 100.0)

    # Write consolidated CSV for manual human inspection
    csv_file = out_dir / "popqa_audit_all_900.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "query_id", "pattern", "seed", "match", "popqa_abstain",
            "popqa_extracted_answer", "popqa_gold_answer", "popqa_aliases",
            "popqa_question", "popqa_score_raw", "tokens_prompt", "tokens_completion",
            "latency_ms", "final_answer_preview"
        ])
        for r in all_rows:
            writer.writerow([
                r["query_id"],
                r["pattern"],
                r["seed"],
                r["match"],
                r["popqa_abstain"],
                r["popqa_extracted_answer"].replace("\n", " ")[:150],
                r["popqa_gold_answer"],
                " | ".join(r["popqa_aliases"]) if isinstance(r["popqa_aliases"], list) else str(r["popqa_aliases"]),
                r["popqa_question"],
                r["popqa_score_raw"],
                r.get("tokens_prompt", 0),
                r.get("tokens_completion", 0),
                r.get("latency_ms", 0.0),
                r["final_answer"].replace("\n", " ")[:200],
            ])

    manifest = {
        "dataset": "popqa",
        "total_records": len(all_rows),
        "total_files": len(RUNS),
        "schema_pattern": "enriched_baseline_record",
        "cells": summary_by_cell,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    logger.info("PopQA audit completed: %d enriched records written to %s", len(all_rows), out_dir)
    return manifest


def main():
    base_dir = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "data" / "audits" / "popqa_v11_scores"
    refs = load_popqa_references()
    score_popqa_all(base_dir, out_dir, refs)


if __name__ == "__main__":
    main()
