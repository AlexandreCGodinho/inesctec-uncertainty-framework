"""Score IFEval baseline runs and produce enriched v11 audit records.

Reads baseline runs in data/report_run_a100*/ (read-only).
Preserves 100% of the baseline telemetry record and appends scoring audit fields.
Outputs enriched audit records to data/audits/ifeval_v11_scores/.
"""

import csv
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.ifeval_scorer import IfevalScorer
from src.scoring.answer_span import extract_answer_span

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


def load_ifeval_references() -> Dict[str, Any]:
    mapping_file = PROJECT_ROOT / "data/ifeval_mapping_150_prompts.jsonl"
    logger.info("Loading IFEval sampled references from %s...", mapping_file)
    ifeval_refs = {}
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    ifeval_refs[str(item["query_id"])] = item
    else:
        logger.warning("Mapping file not found at %s. Falling back to Google Research raw input_data.jsonl...", mapping_file)
        url = "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            idx = 0
            for line in resp:
                if line.strip():
                    item = json.loads(line.decode("utf-8"))
                    ifeval_refs[str(idx)] = item
                    idx += 1
    logger.info("Loaded %d IFEval references into memory.", len(ifeval_refs))
    return ifeval_refs


def score_ifeval_all(
    base_dir: Path,
    out_dir: Path,
    references: Dict[str, Any],
) -> Dict[str, Any]:
    scorer = IfevalScorer()
    all_rows = []
    summary_by_cell = {}

    out_dir.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        s_dir = run["seed_dir"]
        seed = run["seed"]
        pattern = run["pattern"]
        cell_id = f"ifeval_p{pattern}_s{seed}"

        raw_file = base_dir / s_dir / f"ifeval_pattern{pattern}.jsonl"
        if not raw_file.exists():
            raise FileNotFoundError(f"Missing baseline raw file: {raw_file}")

        target_dir = out_dir / s_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_jsonl = target_dir / f"ifeval_pattern{pattern}.jsonl"

        cell_rows = []
        cell_matches = []

        with open(raw_file, "r", encoding="utf-8") as f, open(target_jsonl, "w", encoding="utf-8") as out_f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = str(record["query_id"])
                i_ref = references.get(qid, {})

                final_ans = record.get("final_answer", "")
                extracted_span, span_cat = extract_answer_span(final_ans)
                prompt_text = i_ref.get("prompt", "")

                res = scorer.score(record, i_ref)
                strict_prompt = float(res.get("strict_prompt", 0.0))
                loose_prompt = float(res.get("loose_prompt", 0.0))
                strict_inst = float(res.get("strict_instruction", 0.0))
                loose_inst = float(res.get("loose_instruction", 0.0))
                at_least_one = float(res.get("at_least_one_instruction", 0.0))
                strict_prompt_full = float(res.get("strict_prompt_full_response", strict_prompt))
                loose_prompt_full = float(res.get("loose_prompt_full_response", loose_prompt))
                eval_type = res.get("evaluated_text_type", "final_answer")
                abstain = bool(not extracted_span or not extracted_span.strip())

                # Enriched record: preserves 100% of baseline telemetry and appends audit fields
                enriched_record = dict(record)
                enriched_record["match"] = strict_prompt
                enriched_record["ifeval_extracted_span"] = extracted_span
                enriched_record["ifeval_prompt"] = prompt_text
                enriched_record["strict_prompt"] = strict_prompt
                enriched_record["loose_prompt"] = loose_prompt
                enriched_record["strict_prompt_full_response"] = strict_prompt_full
                enriched_record["loose_prompt_full_response"] = loose_prompt_full
                enriched_record["evaluated_text_type"] = eval_type
                enriched_record["at_least_one_instruction"] = at_least_one
                enriched_record["strict_instruction"] = strict_inst
                enriched_record["loose_instruction"] = loose_inst
                enriched_record["instruction_id_list"] = i_ref.get("instruction_id_list", [])
                enriched_record["follow_instruction_list_strict"] = res.get("follow_instruction_list_strict", [])
                enriched_record["follow_instruction_list_loose"] = res.get("follow_instruction_list_loose", [])
                enriched_record["ifeval_span_category"] = span_cat
                enriched_record["ifeval_abstain"] = abstain
                enriched_record["ifeval_scored_v11"] = True

                out_f.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")
                cell_rows.append(enriched_record)
                cell_matches.append(strict_prompt)

        all_rows.extend(cell_rows)
        full_matches = [r["strict_prompt_full_response"] for r in cell_rows]
        summary_by_cell[cell_id] = {
            "total": len(cell_rows),
            "correct": int(np.sum(cell_matches)),
            "accuracy": float(np.mean(cell_matches)),
            "accuracy_pct": float(np.mean(cell_matches) * 100.0),
            "correct_full_response": int(np.sum(full_matches)),
            "accuracy_full_response_pct": float(np.mean(full_matches) * 100.0),
            "evaluated_text_primary": "span_after_verdict" if pattern == 4 else "final_answer",
        }
        logger.info(
            "Scored %s: %d/%d (%.2f%%) [full response sensitivity: %.2f%%]",
            cell_id, int(np.sum(cell_matches)), len(cell_rows), np.mean(cell_matches) * 100.0, np.mean(full_matches) * 100.0
        )

    # Write consolidated CSV for manual human inspection
    csv_file = out_dir / "ifeval_audit_all_900.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "query_id", "pattern", "seed", "match_strict_prompt", "loose_prompt",
            "strict_prompt_full_response", "loose_prompt_full_response", "evaluated_text_type",
            "at_least_one_instruction", "strict_instruction", "loose_instruction",
            "ifeval_abstain", "instruction_ids",
            "tokens_prompt", "tokens_completion", "latency_ms", "extracted_span_preview", "prompt_preview"
        ])
        for r in all_rows:
            writer.writerow([
                r["query_id"],
                r["pattern"],
                r["seed"],
                r["match"],
                r["loose_prompt"],
                r["strict_prompt_full_response"],
                r["loose_prompt_full_response"],
                r["evaluated_text_type"],
                r.get("at_least_one_instruction", 0.0),
                r["strict_instruction"],
                r["loose_instruction"],
                r["ifeval_abstain"],
                " | ".join(r["instruction_id_list"]),
                r.get("tokens_prompt", 0),
                r.get("tokens_completion", 0),
                r.get("latency_ms", 0.0),
                r["ifeval_extracted_span"].replace("\n", " ")[:150],
                r["ifeval_prompt"].replace("\n", " ")[:200],
            ])

    manifest = {
        "dataset": "ifeval",
        "total_records": len(all_rows),
        "total_files": len(RUNS),
        "schema_pattern": "enriched_baseline_record",
        "harness_provenance": {
            "vendor_package": "src.scoring.vendor.google_ifeval",
            "upstream_commit": "e6890f85757dd84e27ca6df2dd30651dafad28e0",
            "deterministic_seed": 0,
            "langdetect_version": "1.0.9",
        },
        "cells": summary_by_cell,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    logger.info("IFEval audit completed: %d enriched records written to %s", len(all_rows), out_dir)
    return manifest


def main():
    base_dir = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "data" / "audits" / "ifeval_v11_scores"
    refs = load_ifeval_references()
    score_ifeval_all(base_dir, out_dir, refs)


if __name__ == "__main__":
    main()
