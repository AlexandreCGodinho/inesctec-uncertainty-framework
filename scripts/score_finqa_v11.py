#!/usr/bin/env python3
"""
FinQA 900 Automated Scoring Script (Phase 2 Task 5B).

Scores all 900 FinQA telemetry records across:
- Pattern 1 (P1 Governed): 150 items (seed 1234)
- Pattern 2 (P2 Pipeline): 150 items (seed 1234)
- Pattern 3 (P3 Reflexive): 150 items (seed 1234)
- Pattern 4 (P4 Deliberation): 150 items (seed 1234)
- Pattern 4 (P4 Deliberation): 150 items (seed 2345)
- Pattern 4 (P4 Deliberation): 150 items (seed 3456)
Total: 900 records.

Uses the unified extraction rule from src/scoring/finqa_numeric_scorer.py
benchmarked and signed off in Phase 2 Task 5A.

Outputs:
- Scored records in data/audits/finqa_v11_scores/
- Audit summary in data/audits/finqa_v11_scores/finqa_scoring_summary.json
- Markdown report in data/audits/finqa_v11_scores/FINQA_SCORING_REPORT_V11.md
Historical baseline files in data/report_run_a100*/ remain frozen and untouched.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.finqa_numeric_scorer import (
    finqa_numeric_scorer,
    unified_extract_numeric_answer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("score_finqa_v11")

PINNED_FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
FINQA_TEST_URL = (
    f"https://raw.githubusercontent.com/czyssrs/FinQA/{PINNED_FINQA_COMMIT}/dataset/test.json"
)

TARGET_RUNS = [
    {"seed_dir": "report_run_a100", "pattern": 1, "seed": 1234},
    {"seed_dir": "report_run_a100", "pattern": 2, "seed": 1234},
    {"seed_dir": "report_run_a100", "pattern": 3, "seed": 1234},
    {"seed_dir": "report_run_a100", "pattern": 4, "seed": 1234},
    {"seed_dir": "report_run_a100_seed2345", "pattern": 4, "seed": 2345},
    {"seed_dir": "report_run_a100_seed3456", "pattern": 4, "seed": 3456},
]


def load_finqa_references(cache_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load official FinQA references keyed by query_id (id).
    Uses local cache if available, otherwise downloads from pinned GitHub commit.
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached FinQA references from %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["id"]: item.get("qa", {}) for item in data if "id" in item}

    logger.info("Fetching canonical FinQA test set from %s", FINQA_TEST_URL)
    req = urllib.request.Request(FINQA_TEST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_bytes = resp.read()
        data = json.loads(raw_bytes.decode("utf-8"))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(raw_bytes.decode("utf-8"))
        logger.info("Saved references cache to %s", cache_path)

    return {item["id"]: item.get("qa", {}) for item in data if "id" in item}


def score_single_file(
    input_file: Path,
    output_file: Path,
    references: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Score a single FinQA telemetry file."""
    if not input_file.exists():
        logger.warning("Input file %s does not exist, skipping.", input_file)
        return {"file": input_file.name, "total": 0, "matches": 0, "accuracy_pct": 0.0}

    records: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    scored_records: List[Dict[str, Any]] = []
    matches = 0
    abstentions = 0

    for rec in records:
        qid = str(rec.get("query_id", ""))
        ref = references.get(qid, {})

        score_res = finqa_numeric_scorer(rec, ref)
        match_val = float(score_res.get("match", 0.0))

        val, unit, abstain = unified_extract_numeric_answer(rec.get("final_answer", ""))

        updated = dict(rec)
        updated["match"] = match_val
        updated["finqa_extracted_value"] = val
        updated["finqa_extracted_unit"] = unit
        updated["finqa_abstain"] = abstain
        updated["finqa_exe_ans"] = ref.get("exe_ans")
        updated["finqa_gold_answer"] = ref.get("answer")
        updated["finqa_scored_v11"] = True

        scored_records.append(updated)
        if match_val == 1.0:
            matches += 1
        if abstain:
            abstentions += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for r in scored_records:
            f.write(json.dumps(r) + "\n")

    total = len(records)
    acc = (matches / total * 100.0) if total > 0 else 0.0

    logger.info(
        "Scored %s: %d/%d matches (%.2f%%), %d abstentions",
        input_file.name,
        matches,
        total,
        acc,
        abstentions,
    )

    return {
        "file": input_file.name,
        "input_path": str(input_file),
        "output_path": str(output_file),
        "total": total,
        "matches": matches,
        "accuracy_pct": round(acc, 2),
        "abstentions": abstentions,
    }


def run_finqa_scoring_batch(
    base_dir: str = "data",
    output_dir: str = "data/audits/finqa_v11_scores",
    cache_path: Optional[str] = "data/finqa_references_cache.json",
) -> Dict[str, Any]:
    """Execute full 900-item scoring batch."""
    base_path = Path(base_dir)
    out_path = Path(output_dir)
    c_path = Path(cache_path) if cache_path else None

    references = load_finqa_references(c_path)
    logger.info("Loaded %d FinQA reference entries.", len(references))

    summary: Dict[str, Any] = {
        "task": "Phase 2 Task 5B: FinQA 900 Automated Scoring Batch",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pinned_finqa_commit": PINNED_FINQA_COMMIT,
        "scoring_method": "deterministic_rule_based",
        "llm_judge_invoked": False,
        "telemetry_provenance": "Google Colab Pro+ GPU A100 baseline runs (data/report_run_a100*)",
        "judge_model_baseline": "google/gemma-4-12b-it",
        "u_F_status": "Strictly frozen from official baseline runs; not recomputed or altered",
        "tolerance": {"relative": 0.005, "absolute": 1e-06},
        "files_evaluated": [],
        "overall_total": 0,
        "overall_matches": 0,
        "overall_accuracy_pct": 0.0,
        "pattern_summary": {},
    }

    pattern_aggregates: Dict[int, Dict[str, int]] = {1: {"total": 0, "matches": 0},
                                                      2: {"total": 0, "matches": 0},
                                                      3: {"total": 0, "matches": 0},
                                                      4: {"total": 0, "matches": 0}}

    for run_spec in TARGET_RUNS:
        s_dir = run_spec["seed_dir"]
        pat = run_spec["pattern"]
        seed = run_spec["seed"]

        in_f = base_path / s_dir / f"finqa_pattern{pat}.jsonl"
        out_f = out_path / s_dir / f"finqa_pattern{pat}.jsonl"

        f_res = score_single_file(in_f, out_f, references)
        f_res["pattern"] = pat
        f_res["seed"] = seed
        summary["files_evaluated"].append(f_res)

        summary["overall_total"] += f_res["total"]
        summary["overall_matches"] += f_res["matches"]

        pattern_aggregates[pat]["total"] += f_res["total"]
        pattern_aggregates[pat]["matches"] += f_res["matches"]

    if summary["overall_total"] > 0:
        summary["overall_accuracy_pct"] = round(
            summary["overall_matches"] / summary["overall_total"] * 100.0, 2
        )

    for p, stats in pattern_aggregates.items():
        tot = stats["total"]
        m = stats["matches"]
        acc = round((m / tot * 100.0), 2) if tot > 0 else 0.0
        summary["pattern_summary"][f"Pattern_{p}"] = {
            "total": tot,
            "matches": m,
            "accuracy_pct": acc,
        }

    # Save summary report JSON
    out_path.mkdir(parents=True, exist_ok=True)
    summary_json_path = out_path / "finqa_scoring_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown report
    md_report_path = out_path / "FINQA_SCORING_REPORT_V11.md"
    _generate_markdown_report(summary, md_report_path)

    logger.info("Summary report saved to %s", summary_json_path)
    logger.info("Markdown audit report saved to %s", md_report_path)
    return summary


def _generate_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Generate Markdown audit report for Task 5B."""
    lines = [
        "# FinQA 900 Automated Scoring Report (Phase 2 Task 5B)",
        "",
        f"- **Execution Timestamp:** {summary.get('timestamp')}",
        f"- **Pinned FinQA Commit:** `{summary.get('pinned_finqa_commit')}`",
        f"- **Comparison Tolerance:** `rtol={summary['tolerance']['relative']}`, `atol={summary['tolerance']['absolute']}`",
        f"- **Total Evaluated Records:** {summary.get('overall_total')}",
        f"- **Overall Accurate Matches:** {summary.get('overall_matches')} / {summary.get('overall_total')} ({summary.get('overall_accuracy_pct')}%)",
        "",
        "## 1. Summary by Architectural Pattern",
        "",
        "| Pattern | Description | Total Items | Matches | Accuracy (%) |",
        "|---|---|---|---|---|",
    ]

    pat_names = {
        "Pattern_1": "P1 (Governed Workflow)",
        "Pattern_2": "P2 (Role-Separated Pipeline)",
        "Pattern_3": "P3 (Reflexive Single Agent)",
        "Pattern_4": "P4 (Multi-Perspective Deliberation)",
    }

    for p_key, p_data in summary.get("pattern_summary", {}).items():
        p_name = pat_names.get(p_key, p_key)
        tot = p_data["total"]
        m = p_data["matches"]
        acc = p_data["accuracy_pct"]
        lines.append(f"| {p_name} | {p_key} | {tot} | {m} | **{acc:.2f}%** |")

    lines.extend([
        "",
        "## 2. Detailed Per-File Telemetry Breakdown",
        "",
        "| Seed Directory | Pattern | Seed | Records | Matches | Abstentions | Accuracy (%) |",
        "|---|---|---|---|---|---|---|",
    ])

    for f_info in summary.get("files_evaluated", []):
        s_dir = Path(f_info["input_path"]).parent.name
        pat = f_info.get("pattern")
        seed = f_info.get("seed")
        tot = f_info.get("total")
        m = f_info.get("matches")
        abs_cnt = f_info.get("abstentions")
        acc = f_info.get("accuracy_pct")
        lines.append(
            f"| `{s_dir}` | P{pat} | {seed} | {tot} | {m} | {abs_cnt} | **{acc:.2f}%** |"
        )

    lines.extend([
        "",
        "## 3. Operational Verification and Experimental Parity",
        "- **Scoring Method:** Deterministic rule-based evaluation (regex extraction + floating-point tolerance `rtol=0.005`, `atol=1e-06`). Zero LLM judge calls were invoked.",
        "- **Telemetry Provenance:** Evaluated answers and runtime uncertainty metrics (u_F, u_L, tokens, latency) originate strictly from official Colab Pro+ GPU A100 runs (`data/report_run_a100*`) generated with `google/gemma-4-12b-it` in bfloat16.",
        "- **Factual Uncertainty (u_F) Parity:** u_F values remain strictly frozen as generated on the A100. They were not recomputed or altered in Task 5B, guaranteeing 100% parity and comparability with the other datasets.",
        "- Historical baseline telemetry files in `data/report_run_a100*/` remain strictly frozen.",
        "- Scored outputs with extraction audit fields are stored in `data/audits/finqa_v11_scores/`.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score 900 FinQA telemetry records.")
    parser.add_argument("--base-dir", default="data", help="Base input directory containing report_run_a100*")
    parser.add_argument("--output-dir", default="data/audits/finqa_v11_scores", help="Output audit directory")
    parser.add_argument("--cache-path", default="data/finqa_references_cache.json", help="Path to cache references")

    args = parser.parse_args()
    run_finqa_scoring_batch(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        cache_path=args.cache_path,
    )
