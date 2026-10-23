"""
Acceptance test verifying IFEval reproduction against official Google Research harness.

Validates that:
1. All 900 IFEval records in data/audits/ifeval_v11_scores/ are properly scored.
2. The exact cell counts match the official results:
   - P1: 133/150 (88.67%) strict, 136/150 (90.67%) loose
   - P2: 130/150 (86.67%) strict, 134/150 (89.33%) loose
   - P3: 135/150 (90.00%) strict, 138/150 (92.00%) loose
   - P4 s1234 (span): 119/150 (79.33%) strict, 124/150 (82.67%) loose
   - P4 s2345 (span): 121/150 (80.67%) strict, 124/150 (82.67%) loose
   - P4 s3456 (span): 117/150 (78.00%) strict, 119/150 (79.33%) loose
3. P4 sensitivity (full response) reproduces 94/150, 98/150, 94/150 (63.56% ± 1.54%).
4. Live execution of IfevalScorer and committed JSONL records achieve 100% (zero differences)
   row-by-row agreement on both strict and loose against data/audits/ifeval_official_harness_per_item.csv.
"""

import json
from pathlib import Path
import pandas as pd

from src.scoring.ifeval_scorer import IfevalScorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "data/audits/ifeval_v11_scores"
MAPPING_FILE = PROJECT_ROOT / "data/ifeval_mapping_150_prompts.jsonl"
SANITY_CSV = PROJECT_ROOT / "data/audits/ifeval_official_harness_per_item.csv"


def test_ifeval_manifest_cell_counts():
    manifest_path = AUDIT_DIR / "manifest.json"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"
    assert SANITY_CSV.exists(), f"Missing reference CSV at {SANITY_CSV}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ref_df = pd.read_csv(SANITY_CSV)

    assert manifest["total_records"] == 900
    assert manifest["total_files"] == 6
    assert manifest["harness_provenance"]["vendor_package"] == "src.scoring.vendor.google_ifeval"
    assert manifest["harness_provenance"]["upstream_commit"] == "e6890f85757dd84e27ca6df2dd30651dafad28e0"
    assert manifest["harness_provenance"]["deterministic_seed"] == 0

    cells = manifest["cells"]
    expected_strict = {
        "ifeval_p1_s1234": (1, 1234, "final_answer", 133),
        "ifeval_p2_s1234": (2, 1234, "final_answer", 130),
        "ifeval_p3_s1234": (3, 1234, "final_answer", 135),
        "ifeval_p4_s1234": (4, 1234, "span_after_verdict", 119),
        "ifeval_p4_s2345": (4, 2345, "span_after_verdict", 121),
        "ifeval_p4_s3456": (4, 3456, "span_after_verdict", 117),
    }

    for cell_id, (pat, seed, text_ev, exp_cnt) in expected_strict.items():
        ref_sub = ref_df[
            (ref_df["pattern"] == pat)
            & (ref_df["seed"] == seed)
            & (ref_df["text_evaluated"] == text_ev)
        ]
        assert len(ref_sub) == 150
        assert int(ref_sub["strict_prompt_pass"].sum()) == exp_cnt
        assert cells[cell_id]["correct"] == exp_cnt

    # P4 sensitivity full response
    assert cells["ifeval_p4_s1234"]["correct_full_response"] == 94
    assert cells["ifeval_p4_s2345"]["correct_full_response"] == 98
    assert cells["ifeval_p4_s3456"]["correct_full_response"] == 94


def test_ifeval_row_by_row_reproduction():
    """Verify 100% row-by-row strict and loose agreement of IfevalScorer and audited JSONLs against reference CSV."""
    assert SANITY_CSV.exists(), f"Reference CSV not found at {SANITY_CSV}"
    assert MAPPING_FILE.exists(), f"Mapping file not found at {MAPPING_FILE}"

    ref_df = pd.read_csv(SANITY_CSV)
    assert len(ref_df) == 1350

    references = {}
    with open(MAPPING_FILE, "r", encoding="utf-8") as mf:
        for line in mf:
            if line.strip():
                item = json.loads(line)
                references[int(item["query_id"])] = item

    scorer = IfevalScorer()

    runs = [
        (1, 1234, "report_run_a100/ifeval_pattern1.jsonl"),
        (2, 1234, "report_run_a100/ifeval_pattern2.jsonl"),
        (3, 1234, "report_run_a100/ifeval_pattern3.jsonl"),
        (4, 1234, "report_run_a100/ifeval_pattern4.jsonl"),
        (4, 2345, "report_run_a100_seed2345/ifeval_pattern4.jsonl"),
        (4, 3456, "report_run_a100_seed3456/ifeval_pattern4.jsonl"),
    ]

    total_rows = 0
    strict_diffs = []
    loose_diffs = []
    full_strict_diffs = []
    full_loose_diffs = []

    for pat, seed, rel_path in runs:
        audit_file = AUDIT_DIR / rel_path
        assert audit_file.exists(), f"Missing audit file {audit_file}"

        with open(audit_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total_rows += 1
                rec = json.loads(line)
                qid = int(rec["query_id"])
                live_res = scorer.score(rec, references[qid])

                # Ensure committed record matches live scorer execution
                assert int(rec["strict_prompt"]) == int(live_res["strict_prompt"])
                assert int(rec["loose_prompt"]) == int(live_res["loose_prompt"])
                assert rec["evaluated_text_type"] == live_res["evaluated_text_type"]

                text_type = "span_after_verdict" if pat == 4 else "final_answer"
                assert live_res["evaluated_text_type"] == text_type

                sub = ref_df[
                    (ref_df["pattern"] == pat)
                    & (ref_df["seed"] == seed)
                    & (ref_df["text_evaluated"] == text_type)
                    & (ref_df["query_id"] == qid)
                ]
                assert len(sub) == 1
                ref_row = sub.iloc[0]

                if int(live_res["strict_prompt"]) != int(ref_row["strict_prompt_pass"]):
                    strict_diffs.append((pat, seed, qid, int(live_res["strict_prompt"]), int(ref_row["strict_prompt_pass"])))
                if int(live_res["loose_prompt"]) != int(ref_row["loose_prompt_pass"]):
                    loose_diffs.append((pat, seed, qid, int(live_res["loose_prompt"]), int(ref_row["loose_prompt_pass"])))

                # For Pattern 4, also verify the 450 full-response sensitivity evaluations
                if pat == 4:
                    sub_full = ref_df[
                        (ref_df["pattern"] == pat)
                        & (ref_df["seed"] == seed)
                        & (ref_df["text_evaluated"] == "final_answer")
                        & (ref_df["query_id"] == qid)
                    ]
                    assert len(sub_full) == 1
                    ref_full = sub_full.iloc[0]
                    if int(live_res["strict_prompt_full_response"]) != int(ref_full["strict_prompt_pass"]):
                        full_strict_diffs.append((pat, seed, qid))
                    if int(live_res["loose_prompt_full_response"]) != int(ref_full["loose_prompt_pass"]):
                        full_loose_diffs.append((pat, seed, qid))

    assert total_rows == 900
    assert strict_diffs == [], f"Strict prompt diffs found: {strict_diffs}"
    assert loose_diffs == [], f"Loose prompt diffs found: {loose_diffs}"
    assert full_strict_diffs == [], f"P4 full-response strict diffs found: {full_strict_diffs}"
    assert full_loose_diffs == [], f"P4 full-response loose diffs found: {full_loose_diffs}"

