"""
Acceptance test verifying IFEval reproduction against official Google Research harness.

Validates that:
1. All 900 IFEval records in data/audits/ifeval_v11_scores/ are properly scored.
2. The exact cell counts match the official results:
   - P1: 133/150 (88.67%)
   - P2: 130/150 (86.67%)
   - P3: 135/150 (90.00%)
   - P4 s1234 (span): 119/150 (79.33%)
   - P4 s2345 (span): 122/150 (81.33%)
   - P4 s3456 (span): 117/150 (78.00%)
3. P4 sensitivity (full response) reproduces the ~63.6% baseline.
4. Row-by-row agreement with Sanity_Check/ifeval_official_harness_per_item.csv exceeds 99.7%.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "data/audits/ifeval_v11_scores"
SANITY_CSV = PROJECT_ROOT.parent / "Sanity_Check/ifeval_official_harness_per_item.csv"


def test_ifeval_manifest_cell_counts():
    manifest_path = AUDIT_DIR / "manifest.json"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["total_records"] == 900
    assert manifest["total_files"] == 6
    assert manifest["harness_provenance"]["vendor_package"] == "src.scoring.vendor.google_ifeval"
    assert manifest["harness_provenance"]["upstream_commit"] == "e6890f85757dd84e27ca6df2dd30651dafad28e0"
    assert manifest["harness_provenance"]["deterministic_seed"] == 0

    cells = manifest["cells"]
    assert cells["ifeval_p1_s1234"]["correct"] == 133
    assert cells["ifeval_p2_s1234"]["correct"] == 130
    assert cells["ifeval_p3_s1234"]["correct"] == 135
    assert cells["ifeval_p4_s1234"]["correct"] == 119
    assert cells["ifeval_p4_s2345"]["correct"] == 122
    assert cells["ifeval_p4_s3456"]["correct"] == 117

    # P4 sensitivity full response
    assert cells["ifeval_p4_s1234"]["correct_full_response"] == 94
    assert cells["ifeval_p4_s2345"]["correct_full_response"] == 98
    assert cells["ifeval_p4_s3456"]["correct_full_response"] == 94


def test_ifeval_row_by_row_reproduction():
    """Verify row-by-row agreement against the sanity check CSV from Alvaro / Fedi."""
    if not SANITY_CSV.exists():
        pytest.skip(f"Sanity check CSV not found at {SANITY_CSV}")

    ref_df = pd.read_csv(SANITY_CSV)
    
    # Check each cell in audit dir
    runs = [
        (1, 1234, "report_run_a100/ifeval_pattern1.jsonl"),
        (2, 1234, "report_run_a100/ifeval_pattern2.jsonl"),
        (3, 1234, "report_run_a100/ifeval_pattern3.jsonl"),
        (4, 1234, "report_run_a100/ifeval_pattern4.jsonl"),
        (4, 2345, "report_run_a100_seed2345/ifeval_pattern4.jsonl"),
        (4, 3456, "report_run_a100_seed3456/ifeval_pattern4.jsonl"),
    ]

    total_rows = 0
    matching_rows = 0

    for pat, seed, rel_path in runs:
        audit_file = AUDIT_DIR / rel_path
        assert audit_file.exists()
        
        with open(audit_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total_rows += 1
                rec = json.loads(line)
                qid = int(rec["query_id"])
                my_match = int(rec["strict_prompt"])
                
                # In sanity CSV, P1-P3 have text_evaluated == 'final_answer', P4 span has 'span_after_verdict'
                text_type = "span_after_verdict" if pat == 4 else "final_answer"
                sub = ref_df[
                    (ref_df["pattern"] == pat) &
                    (ref_df["seed"] == seed) &
                    (ref_df["text_evaluated"] == text_type) &
                    (ref_df["query_id"] == qid)
                ]
                assert len(sub) == 1
                ref_match = int(sub.iloc[0]["strict_prompt_pass"])
                
                if my_match == ref_match:
                    matching_rows += 1

    # In 900 rows, exactly 1 row differs due to langdetect unseeded vs seeded=0 (qid 119 in seed 2345)
    # Agreement must be >= 99.8% (at least 899/900)
    agreement = matching_rows / total_rows
    assert matching_rows >= 899, f"Expected at least 899 matches, got {matching_rows}/{total_rows} ({agreement:.4%})"
