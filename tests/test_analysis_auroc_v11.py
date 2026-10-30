"""Verification tests for v11 AUROC, sensor diagnostics, FinQA agreement, and appendix tables."""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.add_helper_columns_finqa import parse_label
from scripts.analysis_auroc import (
    add_binary_correctness,
    compute_finqa_agreement,
    compute_finqa_alternative_parsers,
    compute_sensor_auroc_tables,
    compute_silent_failure_auroc,
    compute_uf_coverage_table,
)
from scripts.generate_p4_appendix_tables import build_appendix_tables

REPO_ROOT = Path(__file__).resolve().parent.parent
PER_ITEM_CSV = REPO_ROOT / "data" / "audits" / "per_item_scores_v11.csv"
FINQA_SCORES_DIR = REPO_ROOT / "data" / "audits" / "finqa_v11_scores" / "report_run_a100"


@pytest.fixture(scope="module")
def v11_df() -> pd.DataFrame:
    assert PER_ITEM_CSV.exists(), f"Missing canonical v11 matrix: {PER_ITEM_CSV}"
    df = pd.read_csv(PER_ITEM_CSV)
    assert len(df) == 3600
    return add_binary_correctness(df)


def test_add_binary_correctness_livebench_strict_equality():
    df = pd.DataFrame({
        "dataset": ["livebench", "livebench", "finqa", "finqa"],
        "pattern": ["1", "2", "1", "2"],
        "seed": ["1234", "1234", "1234", "1234"],
        "accuracy": [1.0, 0.375, 1.0, 0.0],
        "headline_bin": [float("nan"), float("nan"), 1.0, 0.0],
    })
    out = add_binary_correctness(df)
    assert out["binary_correct"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_uf_coverage_p4_summary(v11_df: pd.DataFrame):
    cov = compute_uf_coverage_table(v11_df)
    tot = cov["total"]
    assert tot["total"] == 1800
    assert tot["uf_defined"] == 951
    assert tot["coverage_pct"] == pytest.approx(52.8, abs=0.1)
    assert tot["zero_claims"] == 743
    assert tot["parse_error"] == 54
    assert tot["abstention"] == 50
    assert tot["empty_span"] == 2
    assert tot["raw_claims"] == 21471
    assert tot["span_claims"] == 8754
    assert tot["claim_reduction_pct"] == pytest.approx(59.2, abs=0.1)

    by_bm = cov["by_benchmark"]
    assert by_bm["popqa"]["uf_defined"] == 409
    assert by_bm["ifeval"]["uf_defined"] == 386
    assert by_bm["livebench"]["uf_defined"] == 116
    assert by_bm["finqa"]["uf_defined"] == 40


def test_sensor_auroc_and_fdr(v11_df: pd.DataFrame):
    res = compute_sensor_auroc_tables(v11_df, alpha=0.05)

    # 32-test Benjamini-Hochberg family across active individual sensors (8 per benchmark x 4 benchmarks)
    fdr = res["fdr_32_active_tests"]
    assert fdr["n_tests"] == 32
    assert fdr["n_rejected"] == 10

    pooled = res["pooled_sensor_auroc"]
    assert pooled["u_R"]["ifeval"]["P2"] == pytest.approx(0.8219, abs=0.005)
    assert pooled["u_F"]["popqa"]["P4"] == pytest.approx(0.7955, abs=0.005)
    assert pooled["U_composite"]["livebench"]["P4"] == pytest.approx(0.8734, abs=0.005)

    # Multi-seed P4 sensor dispersion across seeds 1234, 2345, 3456
    disp = res["p4_seed_dispersion"]
    assert disp["u_R"]["livebench"]["mean_auroc"] == pytest.approx(0.7862, abs=0.005)
    assert disp["u_R"]["livebench"]["rejected_alpha"] is True
    assert disp["u_C"]["livebench"]["mean_auroc"] == pytest.approx(0.6566, abs=0.005)
    assert disp["u_C"]["livebench"]["rejected_alpha"] is False
    assert disp["u_F"]["popqa"]["mean_auroc"] == pytest.approx(0.7956, abs=0.005)
    assert disp["u_F"]["popqa"]["rejected_alpha"] is True

    # LiveBench u_R truncation sensitivity check
    trunc = res["livebench_uR_truncation_check"]
    assert trunc["P2"]["auroc_no_truncation"] == pytest.approx(0.690, abs=0.005)
    assert trunc["P3"]["auroc_no_truncation"] == pytest.approx(0.633, abs=0.005)
    assert trunc["P4"]["auroc_no_truncation"] == pytest.approx(0.588, abs=0.005)

    # P2 IFEval top-20% u_R flagging
    flagging = res["p2_ifeval_uR_flagging"]
    assert flagging["n_defined"] == 148
    assert flagging["base_accuracy"] == pytest.approx(0.8649, abs=0.005)
    assert flagging["retained_80pct_accuracy"] == pytest.approx(0.9153, abs=0.005)
    assert flagging["errors_caught_in_top20pct"] == 10
    assert flagging["total_errors"] == 20


def test_finqa_agreement_modes(v11_df: pd.DataFrame):
    ag = compute_finqa_agreement(v11_df)

    df_false = ag["dropna_false"]
    assert df_false["p4_across_seeds"]["agree_n"] == 141
    assert df_false["p4_across_seeds"]["disagree_n"] == 9
    assert df_false["p4_across_seeds"]["agree_accuracy"] == pytest.approx(0.7849, abs=0.005)
    assert df_false["p4_across_seeds"]["disagree_accuracy"] == pytest.approx(1 / 3, abs=0.005)
    assert df_false["p1_p2_p3_seed1234"]["1"]["n_questions"] == 59
    assert df_false["p1_p2_p3_seed1234"]["2"]["n_questions"] == 66
    assert df_false["p1_p2_p3_seed1234"]["3"]["n_questions"] == 25
    assert df_false["p1_p2_p3_seed1234"]["1"]["accuracy"] == pytest.approx(0.8870, abs=0.005)

    df_true = ag["dropna_true"]
    assert df_true["p1_p2_p3_seed1234"]["1"]["n_questions"] == 67
    assert df_true["p1_p2_p3_seed1234"]["2"]["n_questions"] == 63
    assert df_true["p1_p2_p3_seed1234"]["3"]["n_questions"] == 20
    assert df_true["p1_p2_p3_seed1234"]["1"]["accuracy"] == pytest.approx(0.8159, abs=0.005)


def test_silent_failure_auroc(v11_df: pd.DataFrame):
    sf = compute_silent_failure_auroc(v11_df)
    assert (sf["finqa"]["p1_silent_failures"], sf["finqa"]["p1_total_errors"]) == (31, 63)
    assert (sf["ifeval"]["p1_silent_failures"], sf["ifeval"]["p1_total_errors"]) == (12, 17)
    assert (sf["livebench"]["p1_silent_failures"], sf["livebench"]["p1_total_errors"]) == (37, 47)
    assert (sf["popqa"]["p1_silent_failures"], sf["popqa"]["p1_total_errors"]) == (77, 120)

    assert sf["finqa"]["setup_a_auroc_flag_silent_items"] == pytest.approx(0.624, abs=0.01)
    assert sf["finqa"]["setup_b_auroc_detect_p4_errors_within_silent"] == pytest.approx(0.409, abs=0.01)
    assert sf["livebench"]["setup_b_auroc_detect_p4_errors_within_silent"] == pytest.approx(0.671, abs=0.01)


def test_finqa_alternative_parsers():
    parsers = compute_finqa_alternative_parsers(FINQA_SCORES_DIR)

    def _matches(rule_key: str) -> dict[str, int]:
        return {pat: parsers[rule_key][pat]["matches"] for pat in ("P1", "P2", "P3", "P4")}

    assert _matches("adjudicated_rule") == {"P1": 87, "P2": 93, "P3": 94, "P4": 114}
    assert _matches("last_token_all_unit_scales_signed") == {"P1": 68, "P2": 62, "P3": 81, "P4": 114}
    assert _matches("last_token_contextual_unit_signed") == {"P1": 66, "P2": 61, "P3": 79, "P4": 113}
    assert _matches("last_token_7scale_unsigned") == {"P1": 70, "P2": 66, "P3": 83, "P4": 117}
    assert _matches("any_token_all_unit_scales_signed") == {"P1": 98, "P2": 119, "P3": 115, "P4": 116}
    assert _matches("any_token_7scale_unsigned") == {"P1": 109, "P2": 128, "P3": 124, "P4": 119}


def test_p4_per_cell_appendix_summaries_and_helper_parser():
    cov_rows, comp_rows = build_appendix_tables()
    cell_cov = [r for r in cov_rows if not r["is_subtotal"]]
    cell_comp = [r for r in comp_rows if not r["is_subtotal"]]
    assert len(cell_cov) == 12
    assert len(cell_comp) == 12
    assert len(cov_rows) == 17  # 12 cells + 4 benchmark subtotals + 1 grand total
    assert len(comp_rows) == 17
    assert sum(r["valid"] for r in cell_cov) == 951
    assert comp_rows[-1]["count"] == 951
    assert comp_rows[-1]["ux_stats"]["count"] == 951

    assert parse_label("12.5%", False) == ("12.5", "%")
    assert parse_label("1250 million", "false") == ("1250", "million")
    assert parse_label("42", "true") == (None, None)
    assert parse_label("", False) == (None, None)


def test_reproducibility_artefacts_and_raw_telemetry_present():
    # Verify raw baseline A100 telemetry (24 jsonl files x 150 lines)
    seed_dirs = [
        ("report_run_a100", [1, 2, 3, 4]),
        ("report_run_a100_seed2345", [4]),
        ("report_run_a100_seed3456", [4]),
    ]
    jsonl_count = 0
    for s_dir, patterns in seed_dirs:
        base = REPO_ROOT / "data" / s_dir
        assert (base / "manifest.json").exists()
        for ds in ("finqa", "ifeval", "livebench", "popqa"):
            for pat in patterns:
                fp = base / f"{ds}_pattern{pat}.jsonl"
                assert fp.exists(), f"Missing baseline telemetry file: {fp}"
                with fp.open(encoding="utf-8") as fh:
                    n_lines = sum(1 for line in fh if line.strip())
                assert n_lines == 150
                jsonl_count += 1
    assert jsonl_count == 24

    # Verify IFEval determinism audit CSVs and generated JSON audit files
    comp_csv = REPO_ROOT / "data" / "audits" / "ifeval_comparison_report.csv"
    unseeded_csv = REPO_ROOT / "data" / "audits" / "ifeval_official_harness_per_item_unseeded.csv"
    auroc_json = REPO_ROOT / "data" / "audits" / "auroc_results_v11.json"
    stats_json = REPO_ROOT / "data" / "audits" / "statistical_results.json"

    for path in (comp_csv, unseeded_csv, auroc_json, stats_json):
        assert path.exists(), f"Missing audit artefact: {path}"

    with unseeded_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # 900 canonical evaluations (6 cells x 150) + 450 P4 full_response sensitivity checks = 1350 rows
    assert len(rows) == 1350

    with auroc_json.open(encoding="utf-8") as fh:
        auroc_data = json.load(fh)
    assert set(auroc_data["sensor_auroc"].keys()) == {
        "pooled_sensor_auroc",
        "livebench_uR_truncation_check",
        "p2_ifeval_uR_flagging",
        "fdr_32_active_tests",
        "p4_seed_dispersion",
    }

    with stats_json.open(encoding="utf-8") as fh:
        stats = json.load(fh)
    assert "holm_stepdown_seed1234" in stats
    assert set(stats["holm_stepdown_seed1234"].keys()) == {"finqa", "livebench", "popqa", "ifeval"}
