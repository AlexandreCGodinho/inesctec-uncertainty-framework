"""Tests for A4: statistical significance & variance."""

import numpy as np
import pandas as pd
import pytest

from scripts.analysis_statistics import (
    _ALL_PATTERN_PAIRS,
    _mcnemar_pvalue,
    _paired_bootstrap_ci,
    _paired_wilcoxon_test,
    _compute_p4_seed_stats,
    run_mcnemar_analysis,
    run_bootstrap_analysis,
    run_livebench_wilcoxon_analysis,
    run_livebench_mcnemar_secondary,
    run_holm_stepdown_seed1234,
    run_fdr,
    _collect_pvals,
)


class TestMcnemarPvalue:
    def test_all_concordant_returns_one(self):
        p = _mcnemar_pvalue(0, 0)
        assert p == 1.0

    def test_large_discordant_asymptotic(self):
        b, c = 30, 10
        p = _mcnemar_pvalue(b, c)
        assert 0 < p < 1

    def test_small_discordant_exact(self):
        b, c = 3, 0
        p = _mcnemar_pvalue(b, c)
        # binomtest(3, 3, 0.5).pvalue = 0.25 (two-sided: P(X=0)+P(X=3) = 0.125+0.125)
        assert p == pytest.approx(0.25, abs=1e-10)

    def test_symmetric_discordant(self):
        b, c = 10, 10
        p = _mcnemar_pvalue(b, c)
        assert p == pytest.approx(1.0, abs=1e-6)

    def test_seed_determinism(self):
        p1 = _mcnemar_pvalue(12, 3)
        p2 = _mcnemar_pvalue(12, 3)
        assert p1 == p2


class TestPairedBootstrapCi:
    def test_no_difference(self):
        diffs = np.zeros(50)
        res = _paired_bootstrap_ci(diffs, n_resamples=1000)
        assert res["observed_mean_diff"] == 0.0
        assert res["ci_95"][0] <= 0 <= res["ci_95"][1]
        assert res["p_value"] >= 0.05

    def test_positive_mean(self):
        np.random.seed(42)
        diffs = np.random.normal(1.0, 0.5, 100)
        res = _paired_bootstrap_ci(diffs, n_resamples=1000)
        assert res["observed_mean_diff"] > 0
        assert res["ci_95"][0] > 0
        assert res["p_value"] <= 0.05

    def test_deterministic(self):
        diffs = np.array([0.5, -0.2, 0.3, 0.1, -0.4])
        r1 = _paired_bootstrap_ci(diffs, n_resamples=500)
        r2 = _paired_bootstrap_ci(diffs, n_resamples=500)
        assert r1["observed_mean_diff"] == r2["observed_mean_diff"]
        assert r1["p_value"] == r2["p_value"]

    def test_single_element(self):
        diffs = np.array([1.0])
        res = _paired_bootstrap_ci(diffs, n_resamples=100)
        assert res["observed_mean_diff"] == 1.0


class TestComputeP4SeedStats:
    def test_three_identical_seeds(self):
        df = pd.DataFrame({
            "seed": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "headline_bin": [1, 1, 0, 1, 1, 0, 1, 1, 0],
        })
        res = _compute_p4_seed_stats(df)
        assert res["mean"] == pytest.approx(2 / 3, abs=1e-6)
        assert res["sd"] == 0.0
        assert res["per_seed"]["0"] == pytest.approx(2 / 3)

    def test_different_seeds(self):
        df = pd.DataFrame({
            "seed": [0, 0, 1, 1, 2, 2],
            "headline_bin": [1, 1, 1, 0, 0, 0],
        })
        res = _compute_p4_seed_stats(df)
        assert res["mean"] == pytest.approx(0.5, abs=1e-6)
        assert res["sd"] > 0

    def test_empty_dataframe(self):
        df = pd.DataFrame({"seed": [], "headline_bin": []})
        res = _compute_p4_seed_stats(df)
        assert res["mean"] is None


class TestRunMcnemarAnalysis:
    def test_popqa_p4_vs_p1(self):
        df = pd.DataFrame({
            "dataset": ["popqa"] * 6,
            "query_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "pattern": ["1", "4", "1", "4", "1", "4"],
            "seed": ["0", "0", "0", "0", "0", "0"],
            "headline_bin": ["1", "0", "1", "1", "0", "0"],
        })
        res = run_mcnemar_analysis(df)
        assert "popqa" in res
        assert "p4_vs_p1" in res["popqa"]
        assert "per_seed" in res["popqa"]["p4_vs_p1"]

    def test_p2_vs_p1(self):
        df = pd.DataFrame({
            "dataset": ["ifeval"] * 4,
            "query_id": ["q1", "q1", "q2", "q2"],
            "pattern": ["1", "2", "1", "2"],
            "seed": ["0", "0", "0", "0"],
            "headline_bin": ["1", "0", "0", "1"],
        })
        res = run_mcnemar_analysis(df)
        v = res["ifeval"]["p2_vs_p1"]
        assert v["b"] == 1  # P1=0, P2=1 (q2)
        assert v["c"] == 1  # P1=1, P2=0 (q1)

    def test_no_data_returns_zero_counts(self):
        df = pd.DataFrame(columns=["dataset", "query_id", "pattern", "seed", "headline_bin"])
        res = run_mcnemar_analysis(df)
        for dataset in ("popqa", "finqa", "ifeval"):
            assert dataset in res
            for pair_key in ("p2_vs_p1", "p3_vs_p1"):
                assert res[dataset][pair_key]["b"] == 0
                assert res[dataset][pair_key]["c"] == 0
                assert res[dataset][pair_key]["p_value"] == 1.0


class TestRunBootstrapAnalysis:
    def test_no_livebench_data(self):
        df = pd.DataFrame({"dataset": ["popqa"], "accuracy": [1.0], "pattern": ["1"]})
        res = run_bootstrap_analysis(df)
        assert res == {}

    def test_p2_vs_p1(self):
        df = pd.DataFrame({
            "dataset": ["livebench"] * 4,
            "query_id": ["q1", "q1", "q2", "q2"],
            "pattern": ["1", "2", "1", "2"],
            "seed": ["0", "0", "0", "0"],
            "accuracy": [0.8, 0.9, 0.6, 0.7],
        })
        res = run_bootstrap_analysis(df)
        assert "overall" in res
        v = res["overall"]["p2_vs_p1"]
        assert v["observed_mean_diff"] == pytest.approx(0.1, abs=0.01)


class TestCollectPvals:
    def test_both_sources(self):
        mcnemar = {
            "popqa": {
                "p2_vs_p1": {"p_value": 0.03, "n": 10},
                "p4_vs_p1": {"per_seed": [{"p_value": 0.04}, {"p_value": 0.06}], "p_value_median": 0.05},
            },
        }
        bootstrap = {
            "overall": {
                "p2_vs_p1": {"p_value": 0.01, "n": 10},
            },
        }
        pvals = _collect_pvals(mcnemar, bootstrap)
        assert len(pvals) == 3
        assert 0.03 in pvals
        assert 0.05 in pvals
        assert 0.01 in pvals

    def test_empty(self):
        assert _collect_pvals({}, {}) == []


class TestRunFdr:
    def test_all_rejected(self):
        res = run_fdr([0.001, 0.002, 0.003], alpha=0.05)
        assert res["n_rejected"] == 3

    def test_none_rejected(self):
        res = run_fdr([0.4, 0.5, 0.6], alpha=0.05)
        assert res["n_rejected"] == 0

    def test_empty(self):
        res = run_fdr([], alpha=0.05)
        assert res["n_tests"] == 0

    def test_known_bh_output(self):
        pvals = [0.01, 0.04, 0.10, 0.50]
        res = run_fdr(pvals, alpha=0.05)
        assert res["n_tests"] == 4
        assert len(res["corrected"]) == 4
        for c in res["corrected"]:
            assert 0 < c <= 1

    def test_deterministic(self):
        pvals = [0.02, 0.03, 0.20]
        r1 = run_fdr(pvals)
        r2 = run_fdr(pvals)
        assert r1["corrected"] == r2["corrected"]
        assert r1["rejected"] == r2["rejected"]


class TestLiveBenchWilcoxonAndSecondaryMcNemar:
    def test_paired_wilcoxon_zero_and_nonzero(self):
        res_zero = _paired_wilcoxon_test(np.array([1.0, 0.375]), np.array([1.0, 0.375]))
        assert res_zero["p_value"] == 1.0
        assert res_zero["n_nonzero"] == 0

        res_pos = _paired_wilcoxon_test(
            np.array([1.0, 1.0, 0.375, 1.0, 1.0, 1.0]),
            np.array([0.0, 0.25, 0.0, 0.0, 0.125, 0.0]),
        )
        assert res_pos["observed_mean_diff"] > 0
        assert res_pos["n_nonzero"] == 6
        assert 0.0 < res_pos["p_value"] < 0.05

    def test_livebench_wilcoxon_primary_all_pairs(self):
        df = pd.DataFrame({
            "dataset": ["livebench"] * 8,
            "query_id": ["q1", "q2", "q1", "q2", "q1", "q2", "q1", "q2"],
            "pattern": ["1", "1", "2", "2", "3", "3", "4", "4"],
            "seed": [1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234],
            "accuracy": [0.0, 0.25, 0.375, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        res = run_livebench_wilcoxon_analysis(df)
        assert "p2_vs_p1" in res
        assert "p3_vs_p2" in res
        assert "p4_vs_p3" in res
        assert res["p2_vs_p1"]["observed_mean_diff"] == pytest.approx((0.375 + 0.75) / 2)

    def test_livebench_mcnemar_secondary_binarised_at_acc1(self):
        # q1: P1 has partial credit (0.375 -> 0), P2 has full credit (1.0 -> 1) => discordant b=1
        # q2: P1 has 1.0 (-> 1), P2 has 1.0 (-> 1) => concordant
        df = pd.DataFrame({
            "dataset": ["livebench"] * 4,
            "query_id": ["q1", "q2", "q1", "q2"],
            "pattern": ["1", "1", "2", "2"],
            "seed": [1234, 1234, 1234, 1234],
            "accuracy": [0.375, 1.0, 1.0, 1.0],
        })
        res = run_livebench_mcnemar_secondary(df, pairs=[(2, 1)])
        assert res["p2_vs_p1"]["b"] == 1
        assert res["p2_vs_p1"]["c"] == 0
        assert res["p2_vs_p1"]["n"] == 2


class TestHolmStepdownSeed1234:
    def test_monotonicity_and_rejection_synthetic(self):
        mcnemar = {
            "finqa": {
                "p2_vs_p1": {"p_value": 0.40},
                "p3_vs_p1": {"p_value": 0.20},
                "p4_vs_p1": {"per_seed": [{"seed": 1234, "p_value": 0.001}], "p_value_median": 0.001},
                "p3_vs_p2": {"p_value": 1.00},
                "p4_vs_p2": {"per_seed": [{"seed": 1234, "p_value": 0.005}], "p_value_median": 0.005},
                "p4_vs_p3": {"per_seed": [{"seed": 1234, "p_value": 0.008}], "p_value_median": 0.008},
            },
        }
        holm = run_holm_stepdown_seed1234(mcnemar, {}, alpha=0.05)
        fq = holm["finqa"]
        assert fq["p4_vs_p1"]["p_holm"] == pytest.approx(0.006)
        assert fq["p4_vs_p2"]["p_holm"] == pytest.approx(0.025)
        assert fq["p4_vs_p3"]["p_holm"] == pytest.approx(0.032)
        assert fq["p4_vs_p1"]["rejected"] is True
        assert fq["p4_vs_p2"]["rejected"] is True
        assert fq["p4_vs_p3"]["rejected"] is True
        assert fq["p2_vs_p1"]["rejected"] is False

    def test_reproduces_all_pairs_holm_seed1234_on_v11_csv(self):
        from pathlib import Path

        csv_path = Path(__file__).resolve().parent.parent / "data" / "audits" / "per_item_scores_v11.csv"
        df = pd.read_csv(csv_path, dtype=str)
        mcnemar = run_mcnemar_analysis(df, pairs=_ALL_PATTERN_PAIRS)
        lb_wilcoxon = run_livebench_wilcoxon_analysis(df, pairs=_ALL_PATTERN_PAIRS)
        holm = run_holm_stepdown_seed1234(mcnemar, lb_wilcoxon, alpha=0.05)

        # FinQA: P4 > {P1, P2, P3} significant; P1/P2/P3 indistinguishable
        assert holm["finqa"]["p4_vs_p1"]["rejected"] is True
        assert holm["finqa"]["p4_vs_p2"]["rejected"] is True
        assert holm["finqa"]["p4_vs_p3"]["rejected"] is True
        assert holm["finqa"]["p2_vs_p1"]["rejected"] is False
        assert holm["finqa"]["p3_vs_p1"]["rejected"] is False
        assert holm["finqa"]["p3_vs_p2"]["rejected"] is False

        # LiveBench: P3 > P1, P4 > P1, and P3 > P2 survive Holm step-down; P4 vs P2 (p_holm=0.059) does not
        assert holm["livebench"]["p4_vs_p1"]["rejected"] is True
        assert holm["livebench"]["p4_vs_p1"]["p_holm"] == pytest.approx(0.009754, abs=1e-4)
        assert holm["livebench"]["p3_vs_p1"]["rejected"] is True
        assert holm["livebench"]["p3_vs_p2"]["rejected"] is True
        assert holm["livebench"]["p2_vs_p1"]["rejected"] is False
        assert holm["livebench"]["p4_vs_p2"]["rejected"] is False
        assert holm["livebench"]["p4_vs_p3"]["rejected"] is False

        # PopQA: zero significant pairs
        assert all(not v["rejected"] for v in holm["popqa"].values())

        # IFEval: P4 < P1 and P4 < P3 survive Holm step-down
        assert holm["ifeval"]["p4_vs_p1"]["rejected"] is True
        assert holm["ifeval"]["p4_vs_p3"]["rejected"] is True
        assert holm["ifeval"]["p4_vs_p2"]["rejected"] is False

