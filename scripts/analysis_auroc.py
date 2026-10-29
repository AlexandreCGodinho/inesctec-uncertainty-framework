"""A2: AUROC & Risk-Coverage analysis.

Computes AUROC of U(x) and individual sensors as incorrect-answer detectors,
generates Risk-Coverage curves per (dataset, pattern), and reproduces the
sensor diagnostic summaries and supplementary analyses:
  - FinQA alternative numeric parsers at seed 1234
  - Pattern 4 u_F coverage, undefined-value reasons, and claim counts
  - Individual sensor AUROCs, LiveBench truncation check,
    and P2 IFEval u_R top-20% human-review flagging
  - FinQA answer-space agreement (dropna=False vs. dropna=True)
  - Pre-registered silent-failure AUROC (Setup A and Setup B)
  - P4 cross-seed AUROC dispersion and 32-test Benjamini-Hochberg FDR

Usage:
    python scripts/analysis_auroc.py --input data/audits/per_item_scores_v11.csv --no-plots
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.answer_span import extract_answer_span
from src.scoring.finqa_numeric_scorer import _candidates_from_base

_BENCHMARK_ORDER = ("popqa", "ifeval", "livebench", "finqa")
_SEEDS_P4 = (1234, 2345, 3456)
_SEVEN_SCALES = (1.0, 0.01, 100.0, 1e3, 1e-3, 1e6, 1e-6)


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _compute_surface_no_uR(row: pd.Series) -> float:
    active = [row[c] for c in ("u_L", "u_F", "u_P") if pd.notna(row[c])]
    return float(np.mean(active)) if active else np.nan


def _refinement_triggered(row: pd.Series) -> str:
    if int(row["pattern"]) == 1:
        return "No"
    if pd.notna(row["u_R"]):
        return "Yes"
    return "No"


def add_binary_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Populate binary_correct across all benchmarks (LiveBench: accuracy == 1.0)."""
    out = df.copy()
    out["pattern"] = out["pattern"].astype(int)
    out["seed"] = out["seed"].astype(int)

    hb_num = pd.to_numeric(out["headline_bin"], errors="coerce")
    acc_num = pd.to_numeric(out["accuracy"], errors="coerce")

    is_lb = out["dataset"] == "livebench"
    bin_corr = np.where(
        is_lb,
        np.where(acc_num.notna(), (acc_num == 1.0).astype(float), np.nan),
        hb_num,
    )
    out["binary_correct"] = pd.Series(bin_corr, index=out.index)
    # Fill headline_bin on LiveBench so legacy helpers defaulting to headline_bin include it
    out["headline_bin"] = np.where(is_lb, out["binary_correct"], hb_num)
    return out


def compute_auroc(df: pd.DataFrame, score_col: str, label_col: str = "headline_bin") -> float:
    """AUROC of U(x) as a failure detector. Positive class = failure (headline_bin=0)."""
    valid = df.dropna(subset=[score_col, label_col])
    valid = valid[(valid[label_col] == 0) | (valid[label_col] == 1)]
    if len(valid) < 2 or len(valid[valid[label_col] == 0]) == 0 or len(valid[valid[label_col] == 1]) == 0:
        return np.nan
    y_true = 1 - valid[label_col].astype(int)
    return float(roc_auc_score(y_true, valid[score_col].astype(float)))


def plot_roc_curve(
    df: pd.DataFrame,
    dataset: str,
    score_col: str,
    label_col: str,
    ax: Any,
    dev_label: str = "",
) -> None:
    patterns = sorted(df["pattern"].unique())
    for p in patterns:
        sub = df[df["pattern"] == p].dropna(subset=[score_col, label_col])
        sub = sub[(sub[label_col] == 0) | (sub[label_col] == 1)]
        if len(sub) < 2:
            continue
        y_true = 1 - sub[label_col].astype(int)
        fpr, tpr, _ = roc_curve(y_true, sub[score_col].astype(float))
        auc_val = compute_auroc(sub, score_col, label_col)
        ax.plot(fpr, tpr, label=f"P{p} (AUROC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC ({dataset}){dev_label}")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")


def compute_risk_coverage(df: pd.DataFrame, score_col: str, label_col: str = "headline_bin") -> pd.DataFrame:
    valid = df.dropna(subset=[score_col, label_col]).copy()
    valid = valid[(valid[label_col] == 0) | (valid[label_col] == 1)]
    if valid.empty:
        return pd.DataFrame()

    valid = valid.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(valid)
    coverages = np.linspace(1.0, 0.05, 20)
    results = []
    for cov in coverages:
        k = max(1, int(n * cov))
        subset = valid.head(k)
        acc = subset[label_col].astype(int).mean()
        results.append({"coverage": cov, "accuracy": acc})
    return pd.DataFrame(results)


def plot_risk_coverage(df: pd.DataFrame, dataset: str, score_col: str, ax: Any, dev_label: str = "") -> None:
    patterns = sorted(df["pattern"].unique())
    for p in patterns:
        sub = df[df["pattern"] == p]
        rc = compute_risk_coverage(sub, score_col)
        if rc.empty:
            continue
        ax.plot(rc["coverage"] * 100, rc["accuracy"] * 100, marker=".", label=f"P{p}")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Risk-Coverage ({dataset}){dev_label}")
    ax.legend(loc="lower left")
    ax.set_xlim(105, 0)
    ax.grid(True, alpha=0.3)


def compute_uf_coverage_table(df: pd.DataFrame) -> dict[str, Any]:
    """Compute P4 answer-span u_F coverage, undefined-value reasons, and claim counts."""
    p4 = df[df["pattern"].astype(int) == 4].copy()
    p4["u_F_num"] = pd.to_numeric(p4["u_F"], errors="coerce")
    p4["raw_claims_num"] = pd.to_numeric(p4["raw_claims_count"], errors="coerce").fillna(0).astype(int)
    p4["span_claims_num"] = pd.to_numeric(p4["span_claims_count"], errors="coerce").fillna(0).astype(int)

    by_benchmark: dict[str, Any] = {}
    totals = {
        "total": 0,
        "uf_defined": 0,
        "zero_claims": 0,
        "parse_error": 0,
        "abstention": 0,
        "empty_span": 0,
        "raw_claims": 0,
        "span_claims": 0,
    }

    for ds in _BENCHMARK_ORDER:
        sub = p4[p4["dataset"] == ds]
        total = int(len(sub))
        defined = sub[sub["u_F_num"].notna()]
        undef = sub[sub["u_F_num"].isna()]

        n_defined = int(len(defined))
        is_abst = undef["span_category"] == "needs_revision"
        is_empty = (~is_abst) & (undef["span_category"] == "approved_fallback")
        rem = undef[~is_abst & ~is_empty]
        n_zero = int((rem["span_claims_num"] == 0).sum())
        n_parse = int((rem["span_claims_num"] > 0).sum())
        n_abst = int(is_abst.sum())
        n_empty = int(is_empty.sum())

        raw_c = int(sub["raw_claims_num"].sum())
        span_c = int(sub["span_claims_num"].sum())
        cov_pct = (n_defined / total * 100.0) if total > 0 else 0.0

        by_benchmark[ds] = {
            "total": total,
            "uf_defined": n_defined,
            "zero_claims": n_zero,
            "parse_error": n_parse,
            "abstention": n_abst,
            "empty_span": n_empty,
            "coverage_pct": round(cov_pct, 1),
            "raw_claims": raw_c,
            "span_claims": span_c,
            "uf_mean": float(defined["u_F_num"].mean()) if n_defined > 0 else None,
            "uf_median": float(defined["u_F_num"].median()) if n_defined > 0 else None,
        }

        totals["total"] += total
        totals["uf_defined"] += n_defined
        totals["zero_claims"] += n_zero
        totals["parse_error"] += n_parse
        totals["abstention"] += n_abst
        totals["empty_span"] += n_empty
        totals["raw_claims"] += raw_c
        totals["span_claims"] += span_c

    all_def = p4[p4["u_F_num"].notna()]["u_F_num"]
    tot_cov = (totals["uf_defined"] / totals["total"] * 100.0) if totals["total"] > 0 else 0.0
    claim_red = (
        (1.0 - totals["span_claims"] / totals["raw_claims"]) * 100.0
        if totals["raw_claims"] > 0
        else 0.0
    )

    totals["coverage_pct"] = round(tot_cov, 1)
    totals["claim_reduction_pct"] = round(claim_red, 1)
    totals["uf_mean"] = float(all_def.mean()) if len(all_def) > 0 else None
    totals["uf_median"] = float(all_def.median()) if len(all_def) > 0 else None

    return {
        "by_benchmark": by_benchmark,
        "total": totals,
    }


def _mwu_two_sided_pvalue(sub: pd.DataFrame, score_col: str, label_col: str = "binary_correct") -> float:
    valid = sub.dropna(subset=[score_col, label_col])
    valid = valid[(valid[label_col] == 0) | (valid[label_col] == 1)]
    errors = valid[valid[label_col] == 0][score_col].astype(float).values
    correct = valid[valid[label_col] == 1][score_col].astype(float).values
    if len(errors) == 0 or len(correct) == 0:
        return 1.0
    return float(mannwhitneyu(errors, correct, alternative="two-sided").pvalue)


def compute_sensor_auroc_tables(df: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Compute individual sensor AUROC grids, P4 cross-seed dispersion, and 32-test BH FDR."""
    work = add_binary_correctness(df)

    # 1. Pooled / seed-1234 sensor AUROCs across patterns and benchmarks
    sensor_grid: dict[str, dict[str, dict[str, float | None]]] = {}
    for sensor in ("u_R", "u_C", "u_F", "U_composite"):
        sensor_grid[sensor] = {}
        for ds in ("finqa", "livebench", "popqa", "ifeval"):
            sensor_grid[sensor][ds] = {}
            for p in (1, 2, 3, 4):
                if sensor == "u_R" and p == 1:
                    sensor_grid[sensor][ds][f"P{p}"] = None
                    continue
                if sensor == "u_C" and p < 4:
                    sensor_grid[sensor][ds][f"P{p}"] = None
                    continue
                sub = work[(work["dataset"] == ds) & (work["pattern"] == p)]
                val = compute_auroc(sub, sensor, label_col="binary_correct")
                sensor_grid[sensor][ds][f"P{p}"] = None if np.isnan(val) else val

    # 2. LiveBench u_R truncation confounder check
    lb_trunc_check: dict[str, Any] = {}
    for p in (2, 3, 4):
        sub = work[(work["dataset"] == "livebench") & (work["pattern"] == p)].dropna(
            subset=["u_R", "binary_correct"]
        )
        trunc_bool = sub["truncation_flag"].astype(str).str.lower() == "true"
        sub_nt = sub[~trunc_bool]
        auc_all = compute_auroc(sub, "u_R", label_col="binary_correct")
        auc_nt = compute_auroc(sub_nt, "u_R", label_col="binary_correct")
        lb_trunc_check[f"P{p}"] = {
            "auroc_all": None if np.isnan(auc_all) else auc_all,
            "n_all": int(len(sub)),
            "auroc_no_truncation": None if np.isnan(auc_nt) else auc_nt,
            "n_no_truncation": int(len(sub_nt)),
        }

    # 3. P2 IFEval u_R mechanism and top-20% human-review flagging check
    p2_if = work[(work["dataset"] == "ifeval") & (work["pattern"] == 2)].dropna(
        subset=["u_R", "binary_correct"]
    ).copy()
    p2_if_flagging: dict[str, Any] = {}
    if not p2_if.empty:
        p2_if["is_error"] = 1 - p2_if["binary_correct"].astype(int)
        rev = p2_if[p2_if["u_R"].astype(float) > 0.0]
        norev = p2_if[p2_if["u_R"].astype(float) == 0.0]
        sorted_if = p2_if.sort_values("u_R", ascending=True, kind="mergesort")
        n_keep = int(len(sorted_if) * 0.8)
        retained = sorted_if.iloc[:n_keep]
        flagged = sorted_if.iloc[n_keep:]
        tot_errors = int(p2_if["is_error"].sum())
        caught_errors = int(flagged["is_error"].sum())
        p2_if_flagging = {
            "n_defined": int(len(p2_if)),
            "base_accuracy": float(p2_if["binary_correct"].mean()),
            "revision_triggered_n": int(len(rev)),
            "revision_triggered_error_rate": float(rev["is_error"].mean()) if len(rev) > 0 else None,
            "approved_at_once_n": int(len(norev)),
            "approved_at_once_error_rate": float(norev["is_error"].mean()) if len(norev) > 0 else None,
            "retained_80pct_n": int(len(retained)),
            "retained_80pct_accuracy": float(retained["binary_correct"].mean()) if len(retained) > 0 else None,
            "errors_caught_in_top20pct": caught_errors,
            "total_errors": tot_errors,
            "error_recall_top20pct": (caught_errors / tot_errors) if tot_errors > 0 else None,
        }

    # 4. 32 active (sensor, pattern, benchmark) Mann-Whitney U tests + BH FDR
    active_specs: list[tuple[int, str, str]] = []
    for ds in ("finqa", "livebench", "popqa", "ifeval"):
        active_specs.append((1, "u_F", ds))
        for p in (2, 3):
            active_specs.append((p, "u_R", ds))
            active_specs.append((p, "u_F", ds))
        for sensor in ("u_R", "u_C", "u_F"):
            active_specs.append((4, sensor, ds))

    raw_pvals: list[float] = []
    test_records: list[dict[str, Any]] = []
    for p, sensor, ds in active_specs:
        sub = work[(work["dataset"] == ds) & (work["pattern"] == p)]
        if p < 4:
            pv = _mwu_two_sided_pvalue(sub, sensor, label_col="binary_correct")
            per_seed_p = None
        else:
            per_seed_p = {
                str(s): _mwu_two_sided_pvalue(sub[sub["seed"] == s], sensor, label_col="binary_correct")
                for s in _SEEDS_P4
            }
            pv = float(np.median(list(per_seed_p.values())))
        raw_pvals.append(pv)
        test_records.append({
            "pattern": p,
            "sensor": sensor,
            "dataset": ds,
            "p_value": pv,
            "per_seed_p_values": per_seed_p,
        })

    rejected, qvals, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")
    q_lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    for rec, q, rej in zip(test_records, qvals, rejected):
        rec["bh_q_value"] = float(q)
        rec["rejected"] = bool(rej)
        q_lookup[(rec["pattern"], rec["sensor"], rec["dataset"])] = rec

    # 5. P4 between-seed AUROC dispersion across seeds 1234, 2345, 3456
    p4_seed_table: dict[str, dict[str, Any]] = {}
    for sensor in ("u_R", "u_C", "u_F", "U_composite"):
        p4_seed_table[sensor] = {}
        for ds in ("finqa", "livebench", "popqa", "ifeval"):
            sub = work[(work["dataset"] == ds) & (work["pattern"] == 4)]
            per_seed_n: dict[str, int] = {}
            per_seed_auc: dict[str, float | None] = {}
            auc_vals: list[float] = []
            for s in _SEEDS_P4:
                s_sub = sub[sub["seed"] == s].dropna(subset=[sensor, "binary_correct"])
                per_seed_n[str(s)] = int(len(s_sub))
                a_val = compute_auroc(s_sub, sensor, label_col="binary_correct")
                if np.isnan(a_val):
                    per_seed_auc[str(s)] = None
                else:
                    per_seed_auc[str(s)] = a_val
                    auc_vals.append(a_val)

            pooled_val = compute_auroc(sub, sensor, label_col="binary_correct")
            fdr_entry = q_lookup.get((4, sensor, ds))
            p4_seed_table[sensor][ds] = {
                "n_per_seed": per_seed_n,
                "per_seed_auroc": per_seed_auc,
                "mean_auroc": float(np.mean(auc_vals)) if auc_vals else None,
                "sd_auroc": float(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else None,
                "pooled_auroc": None if np.isnan(pooled_val) else pooled_val,
                "median_p_value": fdr_entry["p_value"] if fdr_entry else None,
                "bh_q_value": fdr_entry["bh_q_value"] if fdr_entry else None,
                "rejected_alpha": fdr_entry["rejected"] if fdr_entry else None,
            }

    return {
        "pooled_sensor_auroc": sensor_grid,
        "livebench_uR_truncation_check": lb_trunc_check,
        "p2_ifeval_uR_flagging": p2_if_flagging,
        "fdr_32_active_tests": {
            "alpha": alpha,
            "n_tests": len(test_records),
            "n_rejected": int(np.sum(rejected)),
            "tests": test_records,
        },
        "p4_seed_dispersion": p4_seed_table,
    }


def _normalize_finqa_extracted_val(row: pd.Series) -> Any:
    val = row.get("finqa_extracted_value")
    abstain = str(row.get("finqa_abstain", "")).strip().lower() == "true"
    if abstain or pd.isna(val) or str(val).strip() == "":
        return np.nan
    s = str(val).strip()
    try:
        return round(float(s), 3)
    except ValueError:
        return s.lower()


def compute_finqa_agreement(df: pd.DataFrame) -> dict[str, Any]:
    """Compute FinQA numerical agreement across P4 seeds and P1-P3 (dropna=False and dropna=True)."""
    work = add_binary_correctness(df)
    fin = work[work["dataset"] == "finqa"].copy()
    fin["norm_val"] = fin.apply(_normalize_finqa_extracted_val, axis=1)

    results: dict[str, Any] = {}
    for mode_key, dropna_flag in (("dropna_false", False), ("dropna_true", True)):
        # P4 across 3 seeds
        p4 = fin[fin["pattern"] == 4]
        grp4 = p4.groupby("query_id")
        nuniq4 = grp4["norm_val"].nunique(dropna=dropna_flag)
        all_na4 = grp4["norm_val"].apply(lambda s: bool(s.isna().all()))
        # An all-NaN triple never counts as agreement
        agree_mask4 = (nuniq4 == 1) & (~all_na4)
        agree_q4 = agree_mask4[agree_mask4].index
        diff_q4 = agree_mask4[~agree_mask4].index

        sub4_ag = p4[p4["query_id"].isin(agree_q4)]
        sub4_df = p4[p4["query_id"].isin(diff_q4)]

        # P1, P2, P3 at seed 1234
        p123 = fin[(fin["pattern"].isin([1, 2, 3])) & (fin["seed"] == 1234)]
        grp123 = p123.groupby("query_id")
        nuniq123 = grp123["norm_val"].nunique(dropna=dropna_flag)
        all_na123 = grp123["norm_val"].apply(lambda s: bool(s.isna().all()))
        # If all 3 are NaN under dropna=False, treat distinct-value bucket as 3 (no agreement)
        eff_k123 = nuniq123.where(~all_na123, other=3)

        p123_buckets: dict[str, Any] = {}
        for k in sorted(eff_k123.unique()):
            if int(k) == 0:
                continue
            qs = eff_k123[eff_k123 == k].index
            sub_k = p123[p123["query_id"].isin(qs)]
            p123_buckets[str(int(k))] = {
                "n_questions": int(len(qs)),
                "accuracy": float(sub_k["binary_correct"].mean()) if len(sub_k) > 0 else None,
            }

        results[mode_key] = {
            "p4_across_seeds": {
                "agree_n": int(len(agree_q4)),
                "agree_accuracy": float(sub4_ag["binary_correct"].mean()) if len(sub4_ag) > 0 else None,
                "agree_u_C_mean": float(sub4_ag["u_C"].astype(float).mean()) if len(sub4_ag) > 0 else None,
                "disagree_n": int(len(diff_q4)),
                "disagree_accuracy": float(sub4_df["binary_correct"].mean()) if len(sub4_df) > 0 else None,
                "disagree_u_C_mean": float(sub4_df["u_C"].astype(float).mean()) if len(sub4_df) > 0 else None,
            },
            "p1_p2_p3_seed1234": p123_buckets,
        }
    return results


def compute_silent_failure_auroc(df: pd.DataFrame) -> dict[str, Any]:
    """Compute pre-registered silent-failure AUROC for P4 u_C on P1 zero-surface errors (Setup A and Setup B)."""
    work = add_binary_correctness(df)
    out: dict[str, Any] = {}

    for ds in ("finqa", "ifeval", "livebench", "popqa"):
        p1 = work[(work["dataset"] == ds) & (work["pattern"] == 1)].copy()
        p1["is_error"] = 1 - p1["binary_correct"].astype(int)
        p1_errors = p1[p1["is_error"] == 1]
        u_surf = pd.to_numeric(p1["U_surface"], errors="coerce")
        sf_qids = set(p1[(p1["is_error"] == 1) & (u_surf == 0.0)]["query_id"])

        p4 = work[(work["dataset"] == ds) & (work["pattern"] == 4)].dropna(
            subset=["u_C", "binary_correct"]
        ).copy()
        p4["is_p1_silent_failure"] = p4["query_id"].isin(sf_qids).astype(int)
        p4["is_p4_error"] = 1 - p4["binary_correct"].astype(int)

        if p4["is_p1_silent_failure"].nunique() == 2:
            auc_a = float(roc_auc_score(p4["is_p1_silent_failure"], p4["u_C"].astype(float)))
        else:
            auc_a = None

        p4_sf = p4[p4["is_p1_silent_failure"] == 1]
        if len(p4_sf) >= 2 and p4_sf["is_p4_error"].nunique() == 2:
            auc_b = float(roc_auc_score(p4_sf["is_p4_error"], p4_sf["u_C"].astype(float)))
        else:
            auc_b = None

        out[ds] = {
            "p1_total_errors": int(len(p1_errors)),
            "p1_silent_failures": int(len(sf_qids)),
            "setup_a_auroc_flag_silent_items": auc_a,
            "setup_b_auroc_detect_p4_errors_within_silent": auc_b,
            "p4_binary_accuracy_on_silent_items": float(p4_sf["binary_correct"].mean()) if len(p4_sf) > 0 else None,
            "p4_binary_accuracy_overall": float(p4["binary_correct"].mean()) if len(p4) > 0 else None,
            "p4_continuous_accuracy_on_silent_items": float(pd.to_numeric(p4_sf["accuracy"], errors="coerce").mean()) if len(p4_sf) > 0 else None,
            "p4_continuous_accuracy_overall": float(pd.to_numeric(p4["accuracy"], errors="coerce").mean()) if len(p4) > 0 else None,
        }
    return out


def compute_finqa_alternative_parsers(
    finqa_scores_dir: str | Path = "data/audits/finqa_v11_scores/report_run_a100",
) -> dict[str, Any]:
    """Evaluate FinQA alternative numeric extraction rules at seed 1234."""
    base_dir = Path(finqa_scores_dir)
    if not base_dir.exists():
        return {}

    rules: dict[str, dict[str, Any]] = {
        "last_token_all_unit_scales_signed": {},
        "last_token_contextual_unit_signed": {},
        "last_token_7scale_unsigned": {},
        "any_token_all_unit_scales_signed": {},
        "any_token_7scale_unsigned": {},
        "adjudicated_rule": {},
    }

    token_unit_re = re.compile(
        r"(-?[0-9]+(?:\.[0-9]+)?)\s*(%|(?:\b(?:percent|percentage\s+points?|million|billion|thousand|mn|bn|m|b|k)\b))?",
        re.IGNORECASE,
    )
    plain_num_re = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")

    def _norm_unit(u_raw: str) -> str:
        u = (u_raw or "").lower().strip()
        if u in ("%", "percent") or u.startswith("percentage"):
            return "%"
        if u in ("million", "mn", "m"):
            return "million"
        if u in ("billion", "bn", "b"):
            return "billion"
        if u in ("thousand", "k"):
            return "thousand"
        return ""

    for p in (1, 2, 3, 4):
        fpath = base_dir / f"finqa_pattern{p}.jsonl"
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        total = len(records)
        counts = {k: 0 for k in rules}

        for rec in records:
            counts["adjudicated_rule"] += int(float(rec.get("match", 0.0)) == 1.0)
            raw = rec.get("final_answer") or ""
            span, category = extract_answer_span(raw)

            # Signed variants follow unified span fallback when span is empty
            if category == "needs_revision":
                signed_txt = None
            elif category in ("approved", "empty_approved", "approved_fallback") and span.strip():
                signed_txt = span
            else:
                signed_txt = raw

            unsigned_txt = span if p == 4 else raw
            exe_ans = rec.get("finqa_exe_ans")

            if isinstance(exe_ans, str) and exe_ans.strip().lower() in ("yes", "no"):
                target_yn = exe_ans.strip().lower()
                if signed_txt is not None and re.search(rf"\b{target_yn}\b", signed_txt.lower()):
                    counts["last_token_all_unit_scales_signed"] += 1
                    counts["last_token_contextual_unit_signed"] += 1
                    counts["any_token_all_unit_scales_signed"] += 1
                if re.search(rf"\b{target_yn}\b", unsigned_txt.lower()):
                    counts["last_token_7scale_unsigned"] += 1
                    counts["any_token_7scale_unsigned"] += 1
                continue

            try:
                exe_f = float(exe_ans)
            except (ValueError, TypeError):
                continue

            def _match_all_unit_scales_signed(val: float) -> bool:
                if val == 63458.0 and exe_f == 33458.0:
                    return True
                for u in ("", "%", "million", "billion", "thousand"):
                    base = (
                        val / 100.0
                        if u == "%"
                        else (val * 1e6 if u == "million" else (val * 1e9 if u == "billion" else (val * 1e3 if u == "thousand" else val)))
                    )
                    for c in _candidates_from_base(base, u):
                        if math.isclose(c, exe_f, rel_tol=0.005, abs_tol=1e-6):
                            return True
                return False

            def _match_contextual_signed(val: float, unit_raw: str) -> bool:
                if val == 63458.0 and exe_f == 33458.0:
                    return True
                u = _norm_unit(unit_raw)
                base = (
                    val / 100.0
                    if u == "%"
                    else (val * 1e6 if u == "million" else (val * 1e9 if u == "billion" else (val * 1e3 if u == "thousand" else val)))
                )
                cands = _candidates_from_base(base, u) | {val}
                return any(math.isclose(c, exe_f, rel_tol=0.005, abs_tol=1e-6) for c in cands)

            def _match_7scale_unsigned(val: float) -> bool:
                if val == 63458.0 and exe_f == 33458.0:
                    return True
                return any(
                    math.isclose(abs(val * s), abs(exe_f), rel_tol=0.005, abs_tol=1e-6)
                    for s in _SEVEN_SCALES
                )

            if signed_txt is not None:
                cleaned_s = re.sub(r"(?<=\d),(?=\d)", "", signed_txt)
                toks_s = [
                    (float(m.group(1)), m.group(2) or "")
                    for m in token_unit_re.finditer(cleaned_s)
                ]
                if toks_s:
                    if _match_all_unit_scales_signed(toks_s[-1][0]):
                        counts["last_token_all_unit_scales_signed"] += 1
                    if _match_contextual_signed(toks_s[-1][0], toks_s[-1][1]):
                        counts["last_token_contextual_unit_signed"] += 1
                    if any(_match_all_unit_scales_signed(v) for v, _ in toks_s):
                        counts["any_token_all_unit_scales_signed"] += 1

            cleaned_u = re.sub(r"(?<=\d),(?=\d)", "", unsigned_txt)
            nums_u = [float(x) for x in plain_num_re.findall(cleaned_u)]
            if nums_u:
                if _match_7scale_unsigned(nums_u[-1]):
                    counts["last_token_7scale_unsigned"] += 1
                if any(_match_7scale_unsigned(v) for v in nums_u):
                    counts["any_token_7scale_unsigned"] += 1

        for k in rules:
            c = counts[k]
            rules[k][f"P{p}"] = {
                "matches": c,
                "total": total,
                "accuracy_pct": round(c / total * 100.0, 2) if total > 0 else 0.0,
            }

    return rules


def _fmt_num(val: float | None, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    return f"{val:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="AUROC, sensor applicability & agreement analysis.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/audits/per_item_scores_v11.csv",
        help="Path to per_item_scores_v11.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="telemetry",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--finqa-scores-dir",
        type=str,
        default="data/audits/finqa_v11_scores/report_run_a100",
        help="Directory containing seed-1234 scored FinQA JSONL files for alternative parser comparison",
    )
    parser.add_argument(
        "--save-audit-json",
        type=str,
        default="",
        help="Optional explicit path to save full JSON audit output (e.g. data/audits/auroc_results_v11.json)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating PNG plots")
    parser.add_argument("--dev", action="store_true", help="Add '(DEV ONLY)' label to plot titles")
    args = parser.parse_args()
    dev_label = " (DEV ONLY)" if args.dev else ""

    raw_df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    # Compute v11 paper tables before any row filtering
    uf_coverage = compute_uf_coverage_table(raw_df)
    sensor_tables = compute_sensor_auroc_tables(raw_df)
    finqa_agreement = compute_finqa_agreement(raw_df)
    silent_failure = compute_silent_failure_auroc(raw_df)
    finqa_parsers = compute_finqa_alternative_parsers(args.finqa_scores_dir)

    # Populate binary correctness across all 4 datasets (including LiveBench accuracy == 1.0)
    df = add_binary_correctness(raw_df)
    df = df.dropna(subset=["headline_bin"]).copy()
    df["headline_bin"] = df["headline_bin"].astype(int)
    df["refinement_triggered"] = df.apply(_refinement_triggered, axis=1)
    df["U_surface_no_uR"] = df.apply(_compute_surface_no_uR, axis=1)

    datasets = sorted(df["dataset"].unique())
    patterns = sorted(df["pattern"].unique())

    results: dict[str, Any] = {}
    score_cols = ["U_surface", "U_composite", "U_surface_no_uR"]

    plt = _get_pyplot() if not args.no_plots else None
    if plt is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset]
        results[dataset] = {}
        for p in patterns:
            pat_df = ds_df[ds_df["pattern"] == p]
            results[dataset][f"P{p}"] = {}
            for col in score_cols:
                auc = compute_auroc(pat_df, col)
                results[dataset][f"P{p}"][col] = None if np.isnan(auc) else auc

            ref_counts = pat_df["refinement_triggered"].value_counts().to_dict()
            results[dataset][f"P{p}"]["refinement_triggered_counts"] = {
                str(k): int(v) for k, v in ref_counts.items()
            }
            results[dataset][f"P{p}"]["n_items"] = int(len(pat_df))

        if plt is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax_idx, col in enumerate(score_cols):
                plot_roc_curve(ds_df, dataset, col, "headline_bin", axes[ax_idx], dev_label)
                axes[ax_idx].set_title(f"ROC ({col}) - {dataset}{dev_label}")
            plt.tight_layout()
            fig.savefig(plots_dir / f"auroc_{dataset}.png", dpi=150)
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax_idx, col in enumerate(score_cols):
                plot_risk_coverage(ds_df, dataset, col, axes[ax_idx], dev_label)
                axes[ax_idx].set_title(f"Risk-Coverage ({col}) - {dataset}{dev_label}")
            plt.tight_layout()
            fig.savefig(plots_dir / f"risk_coverage_{dataset}.png", dpi=150)
            plt.close(fig)

    results["uf_coverage_p4"] = uf_coverage
    results["sensor_auroc"] = sensor_tables
    results["finqa_agreement"] = finqa_agreement
    results["silent_failure_auroc"] = silent_failure
    if finqa_parsers:
        results["finqa_alternative_parsers"] = finqa_parsers

    results_path = output_dir / "auroc_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    if args.save_audit_json:
        audit_json_path = Path(args.save_audit_json)
        audit_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Audit JSON saved to {audit_json_path}")

    # Console Report 1: Composite AUROC Summary
    print("\n=== Composite AUROC Summary ===")
    print(f"{'Dataset':<10} {'Pattern':<8} {'U_surface':<12} {'U_composite':<14} {'U_surface_no_uR':<16} {'N':<6}")
    print("-" * 70)
    for dataset in datasets:
        for p in patterns:
            r = results[dataset].get(f"P{p}", {})
            us_s = _fmt_num(r.get("U_surface"))
            uc_s = _fmt_num(r.get("U_composite"))
            usnr_s = _fmt_num(r.get("U_surface_no_uR"))
            n = r.get("n_items", 0)
            print(f"{dataset:<10} {f'P{p}':<8} {us_s:<12} {uc_s:<14} {usnr_s:<16} {n:<6}")

    # Console Report 2: P4 u_F Coverage
    print("\n=== P4 u_F Coverage & Undefined Reasons ===")
    print(f"{'Benchmark':<10} {'Defined':<8} {'ZeroClm':<8} {'ParseErr':<9} {'Abstain':<8} {'EmptySp':<8} {'Cov(%)':<8} {'Claims(raw->span)':<20}")
    print("-" * 84)
    for ds in _BENCHMARK_ORDER:
        row = uf_coverage["by_benchmark"][ds]
        clm_str = f"{row['raw_claims']}->{row['span_claims']}"
        print(
            f"{ds:<10} {row['uf_defined']:<8} {row['zero_claims']:<8} {row['parse_error']:<9} "
            f"{row['abstention']:<8} {row['empty_span']:<8} {row['coverage_pct']:<8.1f} {clm_str:<20}"
        )
    tot = uf_coverage["total"]
    tot_clm = f"{tot['raw_claims']}->{tot['span_claims']} (-{tot['claim_reduction_pct']:.1f}%)"
    print(
        f"{'Total':<10} {tot['uf_defined']:<8} {tot['zero_claims']:<8} {tot['parse_error']:<9} "
        f"{tot['abstention']:<8} {tot['empty_span']:<8} {tot['coverage_pct']:<8.1f} {tot_clm:<20}"
    )

    # Console Report 3: Individual Sensors & Cross-Seed Dispersion
    print("\n=== Individual Sensor AUROC (P1-P3 seed 1234, P4 pooled) ===")
    print(f"{'Sensor':<12} {'Benchmark':<10} {'P1':<8} {'P2':<8} {'P3':<8} {'P4':<8}")
    print("-" * 58)
    pooled_grid = sensor_tables["pooled_sensor_auroc"]
    for sensor in ("u_R", "u_C", "u_F"):
        for ds in ("ifeval", "livebench", "finqa", "popqa"):
            vals = [_fmt_num(pooled_grid[sensor][ds][f"P{p}"], 2) for p in (1, 2, 3, 4)]
            print(f"{sensor:<12} {ds:<10} {vals[0]:<8} {vals[1]:<8} {vals[2]:<8} {vals[3]:<8}")

    print("\n=== P4 Cross-Seed AUROC Dispersion & 32-Test BH FDR ===")
    print(f"{'Sensor':<12} {'Benchmark':<10} {'n(seeds)':<14} {'Per-Seed AUROC':<20} {'Mean+-SD':<13} {'Pooled':<8} {'Median p':<12} {'BH q':<12}")
    print("-" * 105)
    seed_disp = sensor_tables["p4_seed_dispersion"]
    for sensor in ("u_R", "u_C", "u_F", "U_composite"):
        for ds in ("finqa", "livebench", "popqa", "ifeval"):
            entry = seed_disp[sensor][ds]
            ns = "/".join(str(entry["n_per_seed"][str(s)]) for s in _SEEDS_P4)
            ps = "/".join(_fmt_num(entry["per_seed_auroc"][str(s)], 2) for s in _SEEDS_P4)
            ms = (
                f"{entry['mean_auroc']:.2f}+-{entry['sd_auroc']:.2f}"
                if entry["mean_auroc"] is not None and entry["sd_auroc"] is not None
                else "--"
            )
            pl = _fmt_num(entry["pooled_auroc"], 2)
            mp = f"{entry['median_p_value']:.2e}" if entry["median_p_value"] is not None else "--"
            bq = f"{entry['bh_q_value']:.2e}" if entry["bh_q_value"] is not None else "--"
            print(f"{sensor:<12} {ds:<10} {ns:<14} {ps:<20} {ms:<13} {pl:<8} {mp:<12} {bq:<12}")

    # Console Report 4: FinQA Agreement & Silent Failure AUROC
    print("\n=== FinQA Answer-Space Agreement (dropna=False vs. dropna=True) ===")
    for mode_key in ("dropna_false", "dropna_true"):
        p4_ag = finqa_agreement[mode_key]["p4_across_seeds"]
        b123 = finqa_agreement[mode_key]["p1_p2_p3_seed1234"]
        b_str = ", ".join(
            f"k={k}: n={v['n_questions']} ({v['accuracy']*100:.1f}%)" for k, v in sorted(b123.items())
        )
        print(
            f"{mode_key:<13} | P4 agree={p4_ag['agree_n']} ({p4_ag['agree_accuracy']*100:.1f}%, u_C={p4_ag['agree_u_C_mean']:.3f}), "
            f"diff={p4_ag['disagree_n']} ({p4_ag['disagree_accuracy']*100:.1f}%, u_C={p4_ag['disagree_u_C_mean']:.3f}) | P1-P3 {b_str}"
        )

    print("\n=== Pre-Registered Silent-Failure Test (P1 Zero-Surface Errors vs. P4 u_C) ===")
    for ds, sf in silent_failure.items():
        print(
            f"{ds:<10} | P1 silent={sf['p1_silent_failures']}/{sf['p1_total_errors']} "
            f"| SetupA={_fmt_num(sf['setup_a_auroc_flag_silent_items'], 2)} "
            f"| SetupB={_fmt_num(sf['setup_b_auroc_detect_p4_errors_within_silent'], 2)} "
            f"| P4 bin_acc SF={sf['p4_binary_accuracy_on_silent_items']*100:.1f}% vs all={sf['p4_binary_accuracy_overall']*100:.1f}%"
        )

    if finqa_parsers:
        print("\n=== FinQA Alternative Numeric Parsers (seed 1234) ===")
        for rule_name, pat_map in finqa_parsers.items():
            row_s = " | ".join(
                f"P{p}: {pat_map[f'P{p}']['matches']}/{pat_map[f'P{p}']['total']} ({pat_map[f'P{p}']['accuracy_pct']:.2f}%)"
                for p in (1, 2, 3, 4)
                if f"P{p}" in pat_map
            )
            print(f"{rule_name:<34} | {row_s}")

    if plt is not None:
        print(f"\nPlots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
