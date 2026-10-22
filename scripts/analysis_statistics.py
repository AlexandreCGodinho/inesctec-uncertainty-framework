"""A4: Statistical Significance & Variance.

Performs paired significance tests across (pattern, dataset) pairs:
  - McNemar's test for binary headline datasets (PopQA, FinQA, IFEval)
  - Paired Wilcoxon signed-rank test on per-item scores as primary test for LiveBench
  - Secondary McNemar test binarised at accuracy == 1.0 for LiveBench
  - Paired bootstrap CIs for LiveBench numeric scores
  - Benjamini-Hochberg FDR correction across the test family
  - Pattern 4 between-seed variance

Usage:
    python scripts/analysis_statistics.py --input data/audits/per_item_scores_v11.csv
    python scripts/analysis_statistics.py --input data/audits/per_item_scores_v11.csv --dev
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests


_BINARY_DATASETS = ("popqa", "finqa", "ifeval")
_PATTERN_PAIRS = [(2, 1), (3, 1), (4, 1)]
_ALL_PATTERN_PAIRS = [(2, 1), (3, 1), (4, 1), (3, 2), (4, 2), (4, 3)]


def _mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        return float(binomtest(b, n, p=0.5).pvalue)
    stat = float((b - c) ** 2) / float(n)
    return float(chi2.sf(stat, 1))


def _paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_resamples)
    ])
    ci = np.percentile(boot_means, [2.5, 97.5]).tolist()
    p_val = 2.0 * min(
        float(np.mean(boot_means >= 0)),
        float(np.mean(boot_means <= 0)),
    )
    return {
        "observed_mean_diff": float(diffs.mean()),
        "ci_95": [float(ci[0]), float(ci[1])],
        "p_value": min(p_val, 1.0),
    }


def _paired_wilcoxon_test(scores_px: np.ndarray, scores_py: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test on per-item continuous scores (primary for LiveBench)."""
    diffs = np.asarray(scores_px, dtype=float) - np.asarray(scores_py, dtype=float)
    n = int(len(diffs))
    if n == 0:
        return {
            "statistic": None,
            "p_value": 1.0,
            "observed_mean_diff": None,
            "n_nonzero": 0,
            "n": 0,
        }
    nonzero = diffs[diffs != 0.0]
    n_nonzero = int(len(nonzero))
    mean_diff = float(diffs.mean())
    if n_nonzero == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "observed_mean_diff": mean_diff,
            "n_nonzero": 0,
            "n": n,
        }
    res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "observed_mean_diff": mean_diff,
        "n_nonzero": n_nonzero,
        "n": n,
    }


def run_livebench_wilcoxon_analysis(
    df: pd.DataFrame,
    pairs: list[tuple[int, int]] = _ALL_PATTERN_PAIRS,
) -> dict:
    """Run paired Wilcoxon signed-rank test on LiveBench per-item accuracy scores."""
    lb_df = df[df["dataset"] == "livebench"].copy()
    if lb_df.empty:
        return {}
    lb_df["pattern"] = lb_df["pattern"].astype(str)
    lb_df = lb_df.dropna(subset=["accuracy"])
    lb_df["accuracy"] = lb_df["accuracy"].astype(float)

    results: dict = {}
    for px, py in pairs:
        pair_key = f"p{px}_vs_p{py}"
        px_df = lb_df[lb_df["pattern"] == str(px)]
        py_df = lb_df[lb_df["pattern"] == str(py)]
        py_scores = py_df.set_index("query_id")["accuracy"]

        if px == 4:
            seed_results = []
            for seed in sorted(px_df["seed"].unique()):
                sub = px_df[px_df["seed"] == seed].set_index("query_id")["accuracy"]
                common = py_scores.index.intersection(sub.index)
                if len(common) < 2:
                    seed_results.append({
                        "seed": int(seed) if str(seed).isdigit() else str(seed),
                        "statistic": None,
                        "p_value": 1.0,
                        "observed_mean_diff": None,
                        "n_nonzero": 0,
                        "n": 0,
                    })
                    continue
                w_res = _paired_wilcoxon_test(sub.loc[common].values, py_scores.loc[common].values)
                w_res["seed"] = int(seed) if str(seed).isdigit() else str(seed)
                seed_results.append(w_res)
            p_vals = [r["p_value"] for r in seed_results if r.get("n", 0) > 0]
            results[pair_key] = {
                "per_seed": seed_results,
                "p_value_median": float(np.median(p_vals)) if p_vals else None,
            }
        else:
            px_scores = px_df.set_index("query_id")["accuracy"]
            common = py_scores.index.intersection(px_scores.index)
            if len(common) < 2:
                results[pair_key] = {
                    "statistic": None,
                    "p_value": 1.0,
                    "observed_mean_diff": None,
                    "n_nonzero": 0,
                    "n": 0,
                }
                continue
            results[pair_key] = _paired_wilcoxon_test(
                px_scores.loc[common].values,
                py_scores.loc[common].values,
            )
    return results


def run_livebench_mcnemar_secondary(
    df: pd.DataFrame,
    pairs: list[tuple[int, int]] = _ALL_PATTERN_PAIRS,
) -> dict:
    """Secondary McNemar check on LiveBench binarised strictly at accuracy == 1.0."""
    lb_df = df[df["dataset"] == "livebench"].copy()
    if lb_df.empty:
        return {}
    lb_df["pattern"] = lb_df["pattern"].astype(str)
    lb_df = lb_df.dropna(subset=["accuracy"])
    lb_df["bin_acc1"] = (lb_df["accuracy"].astype(float) == 1.0).astype(int)

    results: dict = {}
    for px, py in pairs:
        pair_key = f"p{px}_vs_p{py}"
        px_df = lb_df[lb_df["pattern"] == str(px)]
        py_df = lb_df[lb_df["pattern"] == str(py)]
        py_bin = py_df.set_index("query_id")["bin_acc1"]

        if px == 4:
            seed_results = []
            for seed in sorted(px_df["seed"].unique()):
                sub = px_df[px_df["seed"] == seed].set_index("query_id")["bin_acc1"]
                common = py_bin.index.intersection(sub.index)
                if len(common) < 2:
                    seed_results.append({"seed": str(seed), "b": 0, "c": 0, "p_value": 1.0, "n": 0})
                    continue
                py_v = py_bin.loc[common].values
                px_v = sub.loc[common].values
                b = int(np.sum((py_v == 0) & (px_v == 1)))
                c = int(np.sum((py_v == 1) & (px_v == 0)))
                seed_results.append({
                    "seed": int(seed) if str(seed).isdigit() else str(seed),
                    "b": b,
                    "c": c,
                    "p_value": _mcnemar_pvalue(b, c),
                    "n": int(len(common)),
                })
            p_vals = [r["p_value"] for r in seed_results if r["n"] > 0]
            results[pair_key] = {
                "per_seed": seed_results,
                "p_value_median": float(np.median(p_vals)) if p_vals else None,
            }
        else:
            px_bin = px_df.set_index("query_id")["bin_acc1"]
            common = py_bin.index.intersection(px_bin.index)
            if len(common) < 2:
                results[pair_key] = {"b": 0, "c": 0, "p_value": 1.0, "n": 0}
                continue
            py_v = py_bin.loc[common].values
            px_v = px_bin.loc[common].values
            b = int(np.sum((py_v == 0) & (px_v == 1)))
            c = int(np.sum((py_v == 1) & (px_v == 0)))
            results[pair_key] = {
                "b": b,
                "c": c,
                "p_value": _mcnemar_pvalue(b, c),
                "n": int(len(common)),
            }
    return results


def _compute_p4_seed_stats(df: pd.DataFrame) -> dict:
    seeds = sorted(df["seed"].unique())
    acc_per_seed = {}
    for s in seeds:
        sub = df[df["seed"] == s]
        sub = sub.dropna(subset=["headline_bin"])
        sub = sub[(sub["headline_bin"] == 0) | (sub["headline_bin"] == 1)]
        acc = float(sub["headline_bin"].astype(int).mean()) if len(sub) > 0 else None
        acc_per_seed[str(int(s))] = acc
    vals = [v for v in acc_per_seed.values() if v is not None]
    mean_acc = float(np.mean(vals)) if vals else None
    sd_acc = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return {
        "per_seed": acc_per_seed,
        "mean": mean_acc,
        "sd": sd_acc,
    }


def run_mcnemar_analysis(
    df: pd.DataFrame,
    pairs: list[tuple[int, int]] = _PATTERN_PAIRS,
) -> dict:
    results: dict = {}
    for dataset in _BINARY_DATASETS:
        ds_df = df[df["dataset"] == dataset].copy()
        ds_df["pattern"] = ds_df["pattern"].astype(str)
        ds_df = ds_df.dropna(subset=["headline_bin"])
        ds_df["headline_bin"] = ds_df["headline_bin"].astype(str).str.replace(".0", "", regex=False)
        ds_df = ds_df[(ds_df["headline_bin"] == "0") | (ds_df["headline_bin"] == "1")]
        ds_df["headline_bin"] = ds_df["headline_bin"].astype(int)

        if dataset not in results:
            results[dataset] = {}

        for px, py in pairs:
            pair_key = f"p{px}_vs_p{py}"
            px_df = ds_df[ds_df["pattern"] == str(px)]
            queries_py = ds_df[ds_df["pattern"] == str(py)].set_index("query_id")["headline_bin"]

            if px == 4:
                seed_results = []
                for seed in sorted(px_df["seed"].unique()):
                    sub = px_df[px_df["seed"] == seed].set_index("query_id")["headline_bin"]
                    common = queries_py.index.intersection(sub.index)
                    if len(common) < 2:
                        seed_results.append({"b": 0, "c": 0, "p_value": 1.0, "n": 0})
                        continue
                    py_aligned = queries_py.loc[common].values
                    px_aligned = sub.loc[common].values
                    b = int(np.sum((py_aligned == 0) & (px_aligned == 1)))
                    c = int(np.sum((py_aligned == 1) & (px_aligned == 0)))
                    seed_results.append({
                        "b": b,
                        "c": c,
                        "p_value": _mcnemar_pvalue(b, c),
                        "n": int(len(common)),
                    })
                p_vals = [r["p_value"] for r in seed_results if r["n"] > 0]
                results[dataset][pair_key] = {
                    "per_seed": seed_results,
                    "p_value_median": float(np.median(p_vals)) if p_vals else None,
                }
            else:
                py_aligned = queries_py
                px_aligned = px_df.set_index("query_id")["headline_bin"]
                common = py_aligned.index.intersection(px_aligned.index)
                if len(common) < 2:
                    results[dataset][pair_key] = {"b": 0, "c": 0, "p_value": 1.0, "n": 0}
                    continue
                py_v = py_aligned.loc[common].values
                px_v = px_aligned.loc[common].values
                b = int(np.sum((py_v == 0) & (px_v == 1)))
                c = int(np.sum((py_v == 1) & (px_v == 0)))
                results[dataset][pair_key] = {
                    "b": b,
                    "c": c,
                    "p_value": _mcnemar_pvalue(b, c),
                    "n": int(len(common)),
                }

    return results


def run_bootstrap_analysis(df: pd.DataFrame) -> dict:
    results: dict = {}
    lb_df = df[df["dataset"] == "livebench"].copy()

    if lb_df.empty:
        return results

    # Determine categories if they exist
    if "category" in lb_df.columns:
        categories = sorted(lb_df["category"].unique())
    else:
        categories = ["overall"]

    for category in categories:
        if category == "overall":
            cat_df = lb_df
        else:
            cat_df = lb_df[lb_df["category"] == category]

        cat_df = cat_df.dropna(subset=["accuracy"])
        results[category] = {}

        p1_scores = cat_df[cat_df["pattern"] == "1"].set_index("query_id")["accuracy"]

        for px, p1 in _PATTERN_PAIRS:
            pair_key = f"p{px}_vs_p1"
            px_df = cat_df[cat_df["pattern"] == str(px)]

            if px == 4:
                seed_results = []
                for seed in sorted(px_df["seed"].unique()):
                    sub = px_df[px_df["seed"] == seed].set_index("query_id")["accuracy"]
                    common = p1_scores.index.intersection(sub.index)
                    if len(common) < 2:
                        seed_results.append({"observed_mean_diff": None, "ci_95": None, "p_value": 1.0, "n": 0})
                        continue
                    diffs = sub.loc[common].values - p1_scores.loc[common].values
                    res = _paired_bootstrap_ci(diffs)
                    res["n"] = int(len(common))
                    seed_results.append(res)
                p_vals = [r["p_value"] for r in seed_results if r.get("n", 0) > 0]
                results[category][pair_key] = {
                    "per_seed": seed_results,
                    "p_value_median": float(np.median(p_vals)) if p_vals else None,
                }
            else:
                px_scores = px_df.set_index("query_id")["accuracy"]
                common = p1_scores.index.intersection(px_scores.index)
                if len(common) < 2:
                    results[category][pair_key] = {"observed_mean_diff": None, "ci_95": None, "p_value": 1.0, "n": 0}
                    continue
                diffs = px_scores.loc[common].values - p1_scores.loc[common].values
                res = _paired_bootstrap_ci(diffs)
                res["n"] = int(len(common))
                results[category][pair_key] = res

    return results


def run_fdr(pvals: list[float], alpha: float = 0.05) -> dict:
    if not pvals:
        return {"n_tests": 0, "rejected": 0, "corrected": []}
    rejected, corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return {
        "n_tests": int(len(pvals)),
        "alpha": alpha,
        "corrected": [float(c) for c in corrected],
        "rejected": [bool(r) for r in rejected],
        "n_rejected": int(rejected.sum()),
    }


def _collect_pvals(mcnemar: dict, bootstrap: dict) -> list[float]:
    pvals = []
    for dataset, pairs in sorted(mcnemar.items()):
        for pair_key, v in sorted(pairs.items()):
            if "per_seed" in v:
                pvals.append(v["p_value_median"])
            elif isinstance(v.get("p_value"), (int, float)):
                pvals.append(v["p_value"])
    for category, pairs in sorted(bootstrap.items()):
        for pair_key, v in sorted(pairs.items()):
            if "per_seed" in v:
                pvals.append(v["p_value_median"])
            elif isinstance(v.get("p_value"), (int, float)):
                pvals.append(v["p_value"])
    return pvals


def run_p4_variance_analysis(df: pd.DataFrame) -> dict:
    results: dict = {}
    for dataset in _BINARY_DATASETS:
        ds_df = df[(df["dataset"] == dataset) & (df["pattern"] == "4")].copy()
        ds_df = ds_df.dropna(subset=["headline_bin"])
        ds_df = ds_df[(ds_df["headline_bin"] == "0") | (ds_df["headline_bin"] == "1")]
        ds_df["headline_bin"] = ds_df["headline_bin"].astype(int)
        if ds_df.empty:
            continue
        results[dataset] = _compute_p4_seed_stats(ds_df)

    lb_df = df[(df["dataset"] == "livebench") & (df["pattern"] == "4")].copy()
    lb_df = lb_df.dropna(subset=["accuracy"])
    if not lb_df.empty:
        seeds = sorted(lb_df["seed"].unique())
        acc_per_seed = {}
        for s in seeds:
            sub = lb_df[lb_df["seed"] == s]
            acc_per_seed[str(int(s))] = float(sub["accuracy"].mean()) if len(sub) > 0 else None
        vals = [v for v in acc_per_seed.values() if v is not None]
        results["livebench"] = {
            "per_seed": acc_per_seed,
            "mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        }

    return results


def _fmt(val, decimals=4):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical significance & variance analysis.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/audits/per_item_scores_v11.csv",
        help="Path to per_item_scores_v11.csv",
    )
    parser.add_argument("--output-dir", type=str, default="telemetry", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR significance threshold")
    parser.add_argument("--dev", action="store_true", help="Add '(DEV ONLY)' label to output")
    args = parser.parse_args()
    dev_label = " (DEV ONLY)" if args.dev else ""

    df = pd.read_csv(args.input, dtype={"pattern": str, "headline_bin": str})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcnemar_results = run_mcnemar_analysis(df, pairs=_ALL_PATTERN_PAIRS)
    lb_wilcoxon_results = run_livebench_wilcoxon_analysis(df, pairs=_ALL_PATTERN_PAIRS)
    lb_mcnemar_sec = run_livebench_mcnemar_secondary(df, pairs=_ALL_PATTERN_PAIRS)
    bootstrap_results = run_bootstrap_analysis(df)

    all_pvals = _collect_pvals(mcnemar_results, {"overall": lb_wilcoxon_results})
    fdr_results = run_fdr(all_pvals, alpha=args.alpha)

    p4_variance = run_p4_variance_analysis(df)

    results = {
        "mcnemar": mcnemar_results,
        "livebench_wilcoxon_primary": lb_wilcoxon_results,
        "livebench_mcnemar_secondary_acc1": lb_mcnemar_sec,
        "paired_bootstrap": bootstrap_results,
        "fdr": fdr_results,
        "p4_between_seed_variance": p4_variance,
    }

    results_path = output_dir / "statistical_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Summary tables
    print(f"\n=== McNemar Tests{dev_label} ===")
    print(f"{'Dataset':<10} {'Pair':<12} {'b(P1=0,PX=1)':<15} {'c(P1=1,PX=0)':<15} {'p-value':<10} {'n':<6}")
    print("-" * 68)
    for dataset in sorted(mcnemar_results):
        for pair_key, v in sorted(mcnemar_results[dataset].items()):
            if "per_seed" in v:
                pv = _fmt(v["p_value_median"])
                b_str = ", ".join(str(s["b"]) for s in v["per_seed"])
                c_str = ", ".join(str(s["c"]) for s in v["per_seed"])
                n_str = ", ".join(str(s["n"]) for s in v["per_seed"])
                print(f"{dataset:<10} {pair_key:<12} {b_str:<15} {c_str:<15} {pv:<10} {n_str:<6}")
            else:
                pv = _fmt(v["p_value"])
                print(f"{dataset:<10} {pair_key:<12} {v['b']:<15} {v['c']:<15} {pv:<10} {v['n']:<6}")

    print(f"\n=== LiveBench Paired Wilcoxon (Primary){dev_label} ===")
    print(f"{'Pair':<12} {'Mean Diff':<24} {'p-value':<12} {'n_nonzero':<15} {'n':<6}")
    print("-" * 72)
    for pair_key, v in sorted(lb_wilcoxon_results.items()):
        if "per_seed" in v:
            md = ", ".join(_fmt(s.get("observed_mean_diff")) for s in v["per_seed"])
            pv = _fmt(v["p_value_median"])
            nnz = ", ".join(str(s.get("n_nonzero", 0)) for s in v["per_seed"])
            n_str = ", ".join(str(s.get("n", 0)) for s in v["per_seed"])
            print(f"{pair_key:<12} {md:<24} {pv:<12} {nnz:<15} {n_str:<6}")
        else:
            md = _fmt(v.get("observed_mean_diff"))
            pv = _fmt(v.get("p_value"))
            nnz = str(v.get("n_nonzero", 0))
            print(f"{pair_key:<12} {md:<24} {pv:<12} {nnz:<15} {v.get('n', 0):<6}")

    print(f"\n=== LiveBench McNemar Binarised at accuracy==1.0 (Secondary){dev_label} ===")
    print(f"{'Pair':<12} {'b(PY=0,PX=1)':<15} {'c(PY=1,PX=0)':<15} {'p-value':<10} {'n':<6}")
    print("-" * 62)
    for pair_key, v in sorted(lb_mcnemar_sec.items()):
        if "per_seed" in v:
            pv = _fmt(v["p_value_median"])
            b_str = ", ".join(str(s["b"]) for s in v["per_seed"])
            c_str = ", ".join(str(s["c"]) for s in v["per_seed"])
            n_str = ", ".join(str(s["n"]) for s in v["per_seed"])
            print(f"{pair_key:<12} {b_str:<15} {c_str:<15} {pv:<10} {n_str:<6}")
        else:
            pv = _fmt(v["p_value"])
            print(f"{pair_key:<12} {v['b']:<15} {v['c']:<15} {pv:<10} {v['n']:<6}")

    print(f"\n=== Paired Bootstrap (LiveBench){dev_label} ===")
    print(f"{'Category':<12} {'Pair':<12} {'Mean Diff':<12} {'CI 95%':<20} {'p-value':<10} {'n':<6}")
    print("-" * 72)
    for category in sorted(bootstrap_results):
        for pair_key, v in sorted(bootstrap_results[category].items()):
            if "per_seed" in v:
                md = ", ".join(_fmt(s.get("observed_mean_diff")) for s in v["per_seed"])
                ci = ", ".join(str(s.get("ci_95")) for s in v["per_seed"])
                pv = _fmt(v["p_value_median"])
                n_str = ", ".join(str(s.get("n", 0)) for s in v["per_seed"])
                print(f"{category:<12} {pair_key:<12} {md:<12} {ci:<20} {pv:<10} {n_str:<6}")
            else:
                md = _fmt(v.get("observed_mean_diff"))
                ci = str(v.get("ci_95"))
                pv = _fmt(v.get("p_value"))
                print(f"{category:<12} {pair_key:<12} {md:<12} {ci:<20} {pv:<10} {v.get('n', 0):<6}")

    print(f"\n=== FDR Correction (BH){dev_label} ===")
    print(f"Number of tests: {fdr_results['n_tests']}")
    print(f"Rejected at alpha={args.alpha}: {fdr_results['n_rejected']}")
    if fdr_results["n_tests"] > 0:
        for i, (p, r) in enumerate(zip(fdr_results["corrected"], fdr_results["rejected"])):
            print(f"  Test {i+1}: corrected p = {p:.4f} {'✓' if r else '✗'}")

    print(f"\n=== P4 Between-Seed Variance{dev_label} ===")
    print(f"{'Dataset':<12} {'Mean Acc':<12} {'SD':<12} {'Per-Seed':<24}")
    print("-" * 60)
    for dataset in sorted(p4_variance):
        v = p4_variance[dataset]
        m = _fmt(v.get("mean"), 4)
        s = _fmt(v.get("sd"), 4)
        ps = ", ".join(f"{k}={_fmt(val, 4)}" for k, val in sorted(v.get("per_seed", {}).items()))
        print(f"{dataset:<12} {m:<12} {s:<12} {ps:<24}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
