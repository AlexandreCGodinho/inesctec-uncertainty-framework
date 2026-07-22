"""A2: AUROC & Risk-Coverage analysis.

Computes AUROC of U(x) as an incorrect-answer detector and generates
Risk-Coverage curves per (dataset, pattern). Also reports the failure
detector with and without the u_R component.

Usage:
    python scripts/analysis_auroc.py --input telemetry/per_item_scores.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _compute_surface_no_uR(row: pd.Series) -> float:
    active = [row[c] for c in ("u_L", "u_F", "u_P") if pd.notna(row[c])]
    return float(np.mean(active)) if active else np.nan


def _refinement_triggered(row: pd.Series) -> str:
    if row["pattern"] == 1:
        return "No"
    if pd.notna(row["u_R"]):
        return "Yes"
    return "No"


def compute_auroc(df: pd.DataFrame, score_col: str, label_col: str = "headline_bin") -> float:
    """AUROC of U(x) as a failure detector. Positive class = failure (headline_bin=0)."""
    valid = df.dropna(subset=[score_col, label_col])
    valid = valid[(valid[label_col] == 0) | (valid[label_col] == 1)]
    if len(valid) < 2 or len(valid[valid[label_col] == 0]) == 0 or len(valid[valid[label_col] == 1]) == 0:
        return np.nan
    y_true = 1 - valid[label_col].astype(int)
    return float(roc_auc_score(y_true, valid[score_col].astype(float)))


def plot_roc_curve(df: pd.DataFrame, dataset: str, score_col: str, label_col: str, ax: plt.Axes, dev_label: str = "") -> None:
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
    ax.set_title(f"ROC — {dataset}{dev_label}")
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


def plot_risk_coverage(df: pd.DataFrame, dataset: str, score_col: str, ax: plt.Axes, dev_label: str = "") -> None:
    patterns = sorted(df["pattern"].unique())
    for p in patterns:
        sub = df[df["pattern"] == p]
        rc = compute_risk_coverage(sub, score_col)
        if rc.empty:
            continue
        ax.plot(rc["coverage"] * 100, rc["accuracy"] * 100, marker=".", label=f"P{p}")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Risk-Coverage — {dataset}{dev_label}")
    ax.legend(loc="lower left")
    ax.set_xlim(105, 0)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="AUROC & Risk-Coverage analysis.")
    parser.add_argument("--input", type=str, required=True, help="Path to per_item_scores.csv")
    parser.add_argument("--output-dir", type=str, default="telemetry", help="Output directory for results and plots")
    parser.add_argument("--dev", action="store_true", help="Add '(DEV ONLY)' label to plot titles")
    args = parser.parse_args()
    dev_label = " (DEV ONLY)" if args.dev else ""

    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Drop rows with no headline
    df = df.dropna(subset=["headline_bin"]).copy()
    df["headline_bin"] = df["headline_bin"].astype(int)
    df["refinement_triggered"] = df.apply(_refinement_triggered, axis=1)

    # Compute U_surface_no_uR (surface score without u_R component)
    df["U_surface_no_uR"] = df.apply(_compute_surface_no_uR, axis=1)

    datasets = sorted(df["dataset"].unique())
    patterns = sorted(df["pattern"].unique())

    results: dict = {}
    score_cols = ["U_surface", "U_composite", "U_surface_no_uR"]

    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset]
        results[dataset] = {}
        for p in patterns:
            pat_df = ds_df[ds_df["pattern"] == p]
            results[dataset][f"P{p}"] = {}
            for col in score_cols:
                auc = compute_auroc(pat_df, col)
                results[dataset][f"P{p}"][col] = auc

            # Count refinement triggered
            ref_counts = pat_df["refinement_triggered"].value_counts().to_dict()
            results[dataset][f"P{p}"]["refinement_triggered_counts"] = {
                str(k): int(v) for k, v in ref_counts.items()
            }
            results[dataset][f"P{p}"]["n_items"] = len(pat_df)

        # ROC curves per dataset
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax_idx, col in enumerate(score_cols):
            plot_roc_curve(ds_df, dataset, col, "headline_bin", axes[ax_idx], dev_label)
            axes[ax_idx].set_title(f"ROC ({col}) — {dataset}{dev_label}")
        plt.tight_layout()
        fig.savefig(plots_dir / f"auroc_{dataset}.png", dpi=150)
        plt.close(fig)

        # Risk-Coverage per dataset
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax_idx, col in enumerate(score_cols):
            plot_risk_coverage(ds_df, dataset, col, axes[ax_idx], dev_label)
            axes[ax_idx].set_title(f"Risk-Coverage ({col}) — {dataset}{dev_label}")
        plt.tight_layout()
        fig.savefig(plots_dir / f"risk_coverage_{dataset}.png", dpi=150)
        plt.close(fig)

    # Save results as JSON
    results_path = output_dir / "auroc_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Summary table
    print("\n=== AUROC Summary ===")
    print(f"{'Dataset':<10} {'Pattern':<8} {'U_surface':<12} {'U_composite':<14} {'U_surface_no_uR':<16} {'N':<6}")
    print("-" * 70)
    for dataset in datasets:
        for p in patterns:
            r = results[dataset].get(f"P{p}", {})
            us = r.get("U_surface", "N/A")
            uc = r.get("U_composite", "N/A")
            usnr = r.get("U_surface_no_uR", "N/A")
            n = r.get("n_items", 0)
            us_s = f"{us:.3f}" if isinstance(us, float) else str(us)
            uc_s = f"{uc:.3f}" if isinstance(uc, float) else str(uc)
            usnr_s = f"{usnr:.3f}" if isinstance(usnr, float) else str(usnr)
            print(f"{dataset:<10} {f'P{p}':<8} {us_s:<12} {uc_s:<14} {usnr_s:<16} {n:<6}")

    print(f"\nPlots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
