"""A3: Pareto Frontier — Accuracy vs Tokens / Accuracy vs Latency.

Plots per-dataset Pareto frontiers to visualise the cost-accuracy
trade-off across the four patterns.

Usage:
    python scripts/analysis_pareto.py --input telemetry/per_item_scores.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _is_pareto_optimal(points: list[dict], idx: int) -> bool:
    """A point is Pareto-optimal if no other point has both >= accuracy
    AND <= cost (with at least one strict)."""
    p = points[idx]
    for j, q in enumerate(points):
        if j == idx:
            continue
        if q["accuracy"] >= p["accuracy"] and q["cost"] <= p["cost"]:
            if q["accuracy"] > p["accuracy"] or q["cost"] < p["cost"]:
                return False
    return True


def plot_pareto(
    df: pd.DataFrame,
    dataset: str,
    cost_col: str,
    cost_label: str,
    ax: plt.Axes,
    dev_label: str = "",
) -> None:
    sub = df.groupby("pattern").agg(
        accuracy=("headline_bin", "mean"),
        cost=(cost_col, "mean"),
        n_items=("headline_bin", "count"),
    ).reset_index()
    sub["accuracy"] = sub["accuracy"].astype(float)

    points: list[dict] = []
    for _, row in sub.iterrows():
        points.append({
            "pattern": int(row["pattern"]),
            "accuracy": row["accuracy"],
            "cost": row["cost"],
            "n_items": int(row["n_items"]),
        })

    for p in points:
        pareto = _is_pareto_optimal(points, points.index(p))
        marker = "o" if pareto else "s"
        label = f"P{p['pattern']} {'(Pareto)' if pareto else ''}"
        ax.scatter(p["cost"], p["accuracy"] * 100, marker=marker, s=100, label=label)
        offset_x = p["cost"] * 0.03
        offset_y = 0.5
        ax.annotate(
            f"P{p['pattern']} ({p['accuracy']*100:.0f}%, {p['cost']:.0f})",
            (p["cost"], p["accuracy"] * 100),
            (p["cost"] + offset_x, p["accuracy"] * 100 + offset_y),
            fontsize=8,
        )

    # Draw Pareto frontier
    pareto_points = [p for p in points if _is_pareto_optimal(points, points.index(p))]
    pareto_points.sort(key=lambda p: p["cost"])
    if len(pareto_points) >= 2:
        frontier_x = [p["cost"] for p in pareto_points]
        frontier_y = [p["accuracy"] * 100 for p in pareto_points]
        ax.plot(frontier_x, frontier_y, "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel(cost_label)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Pareto Frontier — {dataset} ({cost_label}){dev_label}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto Frontier analysis.")
    parser.add_argument("--input", type=str, required=True, help="Path to per_item_scores.csv")
    parser.add_argument("--output-dir", type=str, default="telemetry", help="Output directory")
    parser.add_argument("--dev", action="store_true", help="Add '(DEV ONLY)' label to plot titles")
    args = parser.parse_args()
    dev_label = " (DEV ONLY)" if args.dev else ""

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["headline_bin"]).copy()
    df["headline_bin"] = df["headline_bin"].astype(int)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].unique())
    cost_configs = [
        ("tokens_completion", "Avg Tokens per Query"),
        ("latency_ms", "Avg Latency per Query (ms)"),
    ]

    all_results: dict = {}

    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset]
        ds_results: list[dict] = []
        for cost_col, cost_label in cost_configs:
            sub = ds_df.groupby("pattern").agg(
                accuracy=("headline_bin", "mean"),
                cost=(cost_col, "mean"),
                n_items=("headline_bin", "count"),
            ).reset_index()

            for _, row in sub.iterrows():
                p = int(row["pattern"])
                ds_results.append({
                    "pattern": p,
                    "cost_metric": cost_col,
                    "accuracy": round(float(row["accuracy"]), 4),
                    "cost": round(float(row["cost"]), 1),
                    "n_items": int(row["n_items"]),
                    "pareto_optimal": False,
                })

        # Mark Pareto-optimal for each cost metric
        for cost_col, _ in cost_configs:
            points = [r for r in ds_results if r["cost_metric"] == cost_col]
            for i, r in enumerate(points):
                r["pareto_optimal"] = _is_pareto_optimal(points, i)

        all_results[dataset] = ds_results

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax_idx, (cost_col, cost_label) in enumerate(cost_configs):
            plot_pareto(ds_df, dataset, cost_col, cost_label, axes[ax_idx], dev_label)
        plt.tight_layout()
        fig.savefig(plots_dir / f"pareto_{dataset}.png", dpi=150)
        plt.close(fig)

    results_path = output_dir / "pareto_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Summary
    print(f"\n=== Pareto Summary ===")
    print(f"{'Dataset':<10} {'Pattern':<8} {'Accuracy':<10} {'Avg Tokens':<12} {'Tok Pareto':<12} {'Avg Lat(ms)':<12} {'Lat Pareto':<12}")
    print("-" * 80)
    for dataset in datasets:
        tokens_data = {r["pattern"]: r for r in all_results[dataset] if r["cost_metric"] == "tokens_completion"}
        lat_data = {r["pattern"]: r for r in all_results[dataset] if r["cost_metric"] == "latency_ms"}
        for p in sorted(tokens_data):
            tr = tokens_data[p]
            lr = lat_data[p]
            print(
                f"{dataset:<10} P{p:<7} {tr['accuracy']*100:>5.1f}%   "
                f"{tr['cost']:>8.1f}      {'Yes' if tr['pareto_optimal'] else 'No':<12} "
                f"{lr['cost']:>8.1f}      {'Yes' if lr['pareto_optimal'] else 'No':<12}"
            )

    print(f"\nPlots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
