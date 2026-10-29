#!/usr/bin/env python3
"""
Generate Pattern 4 per-cell u_F coverage and composite U(x) summary tables (v11).
Read-only ingestion of audit records from data/audits/audit_recomputed_uf_v11/
and U_composite from data/audits/per_item_scores_v11.csv.
Compatible with the Python standard library (no mandatory external dependencies).
"""

import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = REPO_ROOT / "data" / "audits" / "audit_recomputed_uf_v11"
SCORES_CSV = REPO_ROOT / "data" / "audits" / "per_item_scores_v11.csv"

DATASETS = ["popqa", "ifeval", "livebench", "finqa"]
DATASET_LABELS = {
    "popqa": "PopQA",
    "ifeval": "IFEval",
    "livebench": "LiveBench",
    "finqa": "FinQA",
}
SEEDS = [1234, 2345, 3456]

SEED_FOLDERS = {
    1234: "report_run_a100",
    2345: "report_run_a100_seed2345",
    3456: "report_run_a100_seed3456",
}


def load_composite_lookup(csv_path: Path = SCORES_CSV) -> dict:
    """Load U_composite keyed by (dataset, seed, str(query_id)) for Pattern 4."""
    if not csv_path.exists():
        return {}
    lookup = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("pattern", "")) != "4":
                continue
            ds = row["dataset"]
            seed = int(row["seed"])
            qid = str(row["query_id"])
            uc_raw = row.get("U_composite", "")
            uc_val = float(uc_raw) if uc_raw not in ("", None) else None
            lookup[(ds, seed, qid)] = uc_val
    return lookup


def load_all_records(audit_dir: Path = AUDIT_DIR, csv_path: Path = SCORES_CSV):
    """Load all 1,800 audit records across 12 experimental cells in read-only mode."""
    composite_lookup = load_composite_lookup(csv_path)
    records = []
    for d in DATASETS:
        for s in SEEDS:
            folder = SEED_FOLDERS[s]
            filename = f"{d}_pattern4.jsonl"
            filepath = audit_dir / folder / filename

            if not filepath.exists():
                raise FileNotFoundError(f"Missing audit file: {filepath}")

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    qid = str(item.get("query_id"))
                    records.append({
                        "dataset": d,
                        "seed": s,
                        "query_id": qid,
                        "u_F": item.get("u_F"),
                        "U_composite": composite_lookup.get((d, s, qid)),
                        "span_category": item.get("span_category"),
                        "judge_parse_error": bool(item.get("judge_parse_error", False)),
                        "raw_claims_count": item.get("raw_claims_count", 0),
                        "span_claims_count": item.get("span_claims_count", 0),
                        "extracted_answer_span": item.get("extracted_answer_span", ""),
                    })
    return records


def compute_statistics(values):
    """Compute standard descriptive statistics for valid numerical values."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    count = len(values)
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if count > 1 else 0.0
    median_val = statistics.median(values)
    min_val = min(values)
    max_val = max(values)
    return {
        "count": count,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
    }


def build_appendix_tables(records=None):
    """Build structured per-cell summary rows for P4 u_F coverage and (u_F, U_composite) distributions."""
    if records is None:
        records = load_all_records()

    cell_items = {}
    for r in records:
        key = (r["dataset"], r["seed"])
        cell_items.setdefault(key, []).append(r)

    t1_rows = []
    g_tot = g_val = g_zero = g_jerr = g_nrev = g_empt = 0

    for d in DATASETS:
        d_tot = d_val = d_zero = d_jerr = d_nrev = d_empt = 0
        label = DATASET_LABELS[d]

        for s in SEEDS:
            items = cell_items[(d, s)]
            total = len(items)
            valid_items = [it for it in items if it["u_F"] is not None]
            valid = len(valid_items)
            valid_pct = (valid / total) * 100.0 if total > 0 else 0.0

            none_items = [it for it in items if it["u_F"] is None]
            needs_revision = sum(1 for it in none_items if it["span_category"] == "needs_revision")
            judge_error = sum(1 for it in none_items if it["judge_parse_error"])
            empty_span = sum(
                1
                for it in none_items
                if it["extracted_answer_span"] == ""
                and it["span_category"] != "needs_revision"
                and not it["judge_parse_error"]
            )
            zero_claims = sum(
                1
                for it in none_items
                if it["span_claims_count"] == 0
                and not it["judge_parse_error"]
                and it["span_category"] != "needs_revision"
                and it["extracted_answer_span"] != ""
            )

            t1_rows.append({
                "dataset": label,
                "seed": str(s),
                "total": total,
                "valid": valid,
                "pct": f"{valid_pct:.1f}",
                "zero_claims": zero_claims,
                "judge_error": judge_error,
                "needs_revision": needs_revision,
                "empty_span": empty_span,
                "is_subtotal": False,
            })

            d_tot += total
            d_val += valid
            d_zero += zero_claims
            d_jerr += judge_error
            d_nrev += needs_revision
            d_empt += empty_span

        d_pct = (d_val / d_tot) * 100.0 if d_tot > 0 else 0.0
        t1_rows.append({
            "dataset": label,
            "seed": "All",
            "total": d_tot,
            "valid": d_val,
            "pct": f"{d_pct:.1f}",
            "zero_claims": d_zero,
            "judge_error": d_jerr,
            "needs_revision": d_nrev,
            "empty_span": d_empt,
            "is_subtotal": True,
        })

        g_tot += d_tot
        g_val += d_val
        g_zero += d_zero
        g_jerr += d_jerr
        g_nrev += d_nrev
        g_empt += d_empt

    g_pct = (g_val / g_tot) * 100.0 if g_tot > 0 else 0.0
    t1_rows.append({
        "dataset": "Total",
        "seed": "All",
        "total": g_tot,
        "valid": g_val,
        "pct": f"{g_pct:.1f}",
        "zero_claims": g_zero,
        "judge_error": g_jerr,
        "needs_revision": g_nrev,
        "empty_span": g_empt,
        "is_subtotal": True,
    })

    # Descriptive statistics of u_F and U_composite across 12 P4 cells
    t2_rows = []
    global_u_f = []
    global_u_c = []

    def _fmt_stats_row(ds_lbl, seed_lbl, uf_vals, uc_vals, is_sub):
        sf = compute_statistics(uf_vals)
        sc = compute_statistics(uc_vals)
        return {
            "dataset": ds_lbl,
            "seed": seed_lbl,
            "count": sf["count"],
            "uf_mean_sd": f"{sf['mean']:.3f} +- {sf['std']:.3f}" if sf["mean"] is not None else "N/A",
            "uf_median": f"{sf['median']:.3f}" if sf["median"] is not None else "N/A",
            "uf_range": f"{sf['min']:.2f}..{sf['max']:.2f}" if sf["min"] is not None else "N/A",
            "ux_mean_sd": f"{sc['mean']:.3f} +- {sc['std']:.3f}" if sc["mean"] is not None else "N/A",
            "ux_median": f"{sc['median']:.3f}" if sc["median"] is not None else "N/A",
            "ux_range": f"{sc['min']:.2f}..{sc['max']:.2f}" if sc["min"] is not None else "N/A",
            "uf_stats": sf,
            "ux_stats": sc,
            "is_subtotal": is_sub,
        }

    for d in DATASETS:
        label = DATASET_LABELS[d]
        d_u_f = []
        d_u_c = []
        for s in SEEDS:
            items = cell_items[(d, s)]
            uf_vals = [float(it["u_F"]) for it in items if it["u_F"] is not None]
            uc_vals = [
                float(it["U_composite"])
                for it in items
                if it["u_F"] is not None and it["U_composite"] is not None
            ]
            d_u_f.extend(uf_vals)
            d_u_c.extend(uc_vals)
            global_u_f.extend(uf_vals)
            global_u_c.extend(uc_vals)
            t2_rows.append(_fmt_stats_row(label, str(s), uf_vals, uc_vals, False))

        t2_rows.append(_fmt_stats_row(label, "All", d_u_f, d_u_c, True))

    t2_rows.append(_fmt_stats_row("Total", "All", global_u_f, global_u_c, True))
    return t1_rows, t2_rows


def main():
    t1_rows, t2_rows = build_appendix_tables()

    print("\n# PATTERN 4 u_F COVERAGE AND REASONS FOR NONE (12 CELLS)\n")
    print("| Benchmark | Seed | u_F defined | Zero claims | Parse error | Abstention | Empty span | Coverage (%) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in t1_rows:
        bold = "**" if r["is_subtotal"] else ""
        print(
            f"| {bold}{r['dataset']}{bold} | {r['seed']} | {r['valid']} | "
            f"{r['zero_claims']} | {r['judge_error']} | {r['needs_revision']} | "
            f"{r['empty_span']} | {r['pct']} |"
        )

    print("\n# DESCRIPTIVE STATISTICS OF u_F AND U(x) ACROSS 12 P4 CELLS\n")
    print("| Benchmark | Seed | n | u_F Mean +- SD | u_F Median | u_F Range | U(x) Mean +- SD | U(x) Median | U(x) Range |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in t2_rows:
        bold = "**" if r["is_subtotal"] else ""
        print(
            f"| {bold}{r['dataset']}{bold} | {r['seed']} | {r['count']} | "
            f"{r['uf_mean_sd']} | {r['uf_median']} | {r['uf_range']} | "
            f"{r['ux_mean_sd']} | {r['ux_median']} | {r['ux_range']} |"
        )


if __name__ == "__main__":
    main()
