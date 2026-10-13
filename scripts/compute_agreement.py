#!/usr/bin/env python3
"""
Agreement Metrics for FinQA Labeling (Doc 10 + Doc 07)

Computes:
- % Agreement on extracted values (Doc 07 tolerance, unit normalization)
- Cohen's Kappa on abstention decision only (binary: abstain vs non-abstain)
"""

import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score


# ─── Configuration ──────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEDI_FILE = os.path.join(BASE_DIR, "data/finqa_labeling/finqa_labeling_grid_fixed.csv")
PROF_ALEXANDRE_FILE = os.path.join(BASE_DIR, "data/finqa_labeling/finqa_labeling_grid_fixed_helper_columns_AG.xlsx")
PROF_ALEXANDRE_SHEET = "FinQA Labeling"

# NOTE: FEDI_FILE points to the ORIGINAL raw labels (3 Oct), not the modified version.
# All corrections from the adjudication process are kept in finqa_labeling_grid_adjudicated.csv.

PATTERNS = ["A", "B", "C", "D"]
NUM_TOL = 0.01          # Doc 07: absolute tolerance
REL_TOL = 0.01          # Doc 07: relative tolerance (1%)

SUFFIX_FEDI = "fedi"
SUFFIX_PROF = "prof_alexandre"


# ─── Helpers ────────────────────────────────────────────────────
def normalize_unit(u) -> str:
    """Doc 07 unit normalization: strip $, lowercase, strip spaces."""
    if pd.isna(u):
        return "__NA__"
    s = str(u).replace("$", "").strip().lower()
    return s if s else "__EMPTY__"


def values_match(v1, v2) -> bool:
    """Doc 07 numeric tolerance."""
    try:
        f1, f2 = float(v1), float(v2)
        if abs(f1 - f2) <= NUM_TOL:
            return True
        if abs(f1) > 1e-9 and abs(f1 - f2) / abs(f1) <= REL_TOL:
            return True
        return False
    except (ValueError, TypeError):
        return str(v1).strip().lower() == str(v2).strip().lower()


def units_match(u1, u2) -> bool:
    """Check if units agree after normalization."""
    return normalize_unit(u1) == normalize_unit(u2)


def get_abstain(row, pattern, suffix) -> bool:
    """Get abstention decision as native bool."""
    return bool(row[f"abstain_{pattern}_{suffix}"])


def both_abstained(row, pattern) -> bool:
    """Check if both annotators abstained."""
    return get_abstain(row, pattern, SUFFIX_FEDI) and get_abstain(row, pattern, SUFFIX_PROF)


def one_abstained(row, pattern) -> bool:
    """Check if exactly one annotator abstained."""
    return get_abstain(row, pattern, SUFFIX_FEDI) != get_abstain(row, pattern, SUFFIX_PROF)


def get_reason(row, pattern, suffix) -> str:
    """Get normalized abstention reason."""
    val = row[f"abstain_reason_{pattern}_{suffix}"]
    return str(val).strip().lower() if pd.notna(val) else ""


def cells_match(row, pattern) -> bool:
    """Check if two cells fully agree (abstention + value + unit)."""
    if both_abstained(row, pattern):
        return get_reason(row, pattern, SUFFIX_FEDI) == get_reason(row, pattern, SUFFIX_PROF)
    if one_abstained(row, pattern):
        return False
    return values_match(
        row[f"value_{pattern}_{SUFFIX_FEDI}"], row[f"value_{pattern}_{SUFFIX_PROF}"]
    ) and units_match(row[f"unit_{pattern}_{SUFFIX_FEDI}"], row[f"unit_{pattern}_{SUFFIX_PROF}"])


def value_agreement(row, pattern) -> bool:
    """Check if extracted values agree (non-abstained cells only)."""
    if both_abstained(row, pattern) or one_abstained(row, pattern):
        return False
    return values_match(
        row[f"value_{pattern}_{SUFFIX_FEDI}"], row[f"value_{pattern}_{SUFFIX_PROF}"]
    ) and units_match(row[f"unit_{pattern}_{SUFFIX_FEDI}"], row[f"unit_{pattern}_{SUFFIX_PROF}"])


def classify_disagreement(row) -> str:
    """Classify the type of disagreement."""
    if one_abstained(row, "A"):  # check any pattern
        return "abstain_mismatch"
    if not units_match(row["fedi_unit"], row["prof_alexandre_unit"]):
        return "unit"
    return "value"


def compute_value_agreement_metrics(df_res):
    """Compute % agreement on extracted values (non-abstained only)."""
    val_cells = df_res[df_res["value_agreement"].notna()]
    n_val = len(val_cells)
    n_val_agree = int(val_cells["value_agreement"].sum())
    pct = round(n_val_agree / n_val * 100, 2) if n_val > 0 else 0
    per_pattern = {}
    for p in PATTERNS:
        sub = val_cells[val_cells["pattern"] == p]
        n = len(sub)
        a = int(sub["value_agreement"].sum()) if n > 0 else 0
        per_pattern[p] = {"n_values": n, "agree": a, "pct": round(a / n * 100, 1) if n > 0 else None}
    return n_val, n_val_agree, pct, per_pattern


def compute_kappa_metrics(abstain_labels_fedi, abstain_labels_prof):
    """Compute Cohen's Kappa on abstention decision."""
    kappa = round(cohen_kappa_score(abstain_labels_fedi, abstain_labels_prof), 4)
    fedi_count = int(sum(abstain_labels_fedi))
    prof_count = int(sum(abstain_labels_prof))
    both = int(sum(1 for f, p in zip(abstain_labels_fedi, abstain_labels_prof) if f == 1 and p == 1))
    return kappa, fedi_count, prof_count, both


def build_result_row(row, pattern):
    """Build a result dictionary for one cell."""
    if both_abstained(row, pattern) or one_abstained(row, pattern):
        val_agree = None
    else:
        val_agree = value_agreement(row, pattern)
    return {
        "query_id": row["query_id"],
        "pattern": pattern,
        "full_match": cells_match(row, pattern),
        "value_agreement": val_agree,
        "fedi_value": row[f"value_{pattern}_{SUFFIX_FEDI}"],
        "fedi_unit": row[f"unit_{pattern}_{SUFFIX_FEDI}"],
        "fedi_abstain": row[f"abstain_{pattern}_{SUFFIX_FEDI}"],
        "fedi_reason": row[f"abstain_reason_{pattern}_{SUFFIX_FEDI}"],
        "prof_alexandre_value": row[f"value_{pattern}_{SUFFIX_PROF}"],
        "prof_alexandre_unit": row[f"unit_{pattern}_{SUFFIX_PROF}"],
        "prof_alexandre_abstain": row[f"abstain_{pattern}_{SUFFIX_PROF}"],
        "prof_alexandre_reason": row[f"abstain_reason_{pattern}_{SUFFIX_PROF}"],
    }


def main():
    """Compute agreement metrics between Fedi and Prof. Alexandre labels."""
    df_fedi = pd.read_csv(FEDI_FILE)
    df_prof = pd.read_excel(PROF_ALEXANDRE_FILE, sheet_name=PROF_ALEXANDRE_SHEET)

    merged = df_fedi.merge(
        df_prof, on="query_id", how="inner", validate="one_to_one",
        suffixes=(f"_{SUFFIX_FEDI}", f"_{SUFFIX_PROF}")
    )
    assert len(merged) == 30, f"Expected 30 query_ids, got {len(merged)}"

    results = []
    abstain_labels_fedi = []
    abstain_labels_prof = []

    for _, row in merged.iterrows():
        for p in PATTERNS:
            results.append(build_result_row(row, p))
            abstain_labels_fedi.append(int(get_abstain(row, p, SUFFIX_FEDI)))
            abstain_labels_prof.append(int(get_abstain(row, p, SUFFIX_PROF)))

    df_res = pd.DataFrame(results)

    n_val, n_val_agree, pct_val, per_pattern = compute_value_agreement_metrics(df_res)
    kappa, fedi_abst, prof_abst, both_abst = compute_kappa_metrics(abstain_labels_fedi, abstain_labels_prof)

    n_total = len(df_res)
    n_full_agree = int(df_res["full_match"].sum())
    pct_full = round(n_full_agree / n_total * 100, 2)

    disagreements = df_res[~df_res["full_match"]].copy()
    disagreements["diff_type"] = disagreements.apply(
        lambda r: "abstain_mismatch" if r["fedi_abstain"] != r["prof_alexandre_abstain"]
        else ("unit" if not units_match(r["fedi_unit"], r["prof_alexandre_unit"])
        else "value"), axis=1
    )

    val_disagreements = df_res[df_res["value_agreement"] == False].copy()

    out = {
        "metric_1_value_agreement": {
            "description": "% Agreement on extracted values (Doc 07 tolerance, non-abstained cells only)",
            "n_value_cells": n_val,
            "n_agree": n_val_agree,
            "percent_agreement": pct_val,
            "per_pattern": per_pattern,
        },
        "metric_2_kappa_abstention": {
            "description": "Cohen's Kappa on abstention decision (binary)",
            "kappa": kappa,
            "n_cells": n_total,
            "fedi_abstentions": fedi_abst,
            "prof_abstentions": prof_abst,
            "both_abstained": both_abst,
        },
        "metric_3_full_agreement_reference": {
            "description": "Full agreement (value + unit + abstain + reason) — for reference only",
            "n_total": n_total,
            "n_agree": n_full_agree,
            "percent": pct_full,
        },
        "value_disagreements": val_disagreements[["query_id", "pattern", "fedi_value", "fedi_unit",
                                                     "prof_alexandre_value", "prof_alexandre_unit"]].to_dict(orient="records"),
        "full_disagreements": disagreements[["query_id", "pattern", "diff_type",
                                              "fedi_value", "fedi_unit", "fedi_abstain", "fedi_reason",
                                              "prof_alexandre_value", "prof_alexandre_unit", "prof_alexandre_abstain", "prof_alexandre_reason"]].to_dict(orient="records"),
    }

    print(json.dumps(out, indent=2))

    df_res.to_csv("data/finqa_labeling/agreement_details.csv", index=False)
    print("\nDetails saved to data/finqa_labeling/agreement_details.csv")

    print(f"\n{'=' * 60}")
    print("FINQA LABELING AGREEMENT (Doc 10 + Doc 07)")
    print("=" * 60)
    print("\n[1] % AGREEMENT ON VALUES (Doc 07 tolerance)")
    print(f"    Non-abstained cells:    {n_val}")
    print(f"    Agree:                  {n_val_agree} ({pct_val:.1f}%)")
    for p, s in per_pattern.items():
        line = f"      Pattern {p}: {s['agree']}/{s['n_values']} = {s['pct']:.1f}%"
        print(line if s["n_values"] > 0 else f"      Pattern {p}: N/A")
    print("\n[2] COHEN'S KAPPA ON ABSTENTION DECISION (binary)")
    print(f"    Kappa:                  {kappa:.4f}")
    print(f"    Fedi abstentions:       {fedi_abst}")
    print(f"    Prof. Alexandre:        {prof_abst}")
    print(f"    Both abstained:         {both_abst}")
    print("\n[3] FULL AGREEMENT (reference)")
    print(f"    {n_full_agree}/{n_total} ({pct_full:.1f}%)")
    print(f"\nValue disagreements: {len(val_disagreements)}")
    print(f"Full disagreements:  {len(disagreements)}")


if __name__ == "__main__":
    main()
