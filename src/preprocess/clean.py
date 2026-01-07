#!/usr/bin/env python3
"""
Clean EVADIAB clinical Excel sheet into a model-ready CSV.

What it does:
- Reads .xlsx (one sheet, or a specified sheet)
- Keeps only the selected columns:
  age, sex, creat, HBA1C, hdl, ldl, tg, Smoke, history_CAD, BMI_post_imputation, label (+ optional id)
- Drops rows where label is missing
- Encodes sex: 1=male, 2=female -> sex_male: 1/0
- Ensures Smoke is integer in {0,1,2} if possible
- Converts numeric columns to numeric; coerces bad values to NaN
- Saves cleaned CSV + a JSON report with missingness and basic stats

Usage:
  python scripts/clean_excel_to_csv.py \
    --input data/raw/evadiab.xlsx \
    --output data/processed/evadiab_clinical_clean.csv \
    --sheet 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# --- Column configuration (adapt here if your Excel uses different names) ---
RENAME_MAP = {
    # if your sheet has slightly different names, map them here:
    # "HbA1c": "HBA1C",
    # "TG": "tg",
}

# Columns we want from your screenshot
KEEP_COLS = [
    "id",  # optional, used only for tracking; not a feature
    "sex",
    "age",
    "creat",
    "HBA1C",
    "hdl",
    "ldl",
    "tg",
    "Smoke",
    "history_CAD",
    "BMI_post_imputation",
    "label",
]

NUMERIC_COLS = [
    "age",
    "creat",
    "HBA1C",
    "hdl",
    "ldl",
    "tg",
    "BMI_post_imputation",
]

INT_COLS = [
    "Smoke",
    "history_CAD",
    "label",
    "sex",
]


def _standardize_na(df: pd.DataFrame) -> pd.DataFrame:
    # Convert common NA representations to real NaN
    na_values = {"NA", "N/A", "na", "n/a", "Na", "", " ", "None", "none", "-", "--"}
    return df.replace(list(na_values), np.nan)


def clean_excel(
    input_path: Path,
    sheet,
    keep_id: bool,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_excel(input_path, sheet_name=sheet)

    # Normalize column names: strip spaces, keep case as-is, but remove trailing/leading whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Optional renaming (if needed)
    df = df.rename(columns=RENAME_MAP)

    df = _standardize_na(df)

    # Keep only columns that exist
    missing_cols = [c for c in KEEP_COLS if c not in df.columns]
    present_keep_cols = [c for c in KEEP_COLS if c in df.columns]

    if not keep_id and "id" in present_keep_cols:
        present_keep_cols.remove("id")

    df = df[present_keep_cols].copy()

    # Coerce integers where relevant (sex/smoke/history/label)
    for c in INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing label
    if "label" not in df.columns:
        raise ValueError("Column 'label' not found. Please verify column names in your Excel.")
    before = len(df)
    df = df[~df["label"].isna()].copy()
    after = len(df)

    # Encode sex -> sex_male (1=male, 2=female)
    # Your mapping: 1 male, 2 female
    if "sex" in df.columns:
        df["sex_male"] = df["sex"].map({1: 1, 2: 0})
        # If some values are not 1/2, they become NaN; keep them but report
        df = df.drop(columns=["sex"])

    # Ensure history_CAD is 0/1 if possible
    if "history_CAD" in df.columns:
        df["history_CAD"] = df["history_CAD"].where(df["history_CAD"].isin([0, 1]), np.nan)

    # Ensure Smoke is 0/1/2 if possible
    if "Smoke" in df.columns:
        df["Smoke"] = df["Smoke"].where(df["Smoke"].isin([0, 1, 2]), np.nan)

    # Coerce numeric columns properly
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure label is integer 0/1
    df["label"] = df["label"].where(df["label"].isin([0, 1]), np.nan)
    df = df[~df["label"].isna()].copy()
    df["label"] = df["label"].astype(int)

    # Build report
    report = {
        "input_file": str(input_path),
        "sheet": sheet,
        "rows_before_drop_label_na": int(before),
        "rows_after_drop_label_na": int(after),
        "rows_after_drop_invalid_label": int(len(df)),
        "columns_present": list(df.columns),
        "missing_original_columns": missing_cols,
        "missingness": {},
        "basic_stats": {},
    }

    # Missingness per column
    for c in df.columns:
        report["missingness"][c] = {
            "n_missing": int(df[c].isna().sum()),
            "pct_missing": float(df[c].isna().mean() * 100.0),
        }

    # Basic stats for numeric columns
    for c in [c for c in df.columns if c in NUMERIC_COLS or c == "sex_male"]:
        if c in df.columns:
            s = df[c]
            report["basic_stats"][c] = {
                "count": int(s.count()),
                "mean": float(s.mean()) if s.count() else None,
                "std": float(s.std()) if s.count() else None,
                "min": float(s.min()) if s.count() else None,
                "p25": float(s.quantile(0.25)) if s.count() else None,
                "median": float(s.median()) if s.count() else None,
                "p75": float(s.quantile(0.75)) if s.count() else None,
                "max": float(s.max()) if s.count() else None,
            }

    # Label distribution
    report["label_distribution"] = df["label"].value_counts(dropna=False).to_dict()

    return df, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .xlsx file")
    ap.add_argument("--output", required=True, help="Path to output cleaned .csv")
    ap.add_argument("--sheet", default=0, help="Excel sheet index or name (default: 0)")
    ap.add_argument("--keep-id", action="store_true", help="Keep 'id' column in output (default: dropped)")
    ap.add_argument("--report", default=None, help="Path to JSON report (default: output + .report.json)")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # interpret sheet: int if possible else string
    sheet = args.sheet
    try:
        sheet = int(sheet)
    except ValueError:
        pass

    df_clean, report = clean_excel(input_path=input_path, sheet=sheet, keep_id=args.keep_id)

    # Save CSV
    df_clean.to_csv(output_path, index=False)

    # Save report
    report_path = Path(args.report).expanduser().resolve() if args.report else output_path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Clean CSV saved to: {output_path}")
    print(f"🧾 Report saved to:    {report_path}")
    print(f"📌 Final shape:        {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
    print(f"📌 Columns:            {list(df_clean.columns)}")


if __name__ == "__main__":
    main()
