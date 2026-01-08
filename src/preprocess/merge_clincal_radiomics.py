"""
Merge clinical CSV with radiomics CSV (REST + STRESS).

Radiomics CSV (long format):
  patient_id, condition (REST/STRESS), rad1, rad2, ...

Clinical CSV:
  one row per patient_id
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--radiomics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="patient_id")
    parser.add_argument("--condition-col", default="condition")
    args = parser.parse_args()

    # Load
    clinical = pd.read_csv(args.clinical)
    radiomics = pd.read_csv(args.radiomics)

    id_col = args.id_col
    cond_col = args.condition_col

    # Sanity checks
    assert id_col in clinical.columns, "patient_id missing in clinical CSV"
    assert id_col in radiomics.columns, "patient_id missing in radiomics CSV"
    assert cond_col in radiomics.columns, "condition missing in radiomics CSV"

    print(f"Clinical patients:  {clinical[id_col].nunique()}")
    print(f"Radiomics patients: {radiomics[id_col].nunique()}")

    # Normalize condition values (safety)
    radiomics[cond_col] = radiomics[cond_col].str.upper().str.strip()

    # Radiomics features
    rad_features = [c for c in radiomics.columns if c not in [id_col, cond_col]]

    # Pivot radiomics: long -> wide
    rad_wide = (
        radiomics
        .set_index([id_col, cond_col])[rad_features]
        .unstack(cond_col)
    )

    # Flatten columns: feature_CONDITION
    rad_wide.columns = [
        f"{feat}_{cond}"
        for feat, cond in rad_wide.columns
    ]

    rad_wide = rad_wide.reset_index()

    # Merge with clinical
    merged = clinical.merge(rad_wide, on=id_col, how="inner")

    # Final checks
    print(f"Merged patients:   {merged[id_col].nunique()}")
    print(f"Final shape:       {merged.shape}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"✅ Saved merged dataset to: {out_path}")


if __name__ == "__main__":
    main()
