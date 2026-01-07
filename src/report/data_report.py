#!/usr/bin/env python3
"""
Clinical EDA for EVADIAB (pre-modeling).

Generates:
- cohort summary table
- missingness plot
- distributions stratified by outcome
- boxplots by outcome
- correlation heatmap
- categorical vs outcome plots

Outputs saved to: outputs/eda/
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Config
# -----------------------
DATA_PATH = "data/processed/evadiab_clinical_clean.csv"
OUTDIR = Path("outputs/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

CONTINUOUS_COLS = [
    "age", "creat", "HBA1C", "hdl", "ldl", "tg", "BMI_post_imputation"
]

CATEGORICAL_COLS = [
    "sex_male", "Smoke", "history_CAD"
]

LABEL_COL = "label"

sns.set(style="whitegrid")

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(DATA_PATH)

# -----------------------
# 1) Cohort summary table
# -----------------------
summary = []

for col in CONTINUOUS_COLS:
    for lbl in [0, 1]:
        vals = df.loc[df[LABEL_COL] == lbl, col].dropna()
        summary.append({
            "variable": col,
            "label": lbl,
            "n": len(vals),
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTDIR / "cohort_summary_by_label.csv", index=False)

# -----------------------
# 2) Missingness plot
# -----------------------
missing_pct = df.isna().mean().sort_values(ascending=False) * 100

plt.figure()
missing_pct.plot(kind="bar")
plt.ylabel("% missing")
plt.title("Missingness per feature")
plt.tight_layout()
plt.savefig(OUTDIR / "missingness.png", dpi=150)
plt.close()

# -----------------------
# 3) Distributions by outcome
# -----------------------
for col in CONTINUOUS_COLS:
    plt.figure()
    sns.kdeplot(data=df, x=col, hue=LABEL_COL, common_norm=False)
    plt.title(f"{col} distribution by outcome")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{col}_kde_by_label.png", dpi=150)
    plt.close()

# -----------------------
# 4) Boxplots by outcome
# -----------------------
for col in CONTINUOUS_COLS:
    plt.figure()
    sns.boxplot(data=df, x=LABEL_COL, y=col)
    plt.title(f"{col} by outcome")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{col}_boxplot_by_label.png", dpi=150)
    plt.close()

# -----------------------
# 5) Correlation heatmap
# -----------------------
corr = df[CONTINUOUS_COLS + CATEGORICAL_COLS].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Feature correlation heatmap")
plt.tight_layout()
plt.savefig(OUTDIR / "correlation_heatmap.png", dpi=150)
plt.close()

# -----------------------
# 6) Categorical vs outcome
# -----------------------
for col in CATEGORICAL_COLS:
    ct = pd.crosstab(df[col], df[LABEL_COL], normalize="index")
    ct.plot(kind="bar", stacked=True)
    plt.ylabel("Proportion")
    plt.title(f"{col} vs outcome")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{col}_vs_outcome.png", dpi=150)
    plt.close()

print("✅ EDA completed. Results saved to outputs/eda/")
