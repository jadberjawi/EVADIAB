#!/usr/bin/env python3
"""
Compare stability metrics across 3 models (clinical-only, radiomics-only, combined).

Inputs: three CSV files produced by stability script, each with columns:
  roc_auc, pr_auc, brier, acc, repeat, seed, ...

Outputs:
  outputs/compare_models/
    - summary_table.csv
    - boxplot_roc_auc.png
    - boxplot_pr_auc.png
    - boxplot_brier.png
    - hist_roc_auc.png
    - hist_pr_auc.png
    - hist_brier.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_ci95(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    return mean, std, float(mean - 1.96*se), float(mean + 1.96*se)

def boxplot(metric, data_dict, outpath, ylabel):
    plt.figure()
    labels = list(data_dict.keys())
    vals = [data_dict[k][metric].values for k in labels]
    plt.boxplot(vals, labels=labels, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} across 50 splits")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def hist_samebins(metric, data_dict, outpath, xlabel, bins=12):
    all_vals = np.concatenate([data_dict[k][metric].values for k in data_dict.keys()])
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    plt.figure()
    for k in data_dict.keys():
        plt.hist(data_dict[k][metric].values, bins=bin_edges, alpha=0.5, label=k)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"{xlabel} distribution (same bins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--radiomics_csv", required=True)
    ap.add_argument("--combined_csv", required=True)
    ap.add_argument("--outdir", default="outputs/compare_models")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clin = pd.read_csv(args.clinical_csv)
    rad = pd.read_csv(args.radiomics_csv)
    comb = pd.read_csv(args.combined_csv)

    data = {
        "clinical": clin,
        "radiomics": rad,
        "combined": comb
    }

    # Summary table
    rows = []
    for name, df in data.items():
        for metric in ["roc_auc", "pr_auc", "brier", "acc"]:
            mean, std, lo, hi = mean_ci95(df[metric].values)
            rows.append({
                "model": name,
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": int(df.shape[0])
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary_table.csv", index=False)

    # Boxplots
    boxplot("roc_auc", data, outdir / "boxplot_roc_auc.png", "ROC-AUC")
    boxplot("pr_auc",  data, outdir / "boxplot_pr_auc.png",  "PR-AUC")
    boxplot("brier",   data, outdir / "boxplot_brier.png",   "Brier score")

    # Histograms (same bins)
    hist_samebins("roc_auc", data, outdir / "hist_roc_auc.png", "ROC-AUC")
    hist_samebins("pr_auc",  data, outdir / "hist_pr_auc.png",  "PR-AUC")
    hist_samebins("brier",   data, outdir / "hist_brier.png",   "Brier score")

    print("✅ Saved comparison plots + summary to:", outdir)

if __name__ == "__main__":
    main()
