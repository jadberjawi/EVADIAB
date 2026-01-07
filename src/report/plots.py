"""
Plot ROC, PR, and Calibration curves for the trained Logistic Regression pipeline.

Assumptions:
- You trained and saved: models/logreg_pipeline.joblib
- Your cleaned CSV exists: data/processed/evadiab_clinical_clean.csv
- You use the same split settings as training (seed/test_size/stratify)

Run:
  python scripts/plot_logreg_curves.py --config configs/config.yaml
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model-path", default="models/logreg_pipeline.joblib")
    ap.add_argument("--outdir", default="outputs/plots")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(cfg["data"]["csv_path"])
    label_col = cfg["data"]["label_col"]

    # Drop cols if any (you said id is already dropped; still keep config compatibility)
    for c in cfg["data"].get("drop_cols", []):
        if c in df.columns:
            df = df.drop(columns=[c])

    y = df[label_col].astype(int).to_numpy()
    X = df.drop(columns=[label_col])

    # Recreate the same split as training
    seed = int(cfg["project"]["seed"])
    test_size = float(cfg["split"]["test_size"])
    stratify = bool(cfg["split"].get("stratify", True))
    strat = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )

    # Load trained pipeline and predict
    pipe = joblib.load(args.model_path)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Metrics for reference
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    event_rate = float(np.mean(y_test))

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "event_rate_test": event_rate,
        "n_test": int(len(y_test)),
    }
    with open(outdir / "logreg_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -------------------
    # 1) ROC curve
    # -------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"LogReg (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=200)
    plt.close()

    # -------------------
    # 2) Precision-Recall curve
    # -------------------
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision, label=f"LogReg (AP={pr_auc:.3f})")
    plt.hlines(event_rate, 0, 1, linestyles="--", label=f"Baseline={event_rate:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=200)
    plt.close()

    # -------------------
    # 3) Calibration curve (reliability diagram)
    # -------------------
    # Bin predicted probabilities and compare predicted vs observed frequency
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_pred = []
    bin_obs = []
    bin_count = []

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_count.append(int(mask.sum()))
        bin_pred.append(float(np.mean(y_prob[mask])))
        bin_obs.append(float(np.mean(y_test[mask])))

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(bin_pred, bin_obs, marker="o", label="LogReg")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"Calibration Curve (Test Set) — Brier={brier:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "calibration_curve.png", dpi=200)
    plt.close()

    print("✅ Saved plots to:", outdir)
    print("✅ Metrics:", metrics)


if __name__ == "__main__":
    main()
