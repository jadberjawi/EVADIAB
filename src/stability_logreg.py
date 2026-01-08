"""
Stability analysis: Logistic Regression over repeated stratified train/test splits.

Outputs (single folder):
  outputs/stability_logreg/
    - metrics_50splits.csv
    - summary.json
    - roc_auc_hist.png
    - pr_auc_hist.png
    - brier_hist.png

Run:
  python scripts/stability_logreg_50splits.py --config configs/config.yaml --repeats 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data_loader import load_clean_csv
from src.model import build_logreg_pipeline
from src.utils import compute_binary_metrics

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


def mean_ci95(x: np.ndarray):
    """Mean and approximate 95% CI using normal approximation."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    return {"n": int(n), "mean": mean, "std": std, "ci95_low": float(ci_low), "ci95_high": float(ci_high)}


def save_hist(values: np.ndarray, title: str, xlabel: str, outpath: Path):
    plt.figure()
    plt.hist(values, bins=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--outdir", default="outputs/stability_logreg")
    ap.add_argument("--wandb", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Force logistic regression regardless of what's in config (stability run is for LogReg baseline)
    cont = cfg["features"]["continuous"]
    cat = cfg["features"]["categorical"]

    df = load_clean_csv(cfg["data"]["csv_path"])

    # Drop cols if specified (you said id is already dropped; this keeps it robust)
    for c in cfg["data"].get("drop_cols", []):
        if c in df.columns:
            df = df.drop(columns=[c])

    label_col = cfg["data"]["label_col"]
    y_all = df[label_col].astype(int).to_numpy()
    X_all = df.drop(columns=[label_col])

    test_size = float(cfg["split"]["test_size"])
    stratify = bool(cfg["split"].get("stratify", True))
    strat = y_all if stratify else None

    base_seed = int(cfg["project"]["seed"])

    # W&B: single run (clean)
    wb_cfg = cfg.get("wandb", {})
    wandb_mode = args.wandb
    use_wandb = False
    if wandb_mode == "on":
        use_wandb = True
    elif wandb_mode == "off":
        use_wandb = False
    else:  # auto
        use_wandb = bool(wb_cfg.get("enabled", True)) and _WANDB_AVAILABLE

    if use_wandb:
        wandb.init(
            project=cfg["project"]["name"],
            name=f"{cfg['project'].get('run_name','evadiab')}_logreg_stability_{args.repeats}splits",
            config={
                "repeats": args.repeats,
                "base_seed": base_seed,
                "test_size": test_size,
                "stratify": stratify,
                "features": {"continuous": cont, "categorical": cat},
                "model": {"type": "logistic_regression", "params": cfg["model"]["params"]},
                "data": {"csv_path": cfg["data"]["csv_path"], "n": int(len(df)), "event_rate": float(y_all.mean())},
            },
        )

    rows = []
    for i in range(args.repeats):
        seed = base_seed + i

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=test_size,
            random_state=seed,
            stratify=strat,
        )

        pipe = build_logreg_pipeline(
            continuous_cols=cont,
            categorical_cols=cat,
            params=cfg["model"]["params"],
        )
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        m = compute_binary_metrics(y_test, y_prob, threshold=0.5)
        rows.append({
            "repeat": i,
            "seed": seed,
            "n_test": int(len(y_test)),
            "event_rate_test": float(np.mean(y_test)),
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "brier": m["brier"],
            "acc": m["acc"],
        })

    res = pd.DataFrame(rows)
    res_path = outdir / f"metrics_{args.repeats}splits.csv"
    res.to_csv(res_path, index=False)

    # Summary stats
    summary = {
        "n_total": int(len(df)),
        "event_rate_total": float(y_all.mean()),
        "repeats": int(args.repeats),
        "test_size": test_size,
        "metrics": {
            "roc_auc": mean_ci95(res["roc_auc"].to_numpy()),
            "pr_auc": mean_ci95(res["pr_auc"].to_numpy()),
            "brier": mean_ci95(res["brier"].to_numpy()),
            "acc": mean_ci95(res["acc"].to_numpy()),
        }
    }
    summary_path = outdir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    save_hist(res["roc_auc"].to_numpy(), "ROC-AUC across 50 splits", "ROC-AUC", outdir / "roc_auc_hist.png")
    save_hist(res["pr_auc"].to_numpy(), "PR-AUC across 50 splits", "PR-AUC", outdir / "pr_auc_hist.png")
    save_hist(res["brier"].to_numpy(), "Brier score across 50 splits", "Brier", outdir / "brier_hist.png")

    # W&B logging (single run)
    if use_wandb:
        wandb.log({
            "stability/roc_auc_mean": summary["metrics"]["roc_auc"]["mean"],
            "stability/roc_auc_std": summary["metrics"]["roc_auc"]["std"],
            "stability/pr_auc_mean": summary["metrics"]["pr_auc"]["mean"],
            "stability/pr_auc_std": summary["metrics"]["pr_auc"]["std"],
            "stability/brier_mean": summary["metrics"]["brier"]["mean"],
            "stability/brier_std": summary["metrics"]["brier"]["std"],
            "stability/acc_mean": summary["metrics"]["acc"]["mean"],
            "stability/acc_std": summary["metrics"]["acc"]["std"],
        })

        # Log the full table as a W&B Table (still one run, not 50 runs)
        wandb.log({"stability/results_table": wandb.Table(dataframe=res)})

        # Upload plots + csv + summary
        wandb.save(str(res_path))
        wandb.save(str(summary_path))
        wandb.save(str(outdir / "roc_auc_hist.png"))
        wandb.save(str(outdir / "pr_auc_hist.png"))
        wandb.save(str(outdir / "brier_hist.png"))
        wandb.finish()

    print("✅ Stability analysis complete.")
    print(f"📄 Results CSV:   {res_path}")
    print(f"🧾 Summary JSON:  {summary_path}")
    print(f"🖼️  Plots in:     {outdir}")


if __name__ == "__main__":
    main()
