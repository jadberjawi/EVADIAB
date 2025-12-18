import argparse
from pathlib import Path

import numpy as np
import wandb
import yaml
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Reproducibility
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    # W&B init
    project = cfg.get("project", "evadiab-clinical-ml")
    run_name = cfg.get("run_name", "run")

    wandb.init(
        project=project,
        name=run_name,
        config=cfg,  # logs the full YAML config
    )

    # Dummy dataset (placeholder for EVADIAB later)
    ds = cfg.get("dataset", {})
    X, y = make_classification(
        n_samples=int(ds.get("n_samples", 200)),
        n_features=int(ds.get("n_features", 10)),
        random_state=seed,
    )

    # Model
    mcfg = cfg.get("model", {})
    if mcfg.get("type", "logistic_regression") != "logistic_regression":
        raise ValueError(f"Unsupported model.type: {mcfg.get('type')}")

    model = LogisticRegression(max_iter=int(mcfg.get("max_iter", 1000)))
    model.fit(X, y)

    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)

    wandb.log({"train_auc": float(auc)})
    print(f"Dummy run finished — AUC = {auc:.3f}")

    wandb.finish()


if __name__ == "__main__":
    main()
