import argparse
import yaml
from pathlib import Path
import joblib
import wandb

from data_loader import load_clean_csv, make_train_test_split
from model import build_model_pipeline
from utils import set_seed, compute_binary_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    # W&B
    wb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wb_cfg.get("enabled", True))
    if use_wandb:
        wandb.init(
            project=cfg["project"]["name"],
            name=cfg["project"].get("run_name"),
            config=cfg,
        )

    # Load data
    df = load_clean_csv(cfg["data"]["csv_path"])

    # Split
    split = make_train_test_split(
        df=df,
        label_col=cfg["data"]["label_col"],
        drop_cols=cfg["data"].get("drop_cols", []),
        test_size=float(cfg["split"]["test_size"]),
        seed=seed,
        stratify=bool(cfg["split"].get("stratify", True)),
    )

    cont = cfg["features"]["continuous"]
    cat = cfg["features"]["categorical"]

    # Basic logging
    if use_wandb:
        wandb.log({
            "data/n_total": int(len(df)),
            "data/n_train": int(len(split.X_train)),
            "data/n_test": int(len(split.X_test)),
            "data/event_rate_total": float(df[cfg["data"]["label_col"]].mean()),
            "data/missing_total": int(df.isna().sum().sum()),
        })

    # Build + fit model
    pipe = build_model_pipeline(
        model_type=cfg["model"]["type"],
        continuous_cols=cont,
        categorical_cols=cat,
        params=cfg["model"]["params"],
    )
    pipe.fit(split.X_train, split.y_train)

    # Evaluate
    y_prob = pipe.predict_proba(split.X_test)[:, 1]
    metrics = compute_binary_metrics(split.y_test, y_prob, threshold=0.5)

    print("✅ Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if use_wandb:
        wandb.log({f"test/{k}": v for k, v in metrics.items()})

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    model_name = cfg["model"]["type"]
    model_path = Path("models") / f"{model_name}_pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"💾 Saved model to: {model_path}")

    if use_wandb:
        wandb.save(str(model_path))
        wandb.finish()


if __name__ == "__main__":
    main()
