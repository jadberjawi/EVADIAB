import wandb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

def main():
    # Start a W&B run
    wandb.init(
        project="evadiab-clinical-ml",
        name="dummy-logreg-test",
        config={
            "model": "logistic_regression",
            "n_samples": 200,
            "n_features": 10,
            "random_state": 42,
        }
    )

    cfg = wandb.config

    # Fake clinical dataset
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        random_state=cfg.random_state
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)

    # Log metric
    wandb.log({"train_auc": auc})

    print(f"Dummy run finished — AUC = {auc:.3f}")

    wandb.finish()

if __name__ == "__main__":
    main()
