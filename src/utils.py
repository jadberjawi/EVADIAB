import os
import random
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Some metrics can fail if only one class exists; handle safely
    metrics = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["pr_auc"] = None

    metrics["brier"] = float(brier_score_loss(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(int)
    metrics["acc"] = float(accuracy_score(y_true, y_pred))

    metrics["event_rate"] = float(y_true.mean())
    return metrics