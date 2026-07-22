from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_report_dict(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    out = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, zero_division=0),
    }
    # ROC/PR-AUC are undefined with a single class present (e.g. an all-machine
    # attack slice) -- report NaN rather than crash the eval loop.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, probs)
        out["pr_auc"] = average_precision_score(y_true, probs)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out
