"""SHAP explanations for the blender: which of {stylometric, statistical,
encoder} signals actually drove a given prediction. This is the piece that lets
you answer "why did it flag this text" rather than just "it flagged this text",
which matters a lot if the output ever gates a real decision (e.g. a
moderation queue)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from forensics.models.blender import load_blender_models


def explain_predictions(X: pd.DataFrame, fold_index: int = 0) -> shap.Explanation:
    models = load_blender_models()
    explainer = shap.TreeExplainer(models[fold_index])
    return explainer(X)


def top_feature_contributions(explanation: shap.Explanation, row_idx: int, k: int = 8) -> pd.DataFrame:
    values = explanation.values[row_idx]
    feature_names = explanation.feature_names
    order = np.argsort(-np.abs(values))[:k]
    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "shap_value": [values[i] for i in order],
            "feature_value": [explanation.data[row_idx][i] for i in order],
        }
    )


def global_feature_importance(explanation: shap.Explanation) -> pd.DataFrame:
    mean_abs = np.abs(explanation.values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return pd.DataFrame(
        {
            "feature": [explanation.feature_names[i] for i in order],
            "mean_abs_shap": mean_abs[order],
        }
    )
