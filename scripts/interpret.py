#!/usr/bin/env python
"""Global SHAP feature importance for the blender, computed on the
in-distribution test set. Saves a bar chart + CSV of mean |SHAP value| per
feature, showing whether the encoder, statistical detectors, or stylometric
features actually drive predictions."""
import matplotlib.pyplot as plt

from forensics.config import ARTIFACTS_DIR
from forensics.interpret.shap_explain import explain_predictions, global_feature_importance
from forensics.models.blender import make_feature_matrix
from forensics.pipeline import prepare_features_and_encoder

RESULTS_DIR = ARTIFACTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    state = prepare_features_and_encoder()
    split = "test_in_distribution"
    X = make_feature_matrix(state["features"][split], state["eval_logits"][split])

    explanation = explain_predictions(X)
    importance = global_feature_importance(explanation)
    importance.to_csv(RESULTS_DIR / "shap_global_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    top = importance.head(15).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Blender feature importance (in-distribution test set)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_global_importance.png", dpi=150)
    print(importance.head(15))
