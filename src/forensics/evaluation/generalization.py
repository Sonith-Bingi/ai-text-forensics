"""Runs the trained pipeline (encoder ensemble + blender + calibrator) across
every evaluation slice and produces one comparable results table. This is the
core "does it actually generalize" deliverable: in-distribution accuracy alone
would hide whether the model just memorized MAGE's specific domains/generators.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from forensics.config import ARTIFACTS_DIR
from forensics.evaluation.calibration import expected_calibration_error, plot_reliability_diagram
from forensics.evaluation.metrics import classification_report_dict
from forensics.models.blender import load_blender_models, load_calibrator, make_feature_matrix, predict_blender_ensemble

RESULTS_PATH = ARTIFACTS_DIR / "results" / "generalization_results.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

CORE_EVAL_SPLITS = ["test_in_distribution", "test_cross_domain", "test_cross_generator"]


def score_split(split_df: pd.DataFrame, feat_df: pd.DataFrame, encoder_logits: np.ndarray) -> np.ndarray:
    """Returns calibrated P(machine-generated) for every row in split_df."""
    X = make_feature_matrix(feat_df, encoder_logits)
    models = load_blender_models()
    raw_probs = predict_blender_ensemble(models, X)
    calibrator = load_calibrator()
    return calibrator.predict(raw_probs)


def run_generalization_suite(splits: dict, feats: dict, eval_logits: dict) -> dict:
    results = {}
    reliability_inputs = {}

    for name in CORE_EVAL_SPLITS:
        split_df = splits[name]
        y_true = split_df["is_machine"].values
        probs = score_split(split_df, feats[name], eval_logits[name])
        report = classification_report_dict(y_true, probs)
        report["ece"] = expected_calibration_error(probs, y_true)
        results[name] = report
        reliability_inputs[name] = (probs, y_true)
        print(f"[{name}] acc={report['accuracy']:.4f} f1={report['f1']:.4f} "
              f"roc_auc={report['roc_auc']:.4f} ece={report['ece']:.4f}")

    # GPT-4 extension: report the "direct" (non-paraphrased) rows as a 4th
    # generalization slice -- unseen domain AND unseen generator simultaneously.
    gpt4_df = splits["test_gpt4_extension"]
    gpt4_feats = feats["test_gpt4_extension"]
    gpt4_logits = eval_logits["test_gpt4_extension"]
    direct_mask = gpt4_df["style"].isin(["human", "direct"]).values
    direct_df = gpt4_df[direct_mask].reset_index(drop=True)
    direct_feats = gpt4_feats[direct_mask].reset_index(drop=True)
    direct_logits = gpt4_logits[direct_mask]
    probs = score_split(direct_df, direct_feats, direct_logits)
    y_true = direct_df["is_machine"].values
    report = classification_report_dict(y_true, probs)
    report["ece"] = expected_calibration_error(probs, y_true)
    results["test_gpt4_unseen_domain_and_generator"] = report
    reliability_inputs["gpt4_unseen"] = (probs, y_true)
    print(f"[gpt4_unseen_domain_and_generator] acc={report['accuracy']:.4f} "
          f"f1={report['f1']:.4f} roc_auc={report['roc_auc']:.4f} ece={report['ece']:.4f}")

    plot_reliability_diagram(reliability_inputs, str(ARTIFACTS_DIR / "results" / "reliability_diagram.png"))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    from forensics.pipeline import prepare_features_and_encoder

    state = prepare_features_and_encoder()
    run_generalization_suite(state["splits"], state["features"], state["eval_logits"])
