"""Single shared inference path used by both the FastAPI service and the Gradio
demo, so "predict for one piece of text" has exactly one implementation instead
of being re-derived in two places.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from forensics.features.statistical import statistical_features
from forensics.features.stylometric import stylometric_features
from forensics.models.blender import load_blender_models, load_calibrator, make_feature_matrix, predict_blender_ensemble
from forensics.models.encoder import load_fold_models, predict_ensemble


class Predictor:
    """Loads every trained artifact once and reuses it across requests."""

    def __init__(self):
        self.encoder_models = load_fold_models()
        self.blender_models = load_blender_models()
        self.calibrator = load_calibrator()

    def predict(self, text: str) -> dict:
        stylo = stylometric_features(text)
        stat = statistical_features(text)
        encoder_logit = predict_ensemble([text], models=self.encoder_models)[0]

        feat_row = {**stylo, **stat}
        X = make_feature_matrix(pd.DataFrame([feat_row]), np.array([encoder_logit]))
        raw_prob = predict_blender_ensemble(self.blender_models, X)[0]
        calibrated_prob = float(self.calibrator.predict([raw_prob])[0])

        return {
            "probability_machine_generated": calibrated_prob,
            "label": "machine-generated" if calibrated_prob >= 0.5 else "human-written",
            "detectors": {
                "encoder_prob": float(1 / (1 + np.exp(-encoder_logit))),
                "stat_loglik": stat["stat_loglik"],
                "stat_logrank": stat["stat_logrank"],
                "stat_lrr": stat["stat_lrr"],
                "stat_curvature": stat["stat_curvature"],
                "stylometric_burstiness": stylo["burstiness"],
                "stylometric_rep3_rate": stylo["rep3_rate"],
            },
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
