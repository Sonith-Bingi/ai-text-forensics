"""Meta-learner: stacks the LoRA-encoder logit together with the stylometric and
statistical zero-shot detector features into a single LightGBM classifier, then
calibrates its output probabilities.

Why stack rather than just use the encoder alone: the encoder learns *this
training distribution's* surface patterns and can be confidently wrong outside
it (see the cross-domain/cross-generator eval). The statistical detectors
(log-likelihood/log-rank/curvature) are training-free and driven by a
completely different mechanism (how surprising the text is to a small LM), so
their errors are largely uncorrelated with the encoder's -- a blend is more
robust than either alone, which is exactly what the generalization eval in
evaluation/generalization.py is designed to check.

Calibration matters separately from accuracy: a downstream user of this system
(e.g. a moderation queue) needs "70% confidence" to actually mean 70% empirical
frequency, not just a score that happens to rank-order correctly. We fit
isotonic regression on out-of-fold predictions (never on data the blender was
scored on) and validate it with Expected Calibration Error in evaluation/calibration.py.
"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from forensics.config import ARTIFACTS_DIR, CFG, SEED

BLENDER_DIR = ARTIFACTS_DIR / "blender"
BLENDER_DIR.mkdir(parents=True, exist_ok=True)


def make_feature_matrix(feature_df: pd.DataFrame, encoder_logits: np.ndarray) -> pd.DataFrame:
    """`feature_df` is the unified stylometric+statistical frame produced by
    features/cache.py::build_features -- this just appends the encoder signal."""
    X = feature_df.reset_index(drop=True).copy()
    X["encoder_logit"] = encoder_logits
    X["encoder_prob"] = 1 / (1 + np.exp(-encoder_logits))
    X["encoder_margin"] = np.abs(encoder_logits)
    return X


def train_blender_cv(X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, list]:
    skf = StratifiedKFold(n_splits=CFG.blender.n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx])
        model = lgb.train(
            CFG.blender.lgb_params,
            dtr,
            num_boost_round=CFG.blender.lgb_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=200)],
        )
        best_iter = model.best_iteration or CFG.blender.lgb_rounds
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=best_iter)
        models.append(model)
        fold_acc = accuracy_score(y[va_idx], (oof[va_idx] >= 0.5).astype(int))
        print(f"  [blender fold {fold}] acc={fold_acc:.4f} best_iter={best_iter}")

    oof_acc = accuracy_score(y, (oof >= 0.5).astype(int))
    print(f"Blender OOF accuracy: {oof_acc:.4f}")

    for i, model in enumerate(models):
        model.save_model(str(BLENDER_DIR / f"fold{i}.txt"))
    return oof, models


def predict_blender_ensemble(models: list, X: pd.DataFrame) -> np.ndarray:
    preds = np.stack([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
    return preds.mean(axis=0)


def fit_calibrator(oof_probs: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    calibrator.fit(oof_probs, y)
    with open(BLENDER_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    return calibrator


def load_calibrator() -> IsotonicRegression:
    with open(BLENDER_DIR / "calibrator.pkl", "rb") as f:
        return pickle.load(f)


def load_blender_models() -> list:
    models = []
    for path in sorted(BLENDER_DIR.glob("fold*.txt"), key=lambda p: int(p.stem.replace("fold", ""))):
        models.append(lgb.Booster(model_file=str(path)))
    return models


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos
