"""End-to-end orchestration, written so every expensive stage is disk-cached and
re-running after an interruption just resumes instead of recomputing."""
from __future__ import annotations

import numpy as np

from forensics.config import ARTIFACTS_DIR, set_seed
from forensics.data.splits import build_splits
from forensics.features.cache import build_features
from forensics.models.blender import fit_calibrator, make_feature_matrix, train_blender_cv
from forensics.models.encoder import predict_ensemble, run_encoder_cv

LOGITS_DIR = ARTIFACTS_DIR / "encoder_logits"
LOGITS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_SPLITS = ["test_in_distribution", "test_cross_domain", "test_cross_generator", "test_gpt4_extension"]


def prepare_features_and_encoder(force: bool = False) -> dict:
    """Stage 1: data -> features -> encoder CV -> encoder logits on every eval
    split. Returns the splits dict plus a dict of {split_name: feature_df} and
    logit arrays for downstream blending."""
    set_seed()
    splits = build_splits()

    print("\n=== Building stylometric + statistical features ===")
    feats = {name: build_features(name, df, force=force) for name, df in splits.items()}

    oof_path = LOGITS_DIR / "oof.npy"
    if oof_path.exists() and not force:
        oof_logits = np.load(oof_path)
    else:
        print("\n=== Training DeBERTa-v3 + LoRA encoder (5-fold CV) ===")
        oof_logits = run_encoder_cv(splits["train"])
        np.save(oof_path, oof_logits)

    eval_logits = {}
    for name in EVAL_SPLITS:
        path = LOGITS_DIR / f"{name}.npy"
        if path.exists() and not force:
            eval_logits[name] = np.load(path)
            continue
        print(f"\n=== Encoder ensemble inference: {name} ===")
        logits = predict_ensemble(splits[name]["text"].tolist())
        np.save(path, logits)
        eval_logits[name] = logits

    return {"splits": splits, "features": feats, "oof_logits": oof_logits, "eval_logits": eval_logits}


def train_blender_stage(state: dict) -> dict:
    """Stage 2: stack encoder OOF logits + stylometric/statistical features into
    the LightGBM blender, then calibrate its output probabilities."""
    print("\n=== Training meta-learner blender ===")
    train_df = state["splits"]["train"]
    X_train = make_feature_matrix(state["features"]["train"], state["oof_logits"])
    y_train = train_df["is_machine"].values

    oof_probs, blender_models = train_blender_cv(X_train, y_train)
    calibrator = fit_calibrator(oof_probs, y_train)

    state["blender_oof_probs"] = oof_probs
    state["blender_models"] = blender_models
    state["calibrator"] = calibrator
    return state


def train_all() -> dict:
    state = prepare_features_and_encoder()
    state = train_blender_stage(state)
    return state


if __name__ == "__main__":
    train_all()
