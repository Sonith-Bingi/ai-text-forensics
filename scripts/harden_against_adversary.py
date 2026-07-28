#!/usr/bin/env python
"""Closes the loop between the two things trained in this project: uses the
RL-trained adversarial paraphraser to generate evasive rewrites of known
machine-generated text, adds them to the detector's training data as
additional machine-class examples, and retrains the blender on the augmented
set. Then re-evaluates against both the static and adversarial paraphrasers
to check whether the hardened detector actually catches more of what it
previously missed.

Requires: a trained detector (scripts/train.py) and a trained adversarial
paraphraser (scripts/train_adversarial via run_adversarial_training.sh) to
already exist on disk.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from forensics.adversarial.paraphraser import get_tokenizer, load_policy_model, make_prompt
from forensics.adversarial.train_reinforce import CKPT_DIR as ADV_CKPT_DIR
from forensics.config import ARTIFACTS_DIR, CFG, DEVICE, PROCESSED_DIR
from forensics.evaluation.generalization import score_split
from forensics.evaluation.metrics import classification_report_dict
from forensics.features.cache import build_features
from forensics.models.blender import fit_calibrator, make_feature_matrix, train_blender_cv
from forensics.models.encoder import predict_ensemble

N_AUGMENT = 1000
RESULTS_PATH = ARTIFACTS_DIR / "results" / "hardening_report.json"


@torch.no_grad()
def _generate_adversarial_paraphrases(texts: list[str], batch_size: int = 8) -> list[str]:
    model = load_policy_model(ADV_CKPT_DIR, is_trainable=False)
    tok = get_tokenizer()
    cfg = CFG.adversarial
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prompts = [make_prompt(t) for t in batch]
        enc = tok(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_input_len
        ).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=True, top_p=0.9, temperature=1.0)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  generated {len(out)}/{len(texts)}", flush=True)
    return out


def main():
    print("=== Step 1: generate adversarial paraphrases of known machine text ===", flush=True)
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    machine_texts = train_df[train_df["is_machine"] == 1]["text"].tolist()[:N_AUGMENT]
    machine_texts = [t[:600] for t in machine_texts]
    adversarial_texts = _generate_adversarial_paraphrases(machine_texts)
    adversarial_texts = [t for t in adversarial_texts if t.strip()]  # drop any empty generations

    print(f"\n=== Step 2: featurizing {len(adversarial_texts)} augmented rows ===", flush=True)
    aug_df = pd.DataFrame({"text": adversarial_texts, "is_machine": 1})
    aug_feats = build_features("hardening_augment", aug_df, force=True)
    aug_encoder_logits = predict_ensemble(adversarial_texts)

    print("\n=== Step 3: rebuilding training feature matrix with augmented rows ===", flush=True)
    train_feats = pd.read_parquet(ARTIFACTS_DIR / "features" / "train.parquet")
    oof_logits = np.load(ARTIFACTS_DIR / "encoder_logits" / "oof.npy")
    y_train = train_df["is_machine"].values

    X_train = make_feature_matrix(train_feats, oof_logits)
    X_aug = make_feature_matrix(aug_feats, aug_encoder_logits)
    X_combined = pd.concat([X_train, X_aug], ignore_index=True)
    y_combined = np.concatenate([y_train, np.ones(len(X_aug), dtype=int)])

    print(f"Original training rows: {len(X_train)}, augmented rows: {len(X_aug)}, "
          f"combined: {len(X_combined)}", flush=True)

    print("\n=== Step 4: retraining blender on augmented data ===", flush=True)
    oof_probs, blender_models = train_blender_cv(X_combined, y_combined)
    fit_calibrator(oof_probs, y_combined)

    print("\n=== Step 5: re-evaluating hardened detector ===", flush=True)
    splits = {
        name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
        for name in ["test_in_distribution", "test_cross_domain", "test_cross_generator", "test_gpt4_extension"]
    }
    feats = {name: pd.read_parquet(ARTIFACTS_DIR / "features" / f"{name}.parquet") for name in splits}
    eval_logits = {name: np.load(ARTIFACTS_DIR / "encoder_logits" / f"{name}.npy") for name in splits}

    report = {}
    for name, df in splits.items():
        probs = score_split(df, feats[name], eval_logits[name])
        r = classification_report_dict(df["is_machine"].values, probs)
        report[name] = r
        print(f"[hardened][{name}] acc={r['accuracy']:.4f} f1={r['f1']:.4f}", flush=True)

    # The metric that actually matters for this exercise: does the hardened
    # detector catch the SAME adversarial paraphraser's output better now?
    test_id_df = splits["test_in_distribution"]
    machine_mask = test_id_df["is_machine"].values == 1
    holdout_machine_texts = test_id_df[machine_mask]["text"].tolist()[:150]
    holdout_machine_texts = [t[:600] for t in holdout_machine_texts]
    adv_holdout_paraphrases = _generate_adversarial_paraphrases(holdout_machine_texts)

    adv_holdout_df = pd.DataFrame({"text": adv_holdout_paraphrases})
    adv_holdout_feats = build_features("hardening_holdout_adv", adv_holdout_df, force=True)
    adv_holdout_logits = predict_ensemble(adv_holdout_paraphrases)
    adv_probs = score_split(adv_holdout_df.assign(is_machine=1), adv_holdout_feats, adv_holdout_logits)
    adv_report = classification_report_dict(np.ones(len(adv_probs)), adv_probs)
    report["adversarial_holdout_after_hardening"] = adv_report
    print(f"[hardened][adversarial_holdout] detection_rate={adv_report['accuracy']:.4f}", flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nHardening report written to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
