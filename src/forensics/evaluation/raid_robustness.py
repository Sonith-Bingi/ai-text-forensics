"""Scores the trained pipeline on the curated RAID adversarial-attack sample
(data/raid.py) and reports detection rate per attack type. RAID's models
(gpt2, mistral, mpt-chat, ...) were never seen during training -- this is
simultaneously an unseen-generator test AND an attack-robustness test, on top
of MAGE's built-in paraphrase-attack slice (evaluation/robustness.py).
"""
from __future__ import annotations

import pandas as pd

from forensics.evaluation.generalization import score_split
from forensics.evaluation.metrics import classification_report_dict
from forensics.features.cache import build_features


def raid_attack_report(raid_df: pd.DataFrame) -> dict[str, dict]:
    raid_feats = build_features("raid_sample", raid_df)
    from forensics.models.encoder import predict_ensemble

    encoder_logits = predict_ensemble(raid_df["text"].tolist())

    results = {}
    for attack, group in raid_df.groupby("attack"):
        idx = group.index
        sub_df = raid_df.loc[idx].reset_index(drop=True)
        sub_feats = raid_feats.loc[idx].reset_index(drop=True)
        sub_logits = encoder_logits[idx.to_numpy()]
        probs = score_split(sub_df, sub_feats, sub_logits)
        y = sub_df["is_machine"].values
        report = classification_report_dict(y, probs)
        results[attack] = report
        print(f"[RAID attack={attack}] n={report['n']} acc={report['accuracy']:.4f} f1={report['f1']:.4f}")

    return results
