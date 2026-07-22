"""Computes (and disk-caches) stylometric + statistical features for a whole
split at once, so re-running the pipeline after a crash doesn't recompute
40+ minutes of scoring-model forward passes."""
from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from forensics.config import ARTIFACTS_DIR
from forensics.features.statistical import statistical_features
from forensics.features.stylometric import stylometric_features

FEATURES_DIR = ARTIFACTS_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def build_features(split_name: str, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    path = FEATURES_DIR / f"{split_name}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    rows = []
    for text in tqdm(df["text"].tolist(), desc=f"Features [{split_name}]"):
        feats = stylometric_features(text)
        feats.update(statistical_features(text))
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    feat_df.to_parquet(path, index=False)
    return feat_df
