"""Computes (and disk-caches) stylometric + statistical features for a whole
split at once, so re-running the pipeline after a crash doesn't recompute
40+ minutes of scoring-model forward passes.

Checkpoints incrementally (not just at the end) -- a full split at the data
scale used for the bigger detector retrain takes hours, and this environment
has repeatedly dropped mid-run (sandbox restarts, not code bugs). Losing an
entire split's progress to a restart that happens at row 90% would be a
self-inflicted, entirely avoidable waste of that time.
"""
from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from forensics.config import ARTIFACTS_DIR
from forensics.features.statistical import statistical_features
from forensics.features.stylometric import stylometric_features

FEATURES_DIR = ARTIFACTS_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_EVERY = 500  # rows


def build_features(split_name: str, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    path = FEATURES_DIR / f"{split_name}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    partial_path = FEATURES_DIR / f"{split_name}.partial.parquet"
    texts = df["text"].tolist()
    rows: list[dict] = []

    if partial_path.exists() and not force:
        partial_df = pd.read_parquet(partial_path)
        rows = partial_df.to_dict("records")
        print(f"Resuming {split_name} feature extraction from row {len(rows)}/{len(texts)}")

    start = len(rows)
    for i, text in enumerate(tqdm(texts[start:], desc=f"Features [{split_name}]", initial=start, total=len(texts))):
        feats = stylometric_features(text)
        feats.update(statistical_features(text))
        rows.append(feats)

        if (start + i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_parquet(partial_path, index=False)

    feat_df = pd.DataFrame(rows)
    feat_df.to_parquet(path, index=False)
    partial_path.unlink(missing_ok=True)
    return feat_df
