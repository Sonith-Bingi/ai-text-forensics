#!/usr/bin/env python
"""Builds an augmented training set targeting two confirmed gaps that no
amount of MAGE-scale row-count alone fixes:

1. Short/off-length text: the encoder and stylometric features were both
   fine-tuned/tested almost entirely on full-length MAGE documents. A single
   casual sentence gives them near-nothing to work with and both misfired on
   a real example during manual testing (encoder_prob 0.80 on genuine human
   text). Fix: truncate a sample of existing training rows down to short
   spans (8-40 words), keeping the original label -- teaches the encoder and
   stylometric features what short text in both classes actually looks like.

2. Paraphrase robustness generalizing beyond one narrow evader: the hardening
   round (see README) only ever showed the detector adversarial paraphrases
   from one specific RL-tuned policy (v5). Fix: paraphrase a sample of machine
   rows with the *static*, off-the-shelf, non-RL paraphraser instead -- a much
   more generic transformation, so the detector learns "paraphrased machine
   text" as a general pattern rather than one policy's specific style.

Writes data/processed/train.parquet in place (original snapshotted first) so
the existing pipeline (forensics.pipeline.train_all) picks it up with zero
other code changes required.
"""
from __future__ import annotations

import shutil

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM

from forensics.adversarial.paraphraser import get_tokenizer, make_prompt
from forensics.config import CFG, DEVICE, PROCESSED_DIR, SEED

N_SHORT_TEXT = 4000
N_STATIC_PARAPHRASE = 3000
SHORT_MIN_WORDS = 8
SHORT_MAX_WORDS = 40
BATCH_SIZE = 16


def _make_short_text_rows(train_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    sample = train_df.sample(n=min(n, len(train_df)), random_state=seed)
    rng = pd.Series(range(len(sample))).sample(frac=1, random_state=seed).values  # per-row jitter source
    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        words = row["text"].split()
        if len(words) <= SHORT_MIN_WORDS:
            continue
        span = SHORT_MIN_WORDS + (rng[i] % (SHORT_MAX_WORDS - SHORT_MIN_WORDS + 1))
        span = min(span, len(words))
        start_max = max(0, len(words) - span)
        start = int(rng[i]) % (start_max + 1) if start_max > 0 else 0
        snippet = " ".join(words[start : start + span])
        rows.append({"text": snippet, "is_machine": row["is_machine"]})
    return pd.DataFrame(rows)


@torch.no_grad()
def _make_static_paraphrase_rows(train_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    machine_texts = (
        train_df[train_df["is_machine"] == 1]["text"].sample(n=min(n, (train_df["is_machine"] == 1).sum()), random_state=seed).tolist()
    )
    machine_texts = [t[:600] for t in machine_texts]

    tok = get_tokenizer()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.adversarial.paraphraser_model_name, dtype=torch.float32
    ).to(DEVICE).eval()

    outs = []
    for i in range(0, len(machine_texts), BATCH_SIZE):
        batch = machine_texts[i : i + BATCH_SIZE]
        prompts = [make_prompt(t) for t in batch]
        enc = tok(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=CFG.adversarial.max_input_len,
        ).to(DEVICE)
        gen = model.generate(
            **enc, max_new_tokens=CFG.adversarial.max_new_tokens,
            do_sample=True, top_p=0.9, temperature=1.0,
        )
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  static-paraphrased {len(outs)}/{len(machine_texts)}", flush=True)

    outs = [t for t in outs if t.strip()]
    return pd.DataFrame({"text": outs, "is_machine": 1})


def main():
    train_path = PROCESSED_DIR / "train.parquet"
    backup_path = PROCESSED_DIR / "train_45k_snapshot.parquet"
    if not backup_path.exists():
        shutil.copy2(train_path, backup_path)
        print(f"Snapshotted original train set to {backup_path}", flush=True)
    else:
        print(f"Snapshot already exists at {backup_path}, not overwriting", flush=True)

    train_df = pd.read_parquet(backup_path)  # always augment from the ORIGINAL 45K, never a prior augmented run
    print(f"Original training rows: {len(train_df)}", flush=True)

    print("\n=== Building short-text augmentation ===", flush=True)
    short_df = _make_short_text_rows(train_df, N_SHORT_TEXT, seed=SEED)
    print(f"Short-text rows: {len(short_df)}", flush=True)

    print("\n=== Building static-paraphrase augmentation ===", flush=True)
    para_df = _make_static_paraphrase_rows(train_df, N_STATIC_PARAPHRASE, seed=SEED)
    print(f"Static-paraphrase rows: {len(para_df)}", flush=True)

    combined = pd.concat(
        [train_df[["text", "is_machine"]], short_df, para_df], ignore_index=True
    )
    print(f"\nCombined training rows: {len(combined)} "
          f"(original {len(train_df)} + short-text {len(short_df)} + paraphrase {len(para_df)})", flush=True)

    combined.to_parquet(train_path, index=False)
    print(f"Wrote augmented training set to {train_path}", flush=True)


if __name__ == "__main__":
    main()
