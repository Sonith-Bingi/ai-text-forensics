"""Single-text human-vs-machine classifier: distilroberta-base + LoRA (PEFT).

Model choice note: DeBERTa-v3's disentangled attention has no fused kernel on
PyTorch's MPS backend (it bypasses scaled_dot_product_attention entirely),
which measured 5-10x slower per training step than a standard-attention model
of similar size at the same sequence length on an Apple M5 -- a projected 12+
hours for full 5-fold CV. distilroberta-base uses standard attention, which
MPS's optimized SDPA path accelerates properly, and is a legitimate modern
encoder choice in its own right (see config.py for the full rationale).

Why LoRA instead of full fine-tuning: only a small fraction of parameters are
trainable, which (a) trains fast enough on an Apple M5's MPS backend to make
5-fold CV practical in minutes rather than hours, and (b) means each fold's
adapter checkpoint is a few MB instead of the full model size, so we can keep
all 5 fold adapters on disk for ensembling at inference time for free.

5-fold CV (rather than one train/val split) exists for a specific reason: we
need out-of-fold (OOF) logits on the *training* set to feed the meta-learner
in models/blender.py without leaking labels the blender is trying to predict.
At inference time we ensemble all 5 folds' predictions.
"""
from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from forensics.config import ARTIFACTS_DIR, CFG, DEVICE, SEED

ENCODER_DIR = ARTIFACTS_DIR / "encoder_folds"
ENCODER_DIR.mkdir(parents=True, exist_ok=True)

_tok = None


def get_tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(CFG.encoder.model_name)
    return _tok


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray | None = None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = float(self.labels[idx])
        return item


def make_collate(max_len: int = CFG.encoder.max_len):
    tok = get_tokenizer()

    def collate(batch):
        texts = [b["text"] for b in batch]
        # Fixed-length padding (not dynamic per-batch) keeps every batch the same
        # tensor shape, which avoids MPS re-tracing/recompiling its fused kernels
        # for a new shape on every step -- a real cost we measured, not a
        # theoretical one.
        enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        if "label" in batch[0]:
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.float).unsqueeze(-1)
            return enc, labels
        return enc, None

    return collate


def build_model() -> PeftModel:
    # transformers' `from_pretrained` will otherwise auto-select float16, which
    # reliably NaNs out during LoRA training on the MPS backend.
    base = AutoModelForSequenceClassification.from_pretrained(
        CFG.encoder.model_name, num_labels=1, torch_dtype=torch.float32
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CFG.encoder.lora_r,
        lora_alpha=CFG.encoder.lora_alpha,
        lora_dropout=CFG.encoder.lora_dropout,
        target_modules=list(CFG.encoder.lora_target_modules),
    )
    model = get_peft_model(base, lora_cfg)
    return model.to(DEVICE)


@torch.no_grad()
def predict_logits(model, texts: list[str], batch_size: int = 32) -> np.ndarray:
    model.eval()
    collate = make_collate()
    ds = TextDataset(texts)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    out_logits = []
    for enc, _ in dl:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        out_logits.append(logits)
    return np.concatenate(out_logits)


def train_fold(
    train_texts: list[str], train_labels: np.ndarray,
    val_texts: list[str], val_labels: np.ndarray,
    fold: int,
) -> tuple[PeftModel, np.ndarray]:
    model = build_model()
    collate = make_collate()
    dl_tr = DataLoader(
        TextDataset(train_texts, train_labels),
        batch_size=CFG.encoder.batch_size, shuffle=True, collate_fn=collate,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=CFG.encoder.lr, weight_decay=0.01)
    total_steps = (len(dl_tr) // CFG.encoder.grad_accum) * CFG.encoder.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(CFG.encoder.warmup_ratio * total_steps), total_steps
    )

    best_acc, best_val_logits = -1.0, None
    step = 0
    for epoch in range(CFG.encoder.epochs):
        model.train()
        losses = []
        optim.zero_grad()
        for i, (enc, labels) in enumerate(dl_tr):
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            labels = labels.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(**enc).logits, labels)
            (loss / CFG.encoder.grad_accum).backward()
            losses.append(loss.item())
            if (i + 1) % CFG.encoder.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

        val_logits = predict_logits(model, val_texts, batch_size=CFG.encoder.batch_size * 2)
        val_acc = ((val_logits >= 0).astype(int) == val_labels).mean()
        print(f"  [fold {fold}] epoch {epoch + 1}/{CFG.encoder.epochs} "
              f"loss={np.mean(losses):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_val_logits = val_logits
            model.save_pretrained(ENCODER_DIR / f"fold{fold}")

    print(f"  [fold {fold}] best val_acc={best_acc:.4f}")
    return model, best_val_logits


def run_encoder_cv(train_df) -> np.ndarray:
    """Trains CFG.encoder.n_folds LoRA adapters, returns out-of-fold logits
    (same order as train_df) for the meta-learner.

    Resumable at fold granularity: StratifiedKFold with a fixed seed produces
    the same fold assignments every run, so a fold whose OOF logits were
    already saved to disk is skipped rather than retrained. Without this, a
    crash during fold 4 of 5 (a real, repeated failure mode in this
    environment -- sandbox restarts, not code bugs) would silently discard
    3 completed folds' worth of training time on the next run.
    """
    texts = train_df["text"].tolist()
    labels = train_df["is_machine"].values
    oof_logits = np.zeros(len(texts))

    skf = StratifiedKFold(n_splits=CFG.encoder.n_folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels), 1):
        oof_path = ENCODER_DIR / f"fold{fold}_oof.npy"
        if oof_path.exists() and (ENCODER_DIR / f"fold{fold}").exists():
            print(f"\n--- Encoder fold {fold}/{CFG.encoder.n_folds}: already trained, skipping ---")
            oof_logits[va_idx] = np.load(oof_path)
            continue

        print(f"\n--- Encoder fold {fold}/{CFG.encoder.n_folds} ---")
        tr_texts = [texts[i] for i in tr_idx]
        va_texts = [texts[i] for i in va_idx]
        model, val_logits = train_fold(tr_texts, labels[tr_idx], va_texts, labels[va_idx], fold)
        oof_logits[va_idx] = val_logits
        np.save(oof_path, val_logits)  # written last -- its presence means this fold is genuinely done
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    oof_acc = ((oof_logits >= 0).astype(int) == labels).mean()
    print(f"\nEncoder OOF accuracy: {oof_acc:.4f}")
    return oof_logits


def load_fold_models() -> list[PeftModel]:
    models = []
    for fold_dir in sorted(ENCODER_DIR.glob("fold*")):
        base = AutoModelForSequenceClassification.from_pretrained(CFG.encoder.model_name, num_labels=1)
        model = PeftModel.from_pretrained(base, fold_dir).to(DEVICE).eval()
        models.append(model)
    return models


def predict_ensemble(texts: list[str], models: list[PeftModel] | None = None) -> np.ndarray:
    """Average logits across all trained fold adapters -- used for every
    evaluation split and at serving time."""
    if models is None:
        models = load_fold_models()
    all_logits = np.stack([predict_logits(m, texts) for m in models], axis=0)
    return all_logits.mean(axis=0)
