"""Central configuration: paths, seeds, device, and model/training hyperparameters.

Kept as a single module (rather than Hydra) so every other module has one obvious
place to import constants from -- `from forensics.config import CFG`.
"""
from __future__ import annotations

import os

# Both PyTorch and LightGBM/SHAP bundle their own OpenMP runtime; on this
# machine (macOS/Apple Silicon), any multi-threaded LightGBM or SHAP call
# (train, predict, or TreeExplainer) segfaults once torch has been imported in
# the same process, unless OpenMP is pinned to a single thread process-wide.
# Confirmed by isolating the crash to lgb.train()/predict()/shap.TreeExplainer
# specifically -- KMP_DUPLICATE_LIB_OK alone does not fix it, OMP_NUM_THREADS
# does. Must be set before torch/lightgbm/shap are imported anywhere, which is
# why it's the first thing this (universally-imported-first) module does.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

for d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = device()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EncoderConfig:
    # DeBERTa-v3's disentangled attention has no fused/optimized kernel on
    # PyTorch's MPS backend (it doesn't go through scaled_dot_product_attention),
    # which measured 5-10x slower per step than a standard-attention model of
    # similar size at the same sequence length on an M5 -- projected full 5-fold
    # CV time was 12+ hours. distilroberta-base uses standard attention (benefits
    # from MPS's optimized SDPA path) and is a legitimate modern encoder choice
    # in its own right, not just a workaround.
    model_name: str = "distilroberta-base"
    max_len: int = 192  # most human/machine stylistic signal concentrates early in a document
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("query", "value")
    lr: float = 2e-4  # LoRA tolerates a higher LR than full fine-tuning
    epochs: int = 2
    batch_size: int = 32
    grad_accum: int = 1
    warmup_ratio: float = 0.1
    n_folds: int = 5


@dataclass
class StatDetectorConfig:
    scoring_model_name: str = "distilgpt2"
    max_len: int = 512
    # Fast-DetectGPT perturbation sampling
    n_perturbations: int = 20
    perturbation_temperature: float = 1.0


@dataclass
class BlenderConfig:
    n_folds: int = 5
    lgb_params: dict = field(
        default_factory=lambda: dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=20,
            max_depth=-1,
            verbosity=-1,
            seed=SEED,
        )
    )
    lgb_rounds: int = 1000


@dataclass
class AdversarialConfig:
    """RL-trained paraphraser that adversarially targets our own trained
    detector -- see src/forensics/adversarial/. Warm-started from an existing
    paraphrase model (not a raw LM) so the policy already knows how to
    paraphrase on step 1; RL fine-tuning only has to learn evasion, not
    paraphrasing from scratch, which is what makes an overnight unsupervised
    run plausible instead of a very likely non-convergent one.
    """
    paraphraser_model_name: str = "humarin/chatgpt_paraphraser_on_T5_base"
    # Semantic-similarity fidelity gate (see adversarial/reward.py) -- replaces
    # a v1 lexical-overlap fidelity term that a first training run showed does
    # NOT require coherence, and got reward-hacked into incoherent word salad
    # within ~1800 steps. all-MiniLM-L6-v2 is small/fast enough to run once per
    # generated sample inside the training loop without materially slowing it.
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q", "v")
    lr: float = 1e-5
    batch_size: int = 8
    max_input_len: int = 200
    max_new_tokens: int = 80
    min_length_ratio: float = 0.4
    max_length_ratio: float = 2.5
    baseline_momentum: float = 0.95  # EMA decay for the REINFORCE reward baseline
    checkpoint_every: int = 25
    max_steps: int = 2000
    max_seconds: int = 5 * 3600  # hard wall-clock budget; loop exits cleanly at this point regardless of step count


@dataclass
class Config:
    seed: int = SEED
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    stat: StatDetectorConfig = field(default_factory=StatDetectorConfig)
    blender: BlenderConfig = field(default_factory=BlenderConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    # Cap on rows pulled from MAGE for laptop-scale training; None = use all.
    max_train_rows: int | None = 20000
    max_eval_rows: int | None = 4000


CFG = Config()
