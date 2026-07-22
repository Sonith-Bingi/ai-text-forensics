"""Central configuration: paths, seeds, device, and model/training hyperparameters.

Kept as a single module (rather than Hydra) so every other module has one obvious
place to import constants from -- `from forensics.config import CFG`.
"""
from __future__ import annotations

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
    model_name: str = "microsoft/deberta-v3-small"
    max_len: int = 384
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("query_proj", "value_proj")
    lr: float = 2e-4  # LoRA tolerates a higher LR than full fine-tuning
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 2
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
class Config:
    seed: int = SEED
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    stat: StatDetectorConfig = field(default_factory=StatDetectorConfig)
    blender: BlenderConfig = field(default_factory=BlenderConfig)
    # Cap on rows pulled from MAGE for laptop-scale training; None = use all.
    max_train_rows: int | None = 20000
    max_eval_rows: int | None = 4000


CFG = Config()
