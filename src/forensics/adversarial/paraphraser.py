"""LoRA-wrapped paraphraser policy model for RL adversarial training.

Warm-started from `humarin/chatgpt_paraphraser_on_T5_base` (a T5-base model
already fine-tuned specifically for paraphrasing) rather than a raw language
model -- the RL loop only needs to learn "how to evade this detector while
staying a paraphrase," not "how to paraphrase at all," which is what makes
overnight convergence plausible on a laptop.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from forensics.config import CFG, DEVICE

_tok = None


def get_tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(CFG.adversarial.paraphraser_model_name)
    return _tok


def build_policy_model() -> PeftModel:
    base = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.adversarial.paraphraser_model_name, torch_dtype=torch.float32
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=CFG.adversarial.lora_r,
        lora_alpha=CFG.adversarial.lora_alpha,
        lora_dropout=CFG.adversarial.lora_dropout,
        target_modules=list(CFG.adversarial.lora_target_modules),
    )
    model = get_peft_model(base, lora_cfg)
    return model.to(DEVICE)


def load_policy_model(adapter_dir, is_trainable: bool = False) -> PeftModel:
    # PeftModel.from_pretrained defaults to is_trainable=False (built for
    # inference) -- resuming a training run with that default silently leaves
    # every LoRA param requires_grad=False, so loss.backward() fails with
    # "does not require grad and does not have a grad_fn". Confirmed by
    # testing the resume path before the overnight run, not by inspection.
    base = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.adversarial.paraphraser_model_name, torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)
    return model.to(DEVICE)


def make_prompt(text: str) -> str:
    return f"paraphrase: {text}"
