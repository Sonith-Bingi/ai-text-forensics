"""Reward function for RL adversarial paraphrase training.

Rewards a paraphrase for evading our own trained detector, while penalizing
semantic drift from the original text. Without the fidelity term, the policy's
easiest way to maximize reward is to ignore the input and always emit some
fixed human-sounding sentence -- that would show 100% "evasion" while being
useless as an adversarial example. The fidelity term is deliberately cheap
(lexical overlap + length ratio, no extra model inference) since it runs once
per generated sample inside a tight, already-expensive training loop.
"""
from __future__ import annotations

from forensics.config import CFG
from forensics.inference import get_predictor


def _lexical_overlap(a: str, b: str) -> float:
    """Jaccard overlap over lowercased word sets. Not a real semantic
    similarity metric, but enough to penalize a paraphrase that drifts
    completely off-topic, at zero extra model-inference cost."""
    tok_a = set(a.lower().split())
    tok_b = set(b.lower().split())
    if not tok_a or not tok_b:
        return 0.0
    return len(tok_a & tok_b) / len(tok_a | tok_b)


def _length_penalty(original: str, paraphrase: str) -> float:
    orig_len = max(1, len(original.split()))
    para_len = max(1, len(paraphrase.split()))
    ratio = para_len / orig_len
    cfg = CFG.adversarial
    if cfg.min_length_ratio <= ratio <= cfg.max_length_ratio:
        return 1.0
    if ratio < cfg.min_length_ratio:
        return max(0.0, ratio / cfg.min_length_ratio)
    return max(0.0, cfg.max_length_ratio / ratio)


def compute_reward(original: str, paraphrase: str) -> dict:
    """Returns a dict with every reward component, not just the scalar total --
    keeping components visible is what makes a stalled or diverged overnight
    run diagnosable from the logs alone the next morning."""
    if not paraphrase.strip():
        return {"evasion": 0.0, "fidelity": 0.0, "prob_machine": 1.0, "total": -1.0}

    predictor = get_predictor()
    result = predictor.predict(paraphrase)
    prob_machine = result["probability_machine_generated"]
    evasion = 1.0 - prob_machine  # high reward = detector thinks it's human

    overlap = _lexical_overlap(original, paraphrase)
    length_ok = _length_penalty(original, paraphrase)
    fidelity = overlap * length_ok

    total = evasion + CFG.adversarial.fidelity_weight * fidelity
    return {"evasion": evasion, "fidelity": fidelity, "prob_machine": prob_machine, "total": total}
