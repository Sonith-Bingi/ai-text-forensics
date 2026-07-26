"""Reward function for RL adversarial paraphrase training.

v2 -- rewritten after the first training run demonstrated reward hacking. The
original fidelity term (lexical word overlap) doesn't require coherence, only
that some of the original's words appear somewhere in the output -- incoherent
word salad satisfies that just as well as a real paraphrase, and by step ~1800
of the first run the policy had found exactly that exploit, degenerating into
gibberish that fooled the small-scoring-model-based detectors (the same
jargon/high-perplexity blind spot documented in the README) while still
scoring positively on lexical overlap.

Two changes fix this:
  1. Fidelity is now embedding-based semantic similarity (a small sentence-
     transformer), which actually distinguishes "reworded but same meaning"
     from "word salad that happens to share some words" -- confirmed
     concretely: the first run's garbled output scores 0.35 semantic
     similarity to its original, vs 0.79 for a real paraphrase of the same
     text and 0.07 for genuinely unrelated text.
  2. The reward is now MULTIPLICATIVE (evasion * fidelity_gate) instead of
     additive (evasion + weight * fidelity). Additive combination lets high
     evasion alone dominate the total regardless of how low fidelity is --
     that's exactly the gap the policy exploited. Multiplicative combination
     means low fidelity crushes the reward toward zero no matter how well the
     output evades the detector, closing the exploit rather than just
     discouraging it.
"""
from __future__ import annotations

from forensics.config import CFG
from forensics.inference import get_predictor

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(CFG.adversarial.semantic_model_name)
    return _embedder


def semantic_similarity(a: str, b: str) -> float:
    from sentence_transformers import util

    embedder = _get_embedder()
    emb = embedder.encode([a, b])
    return float(util.cos_sim(emb[0:1], emb[1:2])[0][0])


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
    keeping components visible is what made the first run's reward hacking
    diagnosable from the logs after the fact, and is what should catch it
    *during* training this time (watch `fidelity` in the logs, not just
    `total` or `mean_reward`)."""
    if not paraphrase.strip():
        return {"evasion": 0.0, "fidelity": 0.0, "prob_machine": 1.0, "total": -1.0}

    predictor = get_predictor()
    result = predictor.predict(paraphrase)
    prob_machine = result["probability_machine_generated"]
    evasion = 1.0 - prob_machine  # high reward = detector thinks it's human

    sim = semantic_similarity(original, paraphrase)
    length_ok = _length_penalty(original, paraphrase)
    fidelity = max(0.0, sim) * length_ok

    total = evasion * fidelity
    return {"evasion": evasion, "fidelity": fidelity, "semantic_sim": sim, "prob_machine": prob_machine, "total": total}
