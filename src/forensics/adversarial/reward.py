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

import re

from forensics.config import CFG
from forensics.inference import get_predictor

_embedder = None
_REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")  # same character 5+ times in a row, e.g. "EEEEE"


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


def _wellformedness_gate(text: str) -> float:
    """v3 fix -- the v2 fidelity gate (semantic similarity) let a specific
    degenerate pattern through: a coherent paraphrase with a repeated-
    character or ALL-CAPS garbage tail bolted on (e.g. "...she whispered,
    regretting, OMG NO NO MEEEEEEEEEEEEEEEEAKE US OUR BONUS STATION"),
    typically because the policy never produced an EOS token and ran to
    max_new_tokens. A short garbage tail barely moves a whole-sentence
    embedding, so semantic similarity to the original stays high enough to
    clear the fidelity bar -- but the *stylometric* features in the real
    detector (upper_ratio, rep3_rate) flag this pattern hard, which is why
    training against it made held-out detection WORSE, not better: the
    reward signal (evading a small-model-based sub-check) and the actual
    outcome (the full blended detector) disagreed. This is a hard 0/1 gate,
    not a smooth penalty -- there's no partial credit for "mostly coherent
    with gibberish stapled on", the whole point is to make that pattern
    worthless to the policy rather than merely discounted.
    """
    if _REPEATED_CHAR_RE.search(text):
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    # A *consecutive run* of all-caps words, not an overall ratio -- real text
    # legitimately contains scattered acronyms (NASA, USA, UN), which a
    # whole-text ratio check flags as false positives at a low threshold.
    # Actual degenerate spam ("OMG NO NO ... BONUS STATION THAT SHOCK") shows
    # up as a long unbroken run, which scattered acronyms never do.
    run = 0
    for w in words:
        if len(w) > 1 and w.isupper():
            run += 1
            if run >= 4:
                return 0.0
        else:
            run = 0
    return 1.0


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


def _combine(evasion: float, fidelity: float) -> float:
    """Two ways to combine evasion + fidelity into one scalar reward, both
    closing the v1 additive exploit (where high evasion alone could dominate
    regardless of fidelity) but with different training dynamics:

    - "multiplicative": evasion * fidelity. Smooth everywhere, but even a
      solidly coherent paraphrase (fidelity ~0.7-0.8) permanently discounts
      the evasion signal by 20-30% -- the two objectives are always in
      tension, which may slow down learning genuine evasion tactics once
      fidelity is already "good enough".
    - "threshold_gate": full evasion credit once fidelity clears a bar
      (default 0.55), steep penalty proportional to the shortfall below it.
      Decouples the objectives once past the bar -- below it, all that
      matters is climbing back over; above it, all that matters is
      maximizing evasion -- which may let evasion optimization run faster
      without a persistent fidelity tax.
    """
    cfg = CFG.adversarial
    if cfg.reward_mode == "threshold_gate":
        if fidelity >= cfg.fidelity_threshold:
            return evasion
        return (fidelity - cfg.fidelity_threshold) * 2.0
    return evasion * fidelity  # "multiplicative" (default)


def compute_reward(original: str, paraphrase: str) -> dict:
    """Returns a dict with every reward component, not just the scalar total --
    keeping components visible is what made the first run's reward hacking
    diagnosable from the logs after the fact, and is what should catch it
    *during* training this time (watch `fidelity` in the logs, not just
    `total` or `mean_reward`)."""
    if not paraphrase.strip():
        return {"evasion": 0.0, "fidelity": 0.0, "semantic_sim": 0.0, "prob_machine": 1.0, "total": -1.0}

    predictor = get_predictor()
    result = predictor.predict(paraphrase)
    prob_machine = result["probability_machine_generated"]
    evasion = 1.0 - prob_machine  # high reward = detector thinks it's human

    sim = semantic_similarity(original, paraphrase)
    length_ok = _length_penalty(original, paraphrase)
    wellformed = _wellformedness_gate(paraphrase)
    fidelity = max(0.0, sim) * length_ok * wellformed

    total = _combine(evasion, fidelity)
    return {
        "evasion": evasion,
        "fidelity": fidelity,
        "semantic_sim": sim,
        "wellformed": wellformed,
        "prob_machine": prob_machine,
        "total": total,
    }
