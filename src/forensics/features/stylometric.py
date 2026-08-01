"""Hand-crafted stylometric features: cheap, fast, and surprisingly informative
signal about whether text is human- or machine-written (repetition rate, entropy,
lexical diversity, readability all shift measurably under LLM generation).

These are per-text features (unlike the original FakeTextDetector-NLP repo, which
computed pairwise diff/ratio features -- here the task is single-text
classification, so each text gets one feature vector).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
import textstat

from forensics.features.normalize import normalize_text

_WORD_RE = re.compile(r"\w+")
_SENT_SPLIT_RE = re.compile(r"[.!?]+\s")


def _safe_ratio(numerator: float, denom: float) -> float:
    return numerator / denom if denom else 0.0


def punct_ratio(s: str) -> float:
    return _safe_ratio(sum(1 for c in s if unicodedata.category(c).startswith("P")), len(s))


def digit_ratio(s: str) -> float:
    return _safe_ratio(sum(c.isdigit() for c in s), len(s))


def upper_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return _safe_ratio(sum(c.isupper() for c in letters), len(letters))


def char_entropy(s: str) -> float:
    if not s:
        return 0.0
    cnt = Counter(s)
    n = len(s)
    p = np.array([v / n for v in cnt.values()])
    return float(-(p * np.log2(p + 1e-12)).sum())


def type_token_ratio(s: str) -> float:
    toks = _WORD_RE.findall(s.lower())
    return _safe_ratio(len(set(toks)), len(toks))


def repeated_trigram_rate(s: str) -> float:
    toks = re.findall(r"\w+|\S", s)
    if len(toks) < 3:
        return 0.0
    grams = Counter(tuple(toks[i : i + 3]) for i in range(len(toks) - 2))
    tot = max(1, len(toks) - 2)
    repeats = sum(v for v in grams.values() if v > 1)
    return repeats / tot


def sentence_length_stats(s: str) -> tuple[float, float]:
    """Returns (mean, std) of sentence length in words. LLM output tends to have
    unnaturally *low variance* sentence lengths compared to human writing."""
    sents = [sent for sent in _SENT_SPLIT_RE.split(s) if sent.strip()]
    if not sents:
        return 0.0, 0.0
    lens = [len(_WORD_RE.findall(sent)) for sent in sents]
    return float(np.mean(lens)), float(np.std(lens))


def burstiness(s: str) -> float:
    """Word-frequency burstiness: (std - mean) / (std + mean) of word occurrence
    counts. Human text is typically more 'bursty' (topical words cluster); LLM
    text tends to distribute vocabulary more smoothly."""
    toks = _WORD_RE.findall(s.lower())
    if len(toks) < 2:
        return 0.0
    counts = np.array(list(Counter(toks).values()), dtype=float)
    mean, std = counts.mean(), counts.std()
    denom = std + mean
    return float((std - mean) / denom) if denom else 0.0


def readability_features(s: str) -> dict[str, float]:
    if len(s.split()) < 3:
        return {"flesch_reading_ease": 0.0, "gunning_fog": 0.0, "smog_index": 0.0}
    try:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(s),
            "gunning_fog": textstat.gunning_fog(s),
            "smog_index": textstat.smog_index(s),
        }
    except Exception:
        return {"flesch_reading_ease": 0.0, "gunning_fog": 0.0, "smog_index": 0.0}


def stylometric_features(text: str) -> dict[str, float]:
    s = normalize_text(text)
    mean_sent_len, std_sent_len = sentence_length_stats(s)
    feats = {
        "len_chars": len(s),
        "len_words": len(_WORD_RE.findall(s)),
        "punct_ratio": punct_ratio(s),
        "digit_ratio": digit_ratio(s),
        "upper_ratio": upper_ratio(s),
        "char_entropy": char_entropy(s),
        "ttr": type_token_ratio(s),
        "rep3_rate": repeated_trigram_rate(s),
        "mean_sent_len": mean_sent_len,
        "std_sent_len": std_sent_len,
        "burstiness": burstiness(s),
    }
    feats.update(readability_features(s))
    return feats


def stylometric_feature_names() -> list[str]:
    return list(stylometric_features("Example sentence. Another one here!").keys())
