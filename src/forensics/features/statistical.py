"""Zero-shot statistical detectors: no training data needed, just a small causal LM
used as a *scoring* model. These reproduce (in spirit, not as byte-exact paper
reproductions) three published methods that predate/complement supervised
classifiers:

  * Log-Likelihood       -- the oldest baseline: LLM text sits in higher-probability
                             regions of another LLM's distribution than human text.
  * Log-Rank             -- average rank of the observed token in the model's
                             probability-sorted vocabulary; LLM text picks
                             "obvious" (low-rank) tokens more often.
  * LRR (DetectLLM,       -- Su et al. 2023: ratio of log-likelihood to log-rank.
    Su et al. 2023)          Combines both signals into one more robust score.
  * Fast-DetectGPT        -- Bao et al. 2024: instead of DetectGPT's slow
    curvature              perturb-and-rescore loop (which needs a separate
                             mask-infilling model and hundreds of forward passes),
                             Fast-DetectGPT samples alternative tokens directly
                             from the *same* per-position distribution already
                             computed for log-likelihood, and measures how much of
                             an outlier the true continuation's likelihood is
                             relative to those resampled alternatives. This is the
                             whole reason it's "fast": one forward pass gives you
                             everything needed for all `n_perturbations` samples.

All four are combined into a single scoring-model forward pass per text where
possible, and results are cached (a given text is scored once even if it appears
in multiple pairs/splits).
"""
from __future__ import annotations


import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from forensics.config import CFG, DEVICE
from forensics.features.normalize import normalize_text

_tok = None
_model = None


def _lazy_load(model_name: str = CFG.stat.scoring_model_name):
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()
    return _tok, _model


@torch.no_grad()
def _token_logprobs_and_ranks(text: str, max_len: int = CFG.stat.max_len):
    """One forward pass -> per-token log-prob of the *observed* next token, its
    rank among the vocab, and the full log-prob distribution at each position
    (needed for Fast-DetectGPT resampling)."""
    tok, model = _lazy_load()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc.input_ids.to(DEVICE)
    if input_ids.shape[1] < 2:
        return None
    out = model(input_ids)
    logits = out.logits[0, :-1, :]  # predict token t+1 from position t
    targets = input_ids[0, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # rank of the true token within the sorted (descending) probability order
    sorted_idx = torch.argsort(log_probs, dim=-1, descending=True)
    ranks = (sorted_idx == targets.unsqueeze(-1)).float().argmax(dim=-1).float() + 1.0

    return {"log_probs": log_probs, "token_logp": token_logp, "ranks": ranks}


def log_likelihood(text: str) -> float:
    r = _token_logprobs_and_ranks(text)
    if r is None:
        return 0.0
    return r["token_logp"].mean().item()


def log_rank(text: str) -> float:
    r = _token_logprobs_and_ranks(text)
    if r is None:
        return 0.0
    return torch.log(r["ranks"]).mean().item()


def lrr(text: str) -> float:
    """DetectLLM's Log-Likelihood Log-Rank Ratio (Su et al., 2023). Bounded ratio
    that is more stable across text lengths/domains than either signal alone."""
    r = _token_logprobs_and_ranks(text)
    if r is None:
        return 0.0
    ll = r["token_logp"].mean().item()
    lr = torch.log(r["ranks"]).mean().item()
    return ll / lr if abs(lr) > 1e-6 else 0.0


@torch.no_grad()
def fast_detectgpt_curvature(
    text: str, n_perturbations: int = CFG.stat.n_perturbations
) -> float:
    """Conditional probability curvature (Bao et al., 2024), single-model variant.

    Resamples each target token independently from the model's own per-position
    distribution (already computed for log-likelihood -- no extra forward pass),
    building `n_perturbations` alternative "what the model itself would have
    written here" continuations. If the observed text's likelihood is a strong
    outlier (much higher than these self-sampled alternatives), that is exactly
    what we'd expect if the text *was* produced by sampling from this model
    family -- the curvature score captures this as a z-score.
    """
    r = _token_logprobs_and_ranks(text)
    if r is None:
        return 0.0
    log_probs = r["log_probs"]  # (T, V)
    observed_ll = r["token_logp"].sum().item()

    probs = log_probs.exp()
    perturbed_lls = []
    for _ in range(n_perturbations):
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (T,)
        sampled_logp = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        perturbed_lls.append(sampled_logp.sum().item())

    perturbed_lls = torch.tensor(perturbed_lls)
    mean_p, std_p = perturbed_lls.mean().item(), perturbed_lls.std().item()
    if std_p < 1e-6:
        return 0.0
    return (observed_ll - mean_p) / std_p


_cache: dict[str, dict[str, float]] = {}


def statistical_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    if text in _cache:
        return _cache[text]
    r = _token_logprobs_and_ranks(text)
    if r is None:
        feats = {"stat_loglik": 0.0, "stat_logrank": 0.0, "stat_lrr": 0.0, "stat_curvature": 0.0}
    else:
        ll = r["token_logp"].mean().item()
        lr = torch.log(r["ranks"]).mean().item()
        feats = {
            "stat_loglik": ll,
            "stat_logrank": lr,
            "stat_lrr": ll / lr if abs(lr) > 1e-6 else 0.0,
            "stat_curvature": fast_detectgpt_curvature(text),
        }
    _cache[text] = feats
    return feats


def statistical_feature_names() -> list[str]:
    return ["stat_loglik", "stat_logrank", "stat_lrr", "stat_curvature"]
