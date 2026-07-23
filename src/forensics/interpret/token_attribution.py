"""Token-level saliency for the encoder: gradient x embedding-norm attribution.
Simpler than Integrated Gradients (no extra dependency, one backward pass), and
good enough to highlight which spans of text pushed the encoder toward
"machine-generated" -- useful for the demo UI's "why" explanation.
"""
from __future__ import annotations

import html

import numpy as np

from forensics.config import CFG, DEVICE
from forensics.models.encoder import get_tokenizer


def token_saliency(model, text: str, max_len: int = CFG.encoder.max_len) -> list[tuple[str, float]]:
    tok = get_tokenizer()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).detach().clone().requires_grad_(True)

    model.eval()
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logit = out.logits.squeeze()
    logit.backward()

    grad_norm = inputs_embeds.grad.norm(dim=-1).squeeze(0)  # (seq_len,)
    saliency = (grad_norm / (grad_norm.max() + 1e-8)).detach().cpu().numpy()

    tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())
    return list(zip(tokens, saliency.tolist()))


_SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}


def render_html_saliency(token_scores: list[tuple[str, float]]) -> str:
    """Cheap heatmap for the Gradio demo: darker highlight = more influential
    on the "machine-generated" prediction."""
    spans = []
    for tok, score in token_scores:
        if tok in _SPECIAL_TOKENS:
            continue
        display = tok.replace("▁", " ").replace("Ġ", " ")
        # Token text is untrusted as far as HTML goes -- special tokens like
        # "<s>" would otherwise be parsed as real markup (confirmed: it renders
        # as a literal strikethrough across the whole span).
        display = html.escape(display)
        alpha = float(np.clip(score, 0, 1)) * 0.75
        spans.append(f'<span style="background: rgba(220,50,50,{alpha:.2f})">{display}</span>')
    return "<div style='font-family: monospace; line-height:1.8'>" + "".join(spans) + "</div>"
