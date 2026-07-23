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


def token_saliency(
    model, text: str, max_len: int = CFG.encoder.max_len
) -> list[tuple[int, int, float]]:
    """Returns (char_start, char_end, saliency) triples into the *original*
    `text` string -- not raw BPE token strings.

    RoBERTa's byte-level BPE vocabulary entries (e.g. "Ġthe", "Ċ") are each a
    reversible mapping of raw UTF-8 *bytes* to printable placeholder
    characters, not display-ready text -- a single real character (an em dash,
    an arrow, a non-ASCII letter) is frequently split across multiple BPE
    tokens, so decoding each token string in isolation (as opposed to via the
    tokenizer's offset mapping back into the original string) reliably
    produces mojibake for any non-trivial-ASCII input. Using
    `return_offsets_mapping` sidesteps the byte-level encoding entirely: we
    just slice the original string, which is always correct by construction.
    """
    tok = get_tokenizer()
    enc = tok(
        text, return_tensors="pt", truncation=True, max_length=max_len,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
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

    special_ids = set(tok.all_special_ids)
    ids = input_ids.squeeze(0).cpu().tolist()
    results = []
    for tid, (start, end), score in zip(ids, offsets, saliency.tolist()):
        if tid in special_ids or start == end:
            continue
        results.append((start, end, score))
    return results


def render_html_saliency(text: str, token_scores: list[tuple[int, int, float]]) -> str:
    """Cheap heatmap for the Gradio demo: darker highlight = more influential
    on the "machine-generated" prediction. Renders the original text with
    highlighted spans -- untouched gaps between tokens (mostly whitespace)
    are preserved as plain, escaped text so nothing is dropped or garbled."""
    parts = []
    cursor = 0
    for start, end, score in sorted(token_scores):
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        alpha = float(np.clip(score, 0, 1)) * 0.75
        parts.append(
            f'<span style="background: rgba(220,50,50,{alpha:.2f})">{html.escape(text[start:end])}</span>'
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return "<div style='font-family: monospace; line-height:1.8; white-space: pre-wrap'>" + "".join(parts) + "</div>"
