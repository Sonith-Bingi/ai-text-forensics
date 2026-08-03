"""Interactive demo: paste text, get a calibrated verdict, a per-detector score
breakdown, and a token-highlighted view of what pushed the encoder toward
"machine-generated". Run with `python demo/app.py` (needs trained artifacts)."""
from __future__ import annotations

import gradio as gr
import pandas as pd

from forensics.inference import get_predictor
from forensics.interpret.token_attribution import render_html_saliency, token_saliency

EXAMPLES = [
    "I woke up this morning and honestly could not find my keys anywhere, "
    "which was super annoying because I was already running late for the bus.",
    "In conclusion, it is important to note that the aforementioned factors "
    "collectively contribute to a comprehensive understanding of the topic at hand. "
    "Furthermore, these considerations underscore the significance of this analysis.",
]

# Below this, measured accuracy is close enough to a coin flip (59.2% under
# 30 words, and worse still under ~15) that returning a confident-looking
# verdict does more harm than good -- it looks like the system just guessed,
# because it did. Declining to answer is the more honest, more defensible
# behavior for input this short, not a workaround for a weak model.
MIN_WORDS_FOR_VERDICT = 15

_RELIABILITY_COLOR = {
    "very low": "#dc2626",
    "low": "#ea580c",
    "moderate": "#d97706",
    "good": "#65a30d",
    "high": "#16a34a",
    "very high": "#15803d",
}

_CUSTOM_CSS = """
.verdict-card { padding: 1.25rem 1.5rem; border-radius: 12px; margin-bottom: 0.75rem; }
.verdict-label { font-size: 1.4rem; font-weight: 700; margin: 0 0 0.25rem 0; }
.verdict-prob { font-size: 0.95rem; opacity: 0.85; }
.reliability-banner { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.75rem;
  font-size: 0.9rem; border-left: 4px solid; }

/* Text color is fixed (not inherited from the page theme) because the card
   background below is also fixed -- letting text color follow a dark-mode
   theme toggle while the background stays a light pastel is exactly what
   produced near-invisible white-on-light-pink text in dark mode. */
.verdict-machine, .verdict-human, .verdict-insufficient { color: #1f2937; }
.verdict-machine { background: linear-gradient(135deg, #fef2f2, #fee2e2); border: 1px solid #fca5a5; }
.verdict-human { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1px solid #86efac; }
.verdict-insufficient { background: linear-gradient(135deg, #f8fafc, #f1f5f9); border: 1px solid #cbd5e1; }

@media (prefers-color-scheme: dark) {
  .verdict-machine, .verdict-human, .verdict-insufficient { color: #f3f4f6; }
  .verdict-machine { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #b91c1c; }
  .verdict-human { background: linear-gradient(135deg, #052e16, #14532d); border: 1px solid #15803d; }
  .verdict-insufficient { background: linear-gradient(135deg, #1e293b, #334155); border: 1px solid #64748b; }
}
"""


def _verdict_html(result: dict) -> str:
    prob = result["probability_machine_generated"]
    is_machine = result["label"] == "machine-generated"
    css_class = "verdict-machine" if is_machine else "verdict-human"
    label_text = "MACHINE-GENERATED" if is_machine else "HUMAN-WRITTEN"
    icon = "\U0001f916" if is_machine else "\U0001f9d1"
    return (
        f'<div class="verdict-card {css_class}">'
        f'<p class="verdict-label">{icon} {label_text}</p>'
        f'<p class="verdict-prob">{prob:.1%} probability machine-generated</p>'
        f"</div>"
    )


def _reliability_html(result: dict) -> str:
    reliability = result["reliability"]
    n_words = result["word_count"]
    measured_acc = result["reliability_measured_accuracy"]
    color = _RELIABILITY_COLOR.get(reliability, "#6b7280")
    if reliability in ("very low", "low"):
        note = (
            f"This text is short ({n_words} words). On held-out test data, accuracy in this "
            f"length range measured only <b>{measured_acc:.0%}</b> — treat this verdict as a "
            f"weak signal, not a confident answer. See the "
            f'<a href="https://github.com/Sonith-Bingi/ai-text-forensics#results" target="_blank">'
            f"length-vs-accuracy table</a> for the full breakdown."
        )
    else:
        note = (
            f"Reliability: <b>{reliability}</b> for text this length ({n_words} words) — "
            f"measured accuracy ~{measured_acc:.0%} on held-out data."
        )
    return f'<div class="reliability-banner" style="border-color:{color}; background:{color}15;">{note}</div>'


def _insufficient_text_html(n_words: int) -> str:
    return (
        '<div class="verdict-card verdict-insufficient">'
        f'<p class="verdict-label">\U0001f50d Not enough text to analyze</p>'
        f'<p class="verdict-prob">{n_words} word{"s" if n_words != 1 else ""} — need at least '
        f"{MIN_WORDS_FOR_VERDICT} for a verdict worth trusting. Measured accuracy under 30 words is "
        f"only 59% (barely better than a coin flip), so this tool declines to guess rather than "
        f"return a confident-looking answer built on no real signal.</p>"
        f"</div>"
    )


def analyze(text: str):
    if not text or not text.strip():
        return "", "", None, ""

    n_words = len(text.split())
    if n_words < MIN_WORDS_FOR_VERDICT:
        return _insufficient_text_html(n_words), "", None, ""

    predictor = get_predictor()
    result = predictor.predict(text)

    verdict_html = _verdict_html(result)
    reliability_html = _reliability_html(result)

    detectors = result["detectors"]
    df = pd.DataFrame(
        {
            "detector": list(detectors.keys()),
            "value": list(detectors.values()),
        }
    )

    # Use fold-0 of the encoder ensemble for the token saliency visualization
    # (saliency is inherently single-model; averaging heatmaps across 5 folds
    # is not meaningfully different for a demo and would 5x the latency).
    model = predictor.encoder_models[0]
    scores = token_saliency(model, text)
    html = render_html_saliency(text, scores)

    return verdict_html, reliability_html, df, html


with gr.Blocks(title="AI Text Forensics", theme=gr.themes.Soft(primary_hue="indigo"), css=_CUSTOM_CSS) as demo:
    gr.Markdown(
        "# \U0001f50e AI Text Forensics\n"
        "Paste text below to check whether it looks human-written or machine-generated. "
        "Uses a blended encoder + statistical + stylometric ensemble, calibrated on held-out data."
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(lines=10, label="Text to analyze", placeholder="Paste text here...")
            btn = gr.Button("Analyze", variant="primary", size="lg")
            gr.Examples(examples=EXAMPLES, inputs=inp, label="Try an example")
        with gr.Column(scale=1):
            verdict_out = gr.HTML()
            reliability_out = gr.HTML()

    detector_out = gr.BarPlot(
        x="detector", y="value", title="Per-detector signal",
        x_title="", y_title="signal (higher = more machine-like)",
    )

    gr.Markdown("### Token saliency\n*(darker = more influential on the encoder's prediction)*")
    saliency_out = gr.HTML()

    btn.click(analyze, inputs=inp, outputs=[verdict_out, reliability_out, detector_out, saliency_out])

if __name__ == "__main__":
    demo.launch()
