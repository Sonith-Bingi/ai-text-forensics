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


def analyze(text: str):
    if not text or not text.strip():
        return "Enter some text first.", None, ""

    predictor = get_predictor()
    result = predictor.predict(text)
    prob = result["probability_machine_generated"]
    verdict = f"### {result['label'].upper()} — {prob:.1%} probability machine-generated"

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
    html = render_html_saliency(scores)

    return verdict, df, html


with gr.Blocks(title="AI Text Forensics") as demo:
    gr.Markdown("# AI Text Forensics\nPaste text below to check whether it looks human-written or machine-generated.")
    with gr.Row():
        inp = gr.Textbox(lines=8, label="Text to analyze", placeholder="Paste text here...")
    btn = gr.Button("Analyze", variant="primary")
    verdict_out = gr.Markdown()
    with gr.Row():
        detector_out = gr.BarPlot(x="detector", y="value", title="Per-detector signal", vertical=False)
    gr.Markdown("**Token saliency** (darker = more influential on the encoder's prediction):")
    saliency_out = gr.HTML()

    gr.Examples(examples=EXAMPLES, inputs=inp)
    btn.click(analyze, inputs=inp, outputs=[verdict_out, detector_out, saliency_out])

if __name__ == "__main__":
    demo.launch()
