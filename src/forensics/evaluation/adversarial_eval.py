"""Before/after evaluation of the RL adversarial paraphraser: does RL
fine-tuning make the paraphraser measurably better at evading our detector
than the off-the-shelf (non-RL-tuned) paraphrase model alone?

Three conditions per held-out machine-generated text:
  1. clean                  -- the original, unparaphrased text
  2. static_paraphrase       -- paraphrased by the plain pretrained model (no RL)
  3. adversarial_paraphrase  -- paraphrased by the RL-fine-tuned policy

Detection rate (recall on the machine class) is the metric that matters: a
paraphrase "succeeds" as an attack when the detector's P(machine) for a text
that started out correctly flagged as machine drops below 0.5 after rewriting.

Uses `test_in_distribution`'s machine rows -- deliberately NOT the train split
the RL loop paraphrased during training -- so this is a genuine held-out
check, not a report of evasion on texts the policy has already seen.
"""
from __future__ import annotations

import json
import traceback

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM

from forensics.adversarial.paraphraser import get_tokenizer, load_policy_model, make_prompt
from forensics.adversarial.train_reinforce import CKPT_DIR
from forensics.config import ARTIFACTS_DIR, CFG, DEVICE, PROCESSED_DIR
from forensics.inference import get_predictor

RESULTS_PATH = ARTIFACTS_DIR / "results" / "adversarial_eval_report.json"


@torch.no_grad()
def _paraphrase_batch(model, tok, texts: list[str]) -> list[str]:
    cfg = CFG.adversarial
    prompts = [make_prompt(t) for t in texts]
    enc = tok(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_input_len
    ).to(DEVICE)
    model.eval()
    gen = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=True, top_p=0.9, temperature=1.0)
    return tok.batch_decode(gen, skip_special_tokens=True)


def _load_eval_texts(n: int) -> list[str]:
    df = pd.read_parquet(PROCESSED_DIR / "test_in_distribution.parquet")
    texts = df[df["is_machine"] == 1]["text"].tolist()
    return [t[:600] for t in texts[:n]]


def run_adversarial_evaluation(n_texts: int = 150, batch_size: int = 8) -> dict:
    predictor = get_predictor()
    tok = get_tokenizer()
    texts = _load_eval_texts(n_texts)
    print(f"Evaluating on {len(texts)} held-out machine-generated texts", flush=True)

    print("Loading off-the-shelf (non-RL) paraphraser...", flush=True)
    static_model = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.adversarial.paraphraser_model_name, torch_dtype=torch.float32
    ).to(DEVICE)

    adv_model = None
    if CKPT_DIR.exists():
        print("Loading RL-fine-tuned adversarial paraphraser...", flush=True)
        adv_model = load_policy_model(CKPT_DIR, is_trainable=False)
    else:
        print("WARNING: no adversarial checkpoint found -- skipping that condition", flush=True)

    static_paraphrases: list[str] = []
    adv_paraphrases: list[str] = []
    kept_texts: list[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            s_out = _paraphrase_batch(static_model, tok, batch)
            a_out = _paraphrase_batch(adv_model, tok, batch) if adv_model is not None else [None] * len(batch)
        except Exception:  # noqa: BLE001 -- skip a bad batch rather than lose the whole eval
            print(f"WARNING: batch {i} failed:\n{traceback.format_exc()}", flush=True)
            continue
        kept_texts.extend(batch)
        static_paraphrases.extend(s_out)
        adv_paraphrases.extend(a_out)
        print(f"  paraphrased {len(kept_texts)}/{len(texts)}", flush=True)

    results = {"clean": [], "static_paraphrase": [], "adversarial_paraphrase": []}
    for i, text in enumerate(kept_texts):
        results["clean"].append(predictor.predict(text)["probability_machine_generated"])
        results["static_paraphrase"].append(predictor.predict(static_paraphrases[i])["probability_machine_generated"])
        if adv_model is not None:
            results["adversarial_paraphrase"].append(
                predictor.predict(adv_paraphrases[i])["probability_machine_generated"]
            )
        if i % 20 == 0:
            print(f"  scored {i}/{len(kept_texts)}", flush=True)

    report = {}
    for condition, probs in results.items():
        if not probs:
            continue
        s = pd.Series(probs)
        report[condition] = {
            "n": len(s),
            "detection_rate": float((s >= 0.5).mean()),
            "mean_prob_machine": float(s.mean()),
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "report": report,
                "example_original": kept_texts[0] if kept_texts else None,
                "example_static_paraphrase": static_paraphrases[0] if static_paraphrases else None,
                "example_adversarial_paraphrase": adv_paraphrases[0] if adv_paraphrases else None,
            },
            f,
            indent=2,
        )

    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_adversarial_evaluation()
