# AI Text Forensics

Machine-generated text detection, built as a deliberately more rigorous
successor to a toy Kaggle pairwise-classification project. Instead of "which
of these two texts is real" on ~100 labeled examples, this trains and
evaluates a detector on a 400K+ row, 14-domain, 30+ generator benchmark
(MAGE), and — the part almost no student project does — measures whether it
**actually generalizes** to unseen domains, unseen LLMs, and adversarial
paraphrase/obfuscation attacks, rather than just reporting a single
in-distribution accuracy number.

## Why this exists

The starting point was [`FakeTextDetector-NLP`](https://github.com/Sonith-Bingi/FakeTextDetector-NLP),
a solution to Kaggle's "Fake or Real: The Impostor Hunt" (a RoBERTa cross-encoder
+ LightGBM blend on stylometric/perplexity features, trained on a few dozen
labeled text pairs). That pipeline is architecturally reasonable but the
*problem itself* is toy-scale: tiny dataset, one narrow domain, no check for
whether the model generalizes beyond exactly what it was trained on.

This project keeps the same spirit (encoder + zero-shot statistical signal +
stylometry, blended and calibrated) but rebuilds it around a real research
benchmark and adds the engineering and evaluation rigor a production system
would actually need.

## Architecture

```mermaid
flowchart TD
    A[MAGE dataset<br/>435K rows, 14 domains, 30+ generators] --> B[Split builder]
    B --> C[train<br/>seen domains + weak/old generators]
    B --> D[test_in_distribution]
    B --> E[test_cross_domain<br/>held-out domains]
    B --> F[test_cross_generator<br/>held-out GPT-3.5/davinci]
    B --> G[test_gpt4_extension<br/>unseen domain+generator, paraphrase attacks]

    C --> H[Stylometric features]
    C --> I[Statistical zero-shot detectors<br/>log-lik / log-rank / LRR / Fast-DetectGPT]
    C --> J[DeBERTa-v3-small + LoRA<br/>5-fold CV encoder]

    H --> K[LightGBM meta-learner]
    I --> K
    J --> K
    K --> L[Isotonic calibration]
    L --> M[FastAPI service]
    L --> N[Gradio demo]

    D --> O[Generalization + calibration eval]
    E --> O
    F --> O
    G --> O
    G --> P[Paraphrase-attack robustness eval]
    Q[RAID benchmark sample<br/>streamed, surface-level attacks] --> R[Attack robustness eval]
```

## What makes this "advanced" rather than a bigger toy

1. **Held-out generalization by construction, not luck.** `train` only ever
   sees 8 of MAGE's 10 original domains and excludes the strongest
   instruction-tuned generators (gpt-3.5-turbo, text-davinci-002/003). Every
   evaluation slice is drawn exclusively from MAGE's own `test` partition, so
   there's no document-level leakage. See `src/forensics/data/splits.py`.
2. **A genuinely multi-signal ensemble, not just a bigger transformer.** The
   encoder, the statistical zero-shot detectors (which need zero training
   data), and the stylometric features fail in different, largely
   uncorrelated ways — that's the actual reason to blend them, not just
   "more features are better."
3. **Two independent adversarial robustness checks**, not zero: MAGE's own
   GPT-4-paraphrase subset (semantic paraphrase evasion) and a curated,
   streamed sample from the RAID benchmark (Dugan et al., 2024) covering
   character/surface-level obfuscation attacks (homoglyphs, whitespace
   insertion, synonym substitution, case scrambling, ...).
4. **Calibration, not just accuracy.** Isotonic regression on out-of-fold
   predictions + Expected Calibration Error, because "70% confidence" should
   mean 70% empirically, not just rank-order correctly.
5. **Interpretability**: SHAP feature attribution on the blender (which
   signal family actually drove a prediction) and gradient-based token
   saliency on the encoder (which words/spans it focused on).
6. **Production surface**: FastAPI service, Docker/Compose, GitHub Actions CI,
   pytest suite, and a Gradio demo — not just a training notebook.

## Modeling stack

| Component | What it is | Why |
|---|---|---|
| **Encoder** | `microsoft/deberta-v3-small` + LoRA (peft), 5-fold CV | LoRA trains ~0.2% of params, fast enough on Apple MPS for full CV in minutes; fold adapters ensemble for free at inference |
| **Statistical detectors** | Log-likelihood, log-rank, LRR ([Su et al. 2023](https://arxiv.org/abs/2306.05540)), Fast-DetectGPT curvature ([Bao et al. 2024](https://arxiv.org/abs/2310.05130)) via `distilgpt2` | Zero-shot, training-free, mechanistically different failure modes than the encoder |
| **Stylometric features** | Entropy, burstiness, sentence-length variance, repetition rate, readability, TTR | Cheap, fast, and still informative even when the other two disagree |
| **Meta-learner** | LightGBM, 5-fold CV, stacked on encoder logit + all features | Combines signals; isotonic calibration on OOF predictions |

## Dataset

[MAGE](https://huggingface.co/datasets/yaful/MAGE) (Li et al., 2024): 435K
deduplicated rows across 10 original domains (Reddit CMV/ELI5, HellaSwag,
ROCStories, SciGen, SQuAD, TLDR, WritingPrompts, XSum, Yelp) × ~28 generators
(GPT-2/3/3.5/davinci sizes, LLaMA 7-65B, OPT family, GPT-J/NeoX, T0, FLAN-T5,
BLOOM, GLM-130B), plus a bonus test-only extension: 4 more domains (CNN,
DialogSum, IMDB, PubMed) generated by GPT-4, including GPT-4-paraphrased
variants of both human and machine text.

Adversarial robustness stretch goal additionally streams a curated ~1,000-row
sample from [RAID](https://raid-bench.xyz/) (Dugan et al., 2024) covering 9
attack types on generators MAGE never included (gpt2, mistral, mpt-chat) —
pulled via HuggingFace's streaming reader so the full 11.8GB file is never
downloaded.

## Project layout

```
src/forensics/
  config.py                  central config (paths, seeds, hyperparameters)
  data/                       MAGE download+schema, split builder, RAID sampler
  features/                   stylometric + statistical detector feature extraction
  models/                     LoRA encoder, LightGBM blender + calibration
  evaluation/                 metrics, calibration (ECE), generalization suite,
                               paraphrase-attack + RAID-attack robustness
  interpret/                  SHAP explanations, token saliency
  serving/                    FastAPI app + pydantic schemas
  inference.py                shared predict() used by API and demo
  pipeline.py                 disk-cached end-to-end orchestration
scripts/                      CLI entrypoints (download_data / train / evaluate / interpret)
demo/app.py                   Gradio demo
docker/                       Dockerfile + docker-compose.yml
tests/                        pytest suite
.github/workflows/ci.yml      lint + test on push
```

## Running it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export PYTHONPATH=src
python scripts/download_data.py   # MAGE + RAID sample, cached to data/
python scripts/train.py           # features -> 5-fold encoder CV -> blender + calibration
python scripts/evaluate.py        # generalization + calibration + both robustness suites
python scripts/interpret.py       # SHAP global feature importance
python demo/app.py                # Gradio demo at http://localhost:7860

uvicorn forensics.serving.api:app --reload   # API at http://localhost:8000
# or: docker compose -f docker/docker-compose.yml up --build
```

Every expensive stage (feature extraction, encoder training, encoder/blender
inference) is disk-cached under `artifacts/` and `data/`, so re-running any
script after an interruption resumes rather than recomputing from scratch.

## Results

_Populated by `scripts/evaluate.py` — see `artifacts/results/full_evaluation_report.json`
for the full numbers once training completes._

<!-- RESULTS_PLACEHOLDER -->

## Honest limitations

- Trained on a laptop-scale subsample (20K rows) of MAGE for iteration speed,
  not the full 319K-row training partition — accuracy would improve with the
  full set and more compute.
- The statistical detectors use `distilgpt2` as the scoring model; larger
  scoring models generally improve Fast-DetectGPT-style curvature separation.
- RAID robustness numbers come from a ~1,000-row streamed sample, not RAID's
  full benchmark protocol or leaderboard submission format.
- This detects *distributional* signatures of LLM generation; it is not
  forensic proof for any single document, and determined adversaries (heavy
  human editing, adversarial fine-tuning) will erode accuracy further than
  what's measured here.
