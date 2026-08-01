# Obstacles & Engineering Decisions

A running log of the real problems hit while building this project, in the
order they happened, with what actually fixed each one. Kept separate from
`README.md` so the README can stay focused on results; this file is the
"how it actually went" record.

---

## 1. DeBERTa-v3 was 100x slower than benchmarked on Apple Silicon

**Plan:** DeBERTa-v3-small, 384-token sequences.
**Problem:** A short synthetic-sentence benchmark looked fine, but DeBERTa-v3's
disentangled attention has no fused kernel on PyTorch's MPS backend — it
never goes through `scaled_dot_product_attention` at all. On real,
full-length documents this projected to **12+ hours** for 5-fold CV.
**Fix:** Switched to `distilroberta-base` (standard attention, hits MPS's
optimized SDPA path). Same CV dropped to ~35 minutes.
**Lesson:** Benchmark training throughput with real, full-length documents
before committing to an architecture on Apple Silicon — a short synthetic
sentence hides attention-kernel costs that only show up at real sequence
lengths.

## 2. float16 silently NaN'd during LoRA training on MPS

`from_pretrained` auto-selects float16 on MPS by default. LoRA training in
that precision NaN'd out partway through. Fixed with an explicit
`torch_dtype=torch.float32`.

## 3. PyTorch + LightGBM/SHAP segfault when sharing a process

Both bundle their own OpenMP runtime. Running LightGBM `train()`/`predict()`
or `shap.TreeExplainer` in a process where `torch` was already imported
segfaults on this machine. `KMP_DUPLICATE_LIB_OK=TRUE` alone does **not**
fix it — `OMP_NUM_THREADS=1`, set process-wide before any other import,
does. No measurable cost at this project's data scale.

## 4. Byte-level BPE saliency output was mojibake

`convert_ids_to_tokens()` output isn't display text for a byte-level BPE
tokenizer. Fixed by switching to `return_offsets_mapping=True` and slicing
the *original string* instead of decoding tokens back to text.

## 5. `PeftModel.from_pretrained` silently breaks resume training

Its default is `is_trainable=False`. Resuming adversarial-training from a
checkpoint with the default failed with "does not require grad and does not
have a grad_fn" — a genuinely confusing error since nothing in the loading
code looks wrong. Fixed by passing `is_trainable=True` explicitly at the
resume call site. **Only found because resume was crash-tested for real**
(a `SIGKILL` mid-training, not just written and assumed to work) before
trusting it for an unsupervised overnight run.

## 6. RL reward hacking — four rounds, four genuinely distinct exploits

Training an adversarial paraphraser (REINFORCE) to evade the detector hit
reward hacking three separate times, each invisible to the previous fix
because each was a structurally different pattern:

| Version | Exploit | Why the reward missed it | Fix |
|---|---|---|---|
| v1 | Incoherent word salad | Lexical-overlap fidelity doesn't require coherence; additive reward let evasion dominate regardless | Embedding-based fidelity (cosine sim) + multiplicative reward (`evasion × fidelity`) |
| v3 | Coherent prefix + repeated-char/CAPS garbage tail | A short garbage tail barely moves *whole-sentence* embedding similarity, so it cleared the v2 fidelity gate | Hard wellformedness gate: zero fidelity for repeated-char runs or consecutive ALL-CAPS words |
| v4 | Repeated-*word* spam (`"tread tread tread..."`) | The repeated-*character* regex only matches unbroken character runs, not a word repeated with spaces between | Added a consecutive-identical-word-run check |
| — | A single-run "v5 beats v4" claim | Two identical-config runs gave 72.0% and 86.7% detection — sampling noise larger than the effect being measured | Repeated both models 3x each; the real difference was within noise (z≈1.62, not significant) |

None of v1's exploit reproduced in v3, and v3's didn't reproduce in v4 —
each was a genuinely different failure mode, only visible by reading actual
generated text, not by watching the aggregate reward metric. Full detail:
`README.md`'s adversarial-experiment history was moved here; see the git
history of this repo for the original blow-by-blow if needed.

## 7. A cheap smoke test that changed the wrong variable

A small (n=40, single seed, unbatched) test suggested dropping inference
temperature from 1.0 to 0.7 would cut the evader's gibberish rate at no
evasion cost. Adopted, then re-checked at full scale (n=150, unseeded,
batched — matching real usage): reversed completely, 0.7 made evasion
*worse* than the static (non-RL) paraphraser. The policy was optimized for
temperature=1.0 rollouts specifically during training; departing from that
at inference broke it. **Lesson:** a smoke test that changes the sampling
procedure instead of the thing actually being tested isn't testing the same
thing at scale.

## 8. A stale-cache bug that made "encoder fold checkpoints" glob-match a `.npy` file

Added per-fold resume checkpointing (`fold{N}_oof.npy`) to the encoder
trainer. `load_fold_models()`'s `ENCODER_DIR.glob("fold*")` matched *both*
the fold directories and these new `.npy` files, crashing evaluation when
`PeftModel.from_pretrained` tried to load a `.npy` as an adapter. Caught by
an end-to-end tiny-pipeline smoke test *before* launching the real 45K-row
retrain — exactly the discipline that was supposed to catch this. Fixed by
filtering to `.is_dir()`.

## 9. A hardening round that looked like a catastrophic regression — and wasn't

Augmented the detector's training data with 1,000 of the RL evader's own
paraphrases and retrained. The re-evaluation showed the hardest
generalization slice apparently collapsing from 71.2% to 47.0% accuracy.
**It wasn't real.** The hardening script's eval step scored the *entire*
`test_gpt4_extension` file (3,162 rows, 4 styles), while the actual
"unseen domain and generator" metric reported everywhere else only uses the
`direct`/`human` subset (1,562 rows) — the other ~1,600 rows are
paraphrase-attack rows already known to score 20-24%. Comparing the two was
comparing different metrics, not measuring a regression. Fixed the script's
filtering and re-measured on the matching subset: the real cost was **≤0.6
points on all four standard slices**, while detection of the evader's own
attacks rose from 79.1% to 92.9% (z≈6.0, p≪0.0001) — a genuine, close-to-free
win once measured correctly.

**A second, more serious issue surfaced in the same round:** `train_blender_cv()`
and `fit_calibrator()` write to the same fixed path serving reads from. The
first hardening run *silently replaced the live production model* before any
of the above was verified — an unreviewed production change that was only
caught because the (wrong) eval numbers looked bad enough to investigate.
Fixed the script to snapshot the pre-hardening blender automatically and
restore it after producing its report, so a hardened model is now always a
side artifact until a human reviews the trade-off and promotes it explicitly.

## 10. RAID's worst two attacks were a wiring bug, not a model weakness

`homoglyph` (Cyrillic/Greek Latin look-alikes) and `zero_width_space`
(invisible U+200B between every character) scored 72.0% and 52.7% —
the two worst numbers in the whole RAID table. A `normalize_text()` function
that strips zero-width characters *already existed* but was only wired into
the stylometric features; the statistical detectors and the encoder's
tokenizer both still scored the raw, attacked text. Extended the function
with a Cyrillic/Greek→Latin confusables map (built from inspecting the
actual cached attack samples, not guessed) and wired it into every
text-consuming path. Needed zero retraining — normalizing already-clean
training text is a no-op. Result: `zero_width_space` 52.7% → 78.7%
(matching the clean baseline), `homoglyph` 72.0% → 78.7%. The other five
RAID attacks moved by ≤1.4 points, confirming the fix was targeted.

## 11. A two-hour "hang" that was actually a daily token quota

Using a Groq API key to generate business-register training text, batches
started stalling indefinitely — process alive, near-zero CPU, no errors, no
network activity when inspected. Chased this for roughly two hours through a
sequence of increasingly specific (and all wrong) hypotheses, each tested
and ruled out directly rather than assumed:

1. *Rate limiting* — added proper request pacing. Didn't fix it.
2. *Stale reused TCP connection* (`CLOSE_WAIT` observed via `lsof`) — added
   `Connection: close` and shorter timeouts. Didn't fix it.
3. *The `datasets` library leaving multiprocessing state behind* (a
   `resource_tracker: leaked semaphore` warning was visible) — rewrote the
   generator as a standalone script with zero `datasets` import. Didn't fix it.
4. *Background execution losing OS scheduling during idle gaps* — tested
   foreground-only, bounded-batch execution. Didn't fix it either — even a
   10-row batch hung.
5. Isolated the exact failure point with an inline reproduction: **the first
   request in a process always succeeded; the second one hung.** Bypassed
   `requests` entirely with raw `http.client`, which raised a normal
   response instead of hanging — revealing the actual server response:
   `429, "tokens per day (TPD): Limit 100000, Used 99967"`.

**The real cause:** a hard daily token quota (100K tokens/day, free tier),
already almost exhausted. The retry logic treated every 429 as a transient,
worth-retrying rate limit — 5 retries × up to 30s backoff, repeated forever,
per prompt, since a *daily* quota doesn't clear on any retry timescale. 429s
never reach the exception handler that had debug logging, so this produced
zero output for the entire multi-minute retry storm — indistinguishable from
a real hang from the outside. Fixed by detecting "tokens per day" in the 429
body and failing fast instead of retrying. **The four earlier hypotheses
weren't wasted motion exactly, but none of them were the actual bug** — the
lesson is to check the response body of a 429 before building retry logic
around it, not after.

**Practical fallout:** 400 real Groq rows had already consumed nearly the
entire daily budget, meaning the original target of 3,500 rows would have
taken over a week of daily resets. Supplemented the remainder by generating
business-register text directly (same model class as everything else in
this pipeline — an LLM writing text on a topic prompt) instead of waiting
out the quota or falling back to a much smaller local model.

## 12. The detector's accuracy is dominated by text length, not domain

After multiple rounds of training-data augmentation aimed at a specific
failure case (a genuine, short, off-domain human sentence scoring 93-96%
"machine"), the case remained wrong. Measuring accuracy against text length
directly on held-out data (rather than continuing to guess at more data
augmentation) showed why:

| Length | Accuracy | Mean confidence |
|---|---|---|
| <30 words | 59.2% | 0.48 |
| 30-60 words | 75.8% | 0.68 |
| 60-100 words | 81.9% | 0.79 |
| 100-150 words | 90.9% | 0.85 |
| 150-250 words | 93.7% | 0.86 |
| 250+ words | 98.0% | 0.95 |

A clean, monotonic relationship: accuracy climbs from barely-better-than-chance
under 30 words to near-perfect above 250. This is a harder limit than domain
coverage — stylometric features (sentence-length variance, burstiness) and
statistical detectors (which average per-token signal) both need enough
text to produce a reliable signal, and no volume of same-length training
data fully overcomes that. The honest fix isn't more training data, it's
surfacing the caveat: the demo and API now flag low-confidence results on
short input instead of returning a bare, overconfident verdict.
