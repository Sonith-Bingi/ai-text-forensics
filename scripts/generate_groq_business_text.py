#!/usr/bin/env python
"""Generates the Groq business-register machine-text rows in complete
isolation from the `datasets` library (used elsewhere for streaming the
Enron corpus). Split out from build_domain_augmented_training_data.py as a
direct test of a specific hypothesis: every restart of the combined script
re-ran the Enron `datasets.load_dataset(streaming=True)` fetch before
resuming Groq generation, and generation reliably hung shortly after,
every single time, regardless of how long the process had otherwise been
running -- while a fresh, standalone `requests` call made moments earlier
(no `datasets` import in that process at all) worked instantly. A
`multiprocessing.resource_tracker: leaked semaphore` warning appeared on
every run, which is the kind of residual state a library's internal
multiprocessing/threading can leave behind and that can interfere with
unrelated I/O afterward, particularly on macOS. This script never imports
`datasets` at all, to remove that variable entirely.

Only generates the Groq rows; combining with Enron + prior training data
happens in build_domain_augmented_training_data.py's second pass, which
skips regenerating anything this script already checkpointed.
"""
from __future__ import annotations

import os
import random
import time

import pandas as pd
import requests

from forensics.config import PROCESSED_DIR, SEED

N_MACHINE_BUSINESS = 3500
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CHECKPOINT_EVERY = 100  # finer-grained than before, so a hang loses less progress
REQUEST_INTERVAL_SECONDS = 2.1

PROMPT_TEMPLATES = [
    "Write a short professional email (3-5 sentences) to a colleague about {topic}. No subject line, just the email body.",
    "Write a brief internal business memo (3-5 sentences) about {topic}.",
    "Write a short status update (3-5 sentences) for a work project related to {topic}.",
    "Write a short, casual Slack-style message to a coworker about {topic}.",
    "Write 3-5 sentences of commentary on quarterly or financial reporting related to {topic}.",
]
TOPICS = [
    "a delayed shipment", "quarterly revenue numbers", "a client meeting rescheduling",
    "a budget overrun", "a new hire starting next week", "a server outage last night",
    "vendor contract renewal", "an upcoming audit", "office relocation plans",
    "a marketing campaign's performance", "headcount planning for next year",
    "a pricing change for customers", "an internal policy update", "a product launch delay",
    "expense report approvals", "a customer complaint escalation", "team offsite planning",
    "year-end inventory counts", "a compliance training deadline", "a merger rumor",
]

CHECKPOINT_PATH = PROCESSED_DIR / "groq_business_augment.partial.parquet"


class DailyQuotaExhausted(Exception):
    """Raised on a 429 whose body says 'tokens per day' -- retrying with
    backoff is pointless here (confirmed the hard way: every prior "hang" in
    this script was actually 5 retries x up to 30s backoff, repeated forever,
    silently, on a 429 that will not clear for ~24h -- indistinguishable from
    a real hang from the outside, since 429s don't go through the exception
    handler below and so never printed anything). A per-minute (RPM) 429 is
    still worth the normal retry/backoff; only the daily quota is fatal for
    the rest of this run."""


def _groq_generate(key: str, prompt: str, max_retries: int = 5) -> str | None:
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}", "Connection": "close"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 1.0,
                },
                timeout=(10, 30),
            )
            if resp.status_code == 429:
                if "tokens per day" in resp.text.lower() or "TPD" in resp.text:
                    raise DailyQuotaExhausted(resp.text[:300])
                time.sleep(min(2 ** attempt, 30))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            print(f"    request error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}", flush=True)
            if attempt == max_retries - 1:
                return None
            time.sleep(min(2 ** attempt, 30))
    return None


def main():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    # Bounded per-invocation target (defaults to the full N). Originally added
    # while chasing what looked like a process hang; the real cause turned
    # out to be a daily token quota 429 that the old retry logic treated as
    # transient (see DailyQuotaExhausted above) -- keeping this bounded-batch
    # option regardless since running in smaller chunks is harmless and makes
    # partial progress easier to reason about.
    target = int(os.environ.get("GROQ_BATCH_TARGET", N_MACHINE_BUSINESS))
    target = min(target, N_MACHINE_BUSINESS)

    rng = random.Random(SEED)
    rows: list[dict] = []
    if CHECKPOINT_PATH.exists():
        rows = pd.read_parquet(CHECKPOINT_PATH).to_dict("records")
        print(f"Resuming from {len(rows)}/{N_MACHINE_BUSINESS} (this batch's target: {target})", flush=True)
    # Burn the RNG forward to where a fresh process would have left off, so
    # resumed runs don't just replay the same prompts a from-scratch run
    # already used for the rows that are now checkpointed.
    for _ in range(len(rows)):
        rng.choice(PROMPT_TEMPLATES)
        rng.choice(TOPICS)

    last_checkpoint_count = len(rows)
    try:
        while len(rows) < target:
            t0 = time.time()
            template = rng.choice(PROMPT_TEMPLATES)
            topic = rng.choice(TOPICS)
            prompt = template.format(topic=topic)
            text = _groq_generate(key, prompt)
            if text:
                rows.append({"text": text, "is_machine": 1})
            # Gate on having actually grown since the last checkpoint, not just
            # "count is currently a multiple of N" -- that fires on every failed
            # (no-op) iteration once len(rows) happens to sit on a multiple,
            # spamming identical "generated" lines with zero real progress.
            if len(rows) - last_checkpoint_count >= CHECKPOINT_EVERY:
                pd.DataFrame(rows).to_parquet(CHECKPOINT_PATH, index=False)
                print(f"  generated {len(rows)}/{N_MACHINE_BUSINESS}", flush=True)
                last_checkpoint_count = len(rows)
            elapsed = time.time() - t0
            if elapsed < REQUEST_INTERVAL_SECONDS:
                time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
    except DailyQuotaExhausted as e:
        pd.DataFrame(rows).to_parquet(CHECKPOINT_PATH, index=False)
        print(f"DAILY TOKEN QUOTA EXHAUSTED at {len(rows)}/{N_MACHINE_BUSINESS} rows -- "
              f"checkpoint saved, stopping (no point retrying until it resets). Detail: {e}", flush=True)
        raise SystemExit(1)

    pd.DataFrame(rows).to_parquet(CHECKPOINT_PATH, index=False)
    print(f"Done: {len(rows)} rows written to {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    main()
