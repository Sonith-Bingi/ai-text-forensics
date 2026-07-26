#!/usr/bin/env python
"""Compares reward-shaping/hyperparameter candidates for the adversarial
paraphraser CHEAPLY (short training runs + a small held-out mini-eval each)
before committing hours to a full run on any single one of them.

Each candidate trains as an isolated subprocess into its own
artifacts/<name>/ directory (via ADV_EXPERIMENT), so nothing here touches the
real "adversarial" checkpoint or collides with another candidate.

Usage: python scripts/smoke_test_adversarial.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "artifacts" / "smoke_test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_STEPS = "200"
SMOKE_CHECKPOINT_EVERY = "20"
SMOKE_MAX_SECONDS = "1800"  # 30 min hard cap per candidate, in case a batch runs slow
MINI_EVAL_N = 30

CANDIDATES = [
    {"name": "smoke_gate_lr1e5", "ADV_REWARD_MODE": "threshold_gate", "ADV_LR": "1e-5"},
    {"name": "smoke_gate_lr3e5", "ADV_REWARD_MODE": "threshold_gate", "ADV_LR": "3e-5"},
    {"name": "smoke_mult_lr3e5", "ADV_REWARD_MODE": "multiplicative", "ADV_LR": "3e-5"},
]


def _env_for(candidate: dict) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["ADV_EXPERIMENT"] = candidate["name"]
    env["ADV_REWARD_MODE"] = candidate["ADV_REWARD_MODE"]
    env["ADV_LR"] = candidate["ADV_LR"]
    env["ADV_MAX_STEPS"] = SMOKE_STEPS
    env["ADV_CHECKPOINT_EVERY"] = SMOKE_CHECKPOINT_EVERY
    env["ADV_MAX_SECONDS"] = SMOKE_MAX_SECONDS
    return env


def run_candidate(candidate: dict) -> dict:
    name = candidate["name"]
    print(f"\n{'=' * 60}\nSmoke testing: {name} ({candidate})\n{'=' * 60}", flush=True)
    env = _env_for(candidate)

    train_proc = subprocess.run(
        [sys.executable, "-u", "-m", "forensics.adversarial.train_reinforce"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    log_tail = "\n".join(train_proc.stdout.strip().splitlines()[-3:])
    print(f"[{name}] training exit={train_proc.returncode}\n{log_tail}", flush=True)
    if train_proc.returncode != 0:
        print(f"[{name}] TRAINING FAILED:\n{train_proc.stderr[-3000:]}", flush=True)
        return {"name": name, "candidate": candidate, "training_ok": False}

    mini_eval_path = RESULTS_DIR / f"{name}_eval.json"
    eval_code = (
        "from forensics.evaluation.adversarial_eval import run_adversarial_evaluation\n"
        f"run_adversarial_evaluation(n_texts={MINI_EVAL_N}, results_path=__import__('pathlib').Path(r'{mini_eval_path}'))\n"
    )
    eval_proc = subprocess.run(
        [sys.executable, "-u", "-c", eval_code], cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if eval_proc.returncode != 0 or not mini_eval_path.exists():
        print(f"[{name}] MINI-EVAL FAILED:\n{eval_proc.stderr[-3000:]}", flush=True)
        return {"name": name, "candidate": candidate, "training_ok": True, "eval_ok": False}

    with open(mini_eval_path) as f:
        eval_report = json.load(f)
    print(f"[{name}] mini-eval: {json.dumps(eval_report['report'], indent=2)}", flush=True)
    return {"name": name, "candidate": candidate, "training_ok": True, "eval_ok": True, "report": eval_report}


def main():
    results = [run_candidate(c) for c in CANDIDATES]

    print(f"\n{'=' * 60}\nSMOKE TEST SUMMARY (n={MINI_EVAL_N} held-out texts each)\n{'=' * 60}")
    for r in results:
        if not r.get("eval_ok"):
            print(f"{r['name']:20s}  FAILED (training_ok={r.get('training_ok')})")
            continue
        adv = r["report"]["report"].get("adversarial_paraphrase", {})
        print(
            f"{r['name']:20s}  detection_rate={adv.get('detection_rate', float('nan')):.3f}  "
            f"mean_prob_machine={adv.get('mean_prob_machine', float('nan')):.3f}"
        )

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
