"""REINFORCE (policy-gradient with a moving-average baseline) training loop for
the adversarial paraphraser.

Deliberately NOT using TRL's PPOTrainer: this runs unsupervised overnight, and
depending on a fast-moving RL library's exact API/version behavior for that is
a risk not worth taking. A single scalar EMA baseline is simpler than a full
critic network -- no separate value model to load, less memory, one fewer
thing that can silently diverge -- and is a completely standard, legitimate
variance-reduction technique for policy gradients in its own right, not a
watered-down substitute for "real" RL.

Every design choice in this file serves one priority, ranked above training
quality: never lose more than a few minutes of progress to a crash, and always
exit cleanly with a usable checkpoint no matter what happens overnight.
  - try/except around every single step body -- one bad batch (OOM, a
    pathological generation) skips and continues rather than killing the run.
  - checkpoint (adapter + optimizer + step count + baseline) every
    `checkpoint_every` steps, and resume from it automatically on restart.
  - a hard wall-clock budget checked every iteration, independent of step
    count, so the run always terminates on its own and leaves a final report.
  - explicit gc.collect() + MPS cache clearing every step to prevent the slow
    memory creep a long-running generate/forward/backward loop is prone to.
"""
from __future__ import annotations

import gc
import json
import time
import traceback

import pandas as pd
import torch
import torch.nn.functional as F

from forensics.adversarial.paraphraser import build_policy_model, get_tokenizer, load_policy_model, make_prompt
from forensics.adversarial.reward import compute_reward
from forensics.config import ARTIFACTS_DIR, CFG, DEVICE, set_seed

ADV_DIR = ARTIFACTS_DIR / "adversarial"
CKPT_DIR = ADV_DIR / "checkpoint"
LOG_PATH = ADV_DIR / "train_log.jsonl"
STATE_PATH = ADV_DIR / "state.json"
ADV_DIR.mkdir(parents=True, exist_ok=True)


def _log(record: dict) -> None:
    record["ts"] = time.time()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def _load_training_pool() -> list[str]:
    """Machine-generated texts to paraphrase. Drawn from the *train* split
    deliberately -- this is downstream use of an already-trained detector to
    generate an adversarial dataset, not new signal leaking into the
    detector's own train/test evaluation. The held-out final before/after
    evaluation (evaluate_adversary.py) uses a separate eval split instead."""
    df = pd.read_parquet(ARTIFACTS_DIR.parent / "data" / "processed" / "train.parquet")
    texts = df[df["is_machine"] == 1]["text"].tolist()
    return [t[:600] for t in texts]  # cap length for speed/memory


def _cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _save_checkpoint(model, optimizer, step: int, baseline: float, start_time: float):
    model.save_pretrained(CKPT_DIR)
    torch.save(optimizer.state_dict(), ADV_DIR / "optimizer.pt")
    with open(STATE_PATH, "w") as f:
        json.dump(
            {"step": step, "baseline": baseline, "elapsed_seconds": time.time() - start_time},
            f,
        )


def _try_resume():
    if not (CKPT_DIR.exists() and STATE_PATH.exists()):
        return None
    with open(STATE_PATH) as f:
        state = json.load(f)
    model = load_policy_model(CKPT_DIR, is_trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.adversarial.lr)
    opt_path = ADV_DIR / "optimizer.pt"
    if opt_path.exists():
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE))
        except Exception:
            pass  # stale/incompatible optimizer state -- fine to restart momentum, not worth failing over
    return model, optimizer, state["step"], state["baseline"], state.get("elapsed_seconds", 0.0)


def reinforce_step(model, tok, texts: list[str]) -> dict:
    cfg = CFG.adversarial
    prompts = [make_prompt(t) for t in texts]
    enc = tok(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_input_len
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        gen_ids = model.generate(
            **enc, max_new_tokens=cfg.max_new_tokens, do_sample=True, top_p=0.9, temperature=1.0
        )
    paraphrases = tok.batch_decode(gen_ids, skip_special_tokens=True)

    rewards = [compute_reward(orig, para)["total"] for orig, para in zip(texts, paraphrases)]
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

    model.train()
    pad_id = tok.pad_token_id
    mask = (gen_ids != pad_id).float()
    gen_ids_safe = gen_ids.clone()
    gen_ids_safe[gen_ids == pad_id] = 0

    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=gen_ids)
    logp = F.log_softmax(out.logits, dim=-1)
    token_logp = logp.gather(-1, gen_ids_safe.unsqueeze(-1)).squeeze(-1)
    sum_logp = (token_logp * mask).sum(dim=1)

    return {
        "sum_logp": sum_logp,
        "reward_tensor": reward_tensor,
        "rewards": rewards,
        "paraphrases": paraphrases,
    }


def train():
    set_seed()
    cfg = CFG.adversarial
    tok = get_tokenizer()

    resumed = _try_resume()
    if resumed is not None:
        model, optimizer, step, baseline, prior_elapsed = resumed
        _log({"event": "resumed", "step": step, "baseline": baseline})
    else:
        model = build_policy_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        step, baseline, prior_elapsed = 0, 0.0, 0.0
        _log({"event": "started_fresh"})

    pool = _load_training_pool()
    _log({"event": "pool_loaded", "n_texts": len(pool)})

    start_time = time.time() - prior_elapsed  # so the wall-clock budget accounts for prior runs
    import random as _random

    while True:
        elapsed = time.time() - start_time
        if elapsed > cfg.max_seconds:
            _log({"event": "time_budget_reached", "elapsed": elapsed, "step": step})
            break
        if step >= cfg.max_steps:
            _log({"event": "step_budget_reached", "step": step})
            break

        try:
            batch_texts = _random.sample(pool, cfg.batch_size)
            result = reinforce_step(model, tok, batch_texts)

            advantage = result["reward_tensor"] - baseline
            loss = -(advantage * result["sum_logp"]).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_mean_reward = float(result["reward_tensor"].mean().item())
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * batch_mean_reward

            _log(
                {
                    "event": "step",
                    "step": step,
                    "loss": float(loss.item()),
                    "mean_reward": batch_mean_reward,
                    "baseline": baseline,
                    "elapsed": elapsed,
                    "example_paraphrase": result["paraphrases"][0][:150],
                }
            )

            del result, loss, advantage
            step += 1

        except Exception as e:  # noqa: BLE001 -- overnight run must never die on a single bad batch
            _log({"event": "step_error", "step": step, "error": str(e), "traceback": traceback.format_exc()})

        finally:
            _cleanup()

        if step > 0 and step % cfg.checkpoint_every == 0:
            try:
                _save_checkpoint(model, optimizer, step, baseline, start_time)
                _log({"event": "checkpoint_saved", "step": step})
            except Exception as e:  # noqa: BLE001
                _log({"event": "checkpoint_error", "step": step, "error": str(e)})

    _save_checkpoint(model, optimizer, step, baseline, start_time)
    _log({"event": "finished", "step": step, "baseline": baseline})


if __name__ == "__main__":
    train()
