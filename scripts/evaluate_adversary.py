#!/usr/bin/env python
"""Before/after robustness check: does the RL-trained adversarial paraphraser
evade the detector measurably better than the off-the-shelf paraphraser alone?
Requires scripts/train_adversarial (or run_adversarial_training.sh) to have
produced a checkpoint at artifacts/adversarial/checkpoint -- if not found,
this still runs the clean/static-paraphrase comparison and skips the
adversarial condition."""
from forensics.evaluation.adversarial_eval import run_adversarial_evaluation

if __name__ == "__main__":
    run_adversarial_evaluation()
