#!/usr/bin/env python
"""Full evaluation suite: in-distribution / cross-domain / cross-generator /
GPT-4-unseen generalization, calibration (ECE + reliability diagram), the
built-in MAGE paraphrase-attack robustness check, and the RAID surface-attack
robustness check. Requires scripts/train.py to have been run first."""
import json

from forensics.config import ARTIFACTS_DIR
from forensics.data.raid import build_raid_sample
from forensics.evaluation.generalization import run_generalization_suite
from forensics.evaluation.raid_robustness import raid_attack_report
from forensics.evaluation.robustness import paraphrase_attack_report
from forensics.pipeline import prepare_features_and_encoder

RESULTS_DIR = ARTIFACTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    state = prepare_features_and_encoder()

    print("\n########## GENERALIZATION SUITE ##########")
    generalization_results = run_generalization_suite(state["splits"], state["features"], state["eval_logits"])

    print("\n########## PARAPHRASE-ATTACK ROBUSTNESS (MAGE GPT-4 extension) ##########")
    gpt4_df = state["splits"]["test_gpt4_extension"]
    gpt4_feats = state["features"]["test_gpt4_extension"]
    gpt4_logits = state["eval_logits"]["test_gpt4_extension"]
    paraphrase_results = paraphrase_attack_report(gpt4_df, gpt4_feats, gpt4_logits)

    print("\n########## SURFACE-ATTACK ROBUSTNESS (RAID sample) ##########")
    raid_df = build_raid_sample()
    raid_results = raid_attack_report(raid_df)

    with open(RESULTS_DIR / "full_evaluation_report.json", "w") as f:
        json.dump(
            {
                "generalization": generalization_results,
                "paraphrase_attack": paraphrase_results,
                "raid_attack": raid_results,
            },
            f,
            indent=2,
        )
    print(f"\nFull report written to {RESULTS_DIR / 'full_evaluation_report.json'}")
