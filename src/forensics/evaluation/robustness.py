"""Adversarial robustness: does an attacker who paraphrases or lightly perturbs
machine-generated text fool the detector? Two complementary attack families:

  1. MAGE's built-in paraphrase attack (no extra download): the `test_gpt4_extension`
     split already contains GPT-4-paraphrased versions of both human and machine
     text (`style` in {paraphrase_of_human, paraphrase_of_machine}). This directly
     tests semantic-preserving LLM paraphrase evasion, which is the attack that
     matters most in practice (it's cheap, and it's what actual evaders use).

  2. RAID's character/surface-level attacks (misspelling, whitespace, homoglyphs,
     synonym substitution, ...) via a small subsampled pull from the RAID
     benchmark (Dugan et al., 2024) -- see evaluation/raid_robustness.py.

Reporting the *accuracy drop* from clean to attacked text is the point: a
detector that looks great in Table 1 but collapses under attack is not
production-ready, and most student projects never check this at all.
"""
from __future__ import annotations

import pandas as pd

from forensics.evaluation.metrics import classification_report_dict
from forensics.evaluation.generalization import score_split


def paraphrase_attack_report(gpt4_df: pd.DataFrame, gpt4_feats: pd.DataFrame, gpt4_logits) -> dict:
    """Splits the GPT-4 extension slice into clean vs. paraphrased and compares
    detector accuracy on each, holding domain fixed."""
    results = {}

    def _score_style_group(styles: list[str], label: str):
        mask = gpt4_df["style"].isin(styles).values
        sub_df = gpt4_df[mask].reset_index(drop=True)
        sub_feats = gpt4_feats[mask].reset_index(drop=True)
        sub_logits = gpt4_logits[mask]
        # Single-class groups (e.g. "machine only") reduce accuracy to a plain
        # detection rate, which is exactly what we want to report for an
        # evasion-style attack slice.
        probs = score_split(sub_df, sub_feats, sub_logits)
        y = sub_df["is_machine"].values
        report = classification_report_dict(y, probs)
        results[label] = report
        print(f"[{label}] n={report['n']} acc={report['accuracy']:.4f} f1={report['f1']:.4f}")
        return report

    # Machine text, clean vs. paraphrased -- both should be predicted "machine".
    _score_style_group(["direct"], "gpt4_direct_machine_only")
    _score_style_group(["paraphrase_of_machine"], "gpt4_paraphrased_machine_evasion")
    # Human text, clean vs. paraphrased -- paraphrased-human is labeled machine
    # (an LLM touched it), so this measures "does light LLM editing get caught".
    _score_style_group(["human"], "human_clean")
    _score_style_group(["paraphrase_of_human"], "human_llm_paraphrased")

    return results
