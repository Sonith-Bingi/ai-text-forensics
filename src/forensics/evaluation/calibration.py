"""Expected Calibration Error + reliability diagrams. A model can have great
accuracy and still be badly calibrated (e.g. always outputting 0.95 or 0.05) --
ECE is the standard metric for catching that."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def plot_reliability_diagram(results: dict[str, tuple[np.ndarray, np.ndarray]], out_path: str) -> None:
    """`results` maps split_name -> (probs, y_true)."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    for name, (probs, y_true) in results.items():
        n_bins = 10
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        xs, ys = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
            if mask.sum() == 0:
                continue
            xs.append(probs[mask].mean())
            ys.append(y_true[mask].mean())
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical fraction machine-generated")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
