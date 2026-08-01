#!/bin/bash
# Repeats the adversarial evaluation multiple times per experiment to measure
# run-to-run sampling variance (do_sample=True with no fixed seed means each
# run's detection rate is itself a noisy estimate, not a fixed ground truth --
# two "identical" v5 runs gave 72.0% and 86.67%, a 14.7-point swing).
# Sequential, not parallel, to avoid contending for MPS memory across processes.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src

mkdir -p artifacts/results/repeat_eval

for exp in adversarial_v4 adversarial_v5; do
    for i in 1 2 3; do
        out="artifacts/results/repeat_eval/${exp}_run${i}.json"
        log="artifacts/results/repeat_eval/${exp}_run${i}.log"
        if [ -f "$out" ]; then
            echo "$(date): $exp run $i already done, skipping"
            continue
        fi
        echo "$(date): starting $exp run $i"
        ADV_EXPERIMENT="$exp" python -u -c "
from forensics.evaluation.adversarial_eval import run_adversarial_evaluation
from pathlib import Path
report = run_adversarial_evaluation(n_texts=150, batch_size=8, results_path=Path('$out'))
print(report)
" > "$log" 2>&1
        echo "$(date): finished $exp run $i (exit $?)"
    done
done
echo "$(date): all repeat evals done"
