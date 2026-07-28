#!/bin/bash
# Supervises a detector retrain (scripts/train.py: features -> 5-fold encoder CV
# -> blender + calibration). Every stage now checkpoints incrementally
# (features every 500 rows, encoder CV per fold), so this wrapper's job is
# purely to recover from a hard crash (sandbox restart, OOM-kill, segfault) by
# restarting the process, which resumes from the last checkpoint automatically.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src

mkdir -p artifacts/results
SUPERVISOR_LOG=artifacts/results/detector_supervisor_log.txt
STDOUT_LOG=artifacts/results/detector_stdout.log
MAX_RESTARTS=30

restart_count=0
while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    echo "$(date): starting detector training process (restart_count=$restart_count)" >> "$SUPERVISOR_LOG"
    python -u scripts/train.py >> "$STDOUT_LOG" 2>&1
    exit_code=$?
    echo "$(date): detector training process exited with code $exit_code" >> "$SUPERVISOR_LOG"

    if [ "$exit_code" -eq 0 ]; then
        echo "$(date): clean exit -- detector training finished, stopping supervisor" >> "$SUPERVISOR_LOG"
        break
    fi

    restart_count=$((restart_count + 1))
    echo "$(date): non-zero exit, will restart in 5s (attempt $restart_count/$MAX_RESTARTS)" >> "$SUPERVISOR_LOG"
    sleep 5
done

echo "$(date): supervisor loop ended (restart_count=$restart_count)" >> "$SUPERVISOR_LOG"
