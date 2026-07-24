#!/bin/bash
# Supervises the overnight REINFORCE adversarial-paraphraser training run.
# The Python process enforces its own wall-clock budget and checkpoints every
# few steps; this wrapper's only job is to recover from a hard crash (segfault,
# OOM-kill by the OS, anything that isn't a clean Python exception) by
# restarting the process, which resumes from the last checkpoint automatically.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src

mkdir -p artifacts/adversarial
SUPERVISOR_LOG=artifacts/adversarial/supervisor_log.txt
STDOUT_LOG=artifacts/adversarial/stdout.log
MAX_RESTARTS=30

restart_count=0
while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    echo "$(date): starting training process (restart_count=$restart_count)" >> "$SUPERVISOR_LOG"
    python -u -m forensics.adversarial.train_reinforce >> "$STDOUT_LOG" 2>&1
    exit_code=$?
    echo "$(date): training process exited with code $exit_code" >> "$SUPERVISOR_LOG"

    if [ "$exit_code" -eq 0 ]; then
        echo "$(date): clean exit -- training finished on its own, stopping supervisor" >> "$SUPERVISOR_LOG"
        break
    fi

    restart_count=$((restart_count + 1))
    echo "$(date): non-zero exit, will restart in 5s (attempt $restart_count/$MAX_RESTARTS)" >> "$SUPERVISOR_LOG"
    sleep 5
done

echo "$(date): supervisor loop ended (restart_count=$restart_count)" >> "$SUPERVISOR_LOG"
