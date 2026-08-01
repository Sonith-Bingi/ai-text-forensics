#!/bin/bash
# Same watchdog pattern as run_domain_augment_build.sh (touch-on-restart fix
# included), pointed at the datasets-free generate_groq_business_text.py --
# see that script's docstring for why it's isolated.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src

CHECKPOINT=data/processed/groq_business_augment.partial.parquet
LOG=artifacts/results/groq_business_generation.log
SUPERVISOR_LOG=artifacts/results/groq_business_generation_supervisor_log.txt
STALL_TIMEOUT=600
MAX_RESTARTS=50

mkdir -p artifacts/results
restart_count=0

while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    echo "$(date): starting generation process (restart_count=$restart_count)" >> "$SUPERVISOR_LOG"
    [ -f "$CHECKPOINT" ] && touch "$CHECKPOINT"
    python -u scripts/generate_groq_business_text.py >> "$LOG" 2>&1 &
    PID=$!

    while kill -0 "$PID" 2>/dev/null; do
        sleep 30
        if [ -f "$CHECKPOINT" ]; then
            LAST_MOD=$(stat -f %m "$CHECKPOINT" 2>/dev/null || echo 0)
            NOW=$(date +%s)
            AGE=$((NOW - LAST_MOD))
            if [ "$AGE" -gt "$STALL_TIMEOUT" ]; then
                echo "$(date): checkpoint stale for ${AGE}s, killing hung process pid=$PID" >> "$SUPERVISOR_LOG"
                kill -9 "$PID" 2>/dev/null
                break
            fi
        fi
    done

    wait "$PID" 2>/dev/null
    exit_code=$?
    echo "$(date): process exited with code $exit_code" >> "$SUPERVISOR_LOG"

    if [ "$exit_code" -eq 0 ]; then
        echo "$(date): clean exit -- generation finished, stopping supervisor" >> "$SUPERVISOR_LOG"
        break
    fi

    restart_count=$((restart_count + 1))
    echo "$(date): non-zero exit or killed hang, restarting in 5s (attempt $restart_count/$MAX_RESTARTS)" >> "$SUPERVISOR_LOG"
    sleep 5
done

echo "$(date): supervisor loop ended (restart_count=$restart_count)" >> "$SUPERVISOR_LOG"
