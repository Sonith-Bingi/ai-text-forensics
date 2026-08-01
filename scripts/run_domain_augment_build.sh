#!/bin/bash
# Supervises scripts/build_domain_augmented_training_data.py with a watchdog,
# not just a restart-on-crash loop -- the failure mode observed in practice
# wasn't a clean crash (which the simple pattern used elsewhere in this repo
# already handles), it was a silent HANG: a TCP connection left in CLOSE_WAIT
# after a network blip/sleep, which requests' per-call timeout did not
# reliably recover from. The process stayed alive (visible in ps) but made
# zero checkpoint progress for 2.5+ hours. A restart-on-exit-code supervisor
# is blind to this since the process never actually exits.
#
# Fix: watch the checkpoint file's mtime. If it goes stale for too long while
# the process is still alive, kill it and restart -- the checkpoint resume
# logic already in build_domain_augmented_training_data.py picks up exactly
# where it left off.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src

CHECKPOINT=data/processed/groq_business_augment.partial.parquet
LOG=artifacts/results/domain_augment_build.log
SUPERVISOR_LOG=artifacts/results/domain_augment_supervisor_log.txt
STALL_TIMEOUT=900   # seconds without checkpoint progress before considered hung
MAX_RESTARTS=30

mkdir -p artifacts/results
restart_count=0

while [ "$restart_count" -lt "$MAX_RESTARTS" ]; do
    echo "$(date): starting build process (restart_count=$restart_count)" >> "$SUPERVISOR_LOG"
    # Reset the staleness clock on every (re)start -- a checkpoint's mtime is
    # left over from whenever it was last written, which could be from a
    # PRIOR run's hang. Without this, restarting after killing a hung process
    # immediately re-reads that same stale mtime and kills the brand-new
    # process within one 30s poll, before it has any chance to make progress
    # (this actually happened: two restarts in the first minute, both firing
    # off the same leftover timestamp). Touching it establishes "now" as the
    # baseline for THIS attempt.
    [ -f "$CHECKPOINT" ] && touch "$CHECKPOINT"
    python -u scripts/build_domain_augmented_training_data.py >> "$LOG" 2>&1 &
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
        echo "$(date): clean exit -- build finished, stopping supervisor" >> "$SUPERVISOR_LOG"
        break
    fi

    restart_count=$((restart_count + 1))
    echo "$(date): non-zero exit or killed hang, restarting in 5s (attempt $restart_count/$MAX_RESTARTS)" >> "$SUPERVISOR_LOG"
    sleep 5
done

echo "$(date): supervisor loop ended (restart_count=$restart_count)" >> "$SUPERVISOR_LOG"
