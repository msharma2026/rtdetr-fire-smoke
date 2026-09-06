#!/usr/bin/env bash
# Supervisor for the stage-1 run. Attaches to an ALREADY-RUNNING job without
# disturbing it: it only ever acts when no train.py process exists.
#
# Motivation: the 80-epoch run died at epoch 33 with "CUDA error: unknown error"
# (most likely WSL2 /dev/dxg context loss -- no reboot, no driver update, no
# TDR event) and then sat idle for 7.4 h until noticed. save_period/last.pt
# meant nothing was lost except time. This turns that into a ~1 min gap.
#
# Stops permanently on: PAUSE file, 80 epochs reached, or MAX_FAILS restarts.
set -u

ROOT="$HOME/repos/fire_detection"
CSV="$ROOT/runs/stage1_fasdd/results.csv"
LOG="/tmp/stage1_supervisor.log"
TARGET=80
MAX_FAILS=10
fails=0

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "supervisor started (target ${TARGET} epochs, max ${MAX_FAILS} restarts)"

while true; do
    sleep 120

    if [ -f "$ROOT/PAUSE" ]; then
        say "PAUSE present -- supervisor exiting, will not restart"
        exit 0
    fi

    epochs=0
    [ -f "$CSV" ] && epochs=$(( $(wc -l < "$CSV") - 1 ))
    if [ "$epochs" -ge "$TARGET" ]; then
        say "reached ${epochs}/${TARGET} epochs -- done, exiting"
        exit 0
    fi

    if pgrep -f "scripts/train.py --stage 1" > /dev/null; then
        continue
    fi

    fails=$(( fails + 1 ))
    if [ "$fails" -gt "$MAX_FAILS" ]; then
        say "FAILED ${MAX_FAILS} times, giving up at epoch ${epochs}"
        exit 1
    fi

    say "no train.py alive at epoch ${epochs}/${TARGET} -- restart #${fails}"
    cd "$ROOT" || exit 1
    # shellcheck disable=SC1091
    source venv/bin/activate
    setsid nohup python -u scripts/train.py --stage 1 --patience 0 \
        --save-period 10 --resume >> "/tmp/stage1_auto_${fails}.log" 2>&1 < /dev/null &
    disown
    say "relaunched (log /tmp/stage1_auto_${fails}.log), waiting 10 min before next check"
    sleep 600
done
