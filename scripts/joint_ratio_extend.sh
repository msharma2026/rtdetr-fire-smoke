#!/usr/bin/env bash
# Extends the joint-training ratio ladder past 11, one step at a time until
# D-Fire val stops climbing -- same "extend until interior" discipline as
# fair_control.sh's LR ladder, rather than pre-committing to a fixed grid.
#
# Ladder so far: ratio 1 -> 0.805, 5.5 -> 0.833, 11 -> 0.844 (D-Fire val).
# Gains are diminishing (+0.028, then +0.011) but still positive at the last
# tested point, so the peak has not been bracketed yet.
set -u

ROOT="$HOME/repos/fire_detection"
cd "$ROOT" || exit 1
source venv/bin/activate

LOGDIR="$ROOT/runs/joint_ladder"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/driver_extend.log"
EPOCHS=2
LR0=1e-4
LADDER=(11 16 22 30)   # 11 already done; walk forward from here
MAX_STEPS=3            # cap: at most 3 NEW points past the existing 11

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

best_map50() {
    local csv="runs/$1/results.csv"
    [ -f "$csv" ] || return 1
    awk -F, 'NR>1 && $8+0>m {m=$8+0} END{if(m>0) printf "%.5f", m}' "$csv"
}

say "waiting for any running train.py to finish (GPU handoff)..."
while pgrep -f 'scripts/trai[n]\.py' > /dev/null; do
    sleep 30
done
say "GPU clear -- extending joint ratio ladder"

prev_score=$(best_map50 "joint_r11_explore")
say "starting point: ratio=11, D-Fire val mAP50=$prev_score"

steps=0
idx=1   # LADDER[0]=11 already done, start extending from LADDER[1]=16
while [ "$idx" -lt "${#LADDER[@]}" ] && [ "$steps" -lt "$MAX_STEPS" ]; do
    ratio="${LADDER[$idx]}"
    name="joint_r${ratio}_explore"

    say "--- staging ratio=$ratio ---"
    python -u scripts/prepare_joint_data.py --ratio "$ratio" >> "$LOGDIR/stage_r${ratio}.log" 2>&1

    if [ -f "runs/$name/weights/best.pt" ]; then
        say "SKIP $name (best.pt exists, D-Fire val mAP50=$(best_map50 "$name"))"
    else
        say "START $name  data=joint_r${ratio}.yaml  epochs=$EPOCHS  lr0=$LR0"
        python -u scripts/train.py --stage 2 \
            --data "joint_r${ratio}.yaml" \
            --weights runs/stage1_fasdd/weights/best.pt \
            --freeze 0 --bblr 0.3 --lr0 "$LR0" --epochs "$EPOCHS" --patience 0 \
            --workers 4 --save-period 0 \
            --name "$name" >> "$LOGDIR/${name}.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ] || [ ! -f "runs/$name/weights/best.pt" ]; then
            say "FAIL  $name exit=$rc -- see $LOGDIR/${name}.log -- stopping extension"
            break
        fi
    fi

    new_score=$(best_map50 "$name")
    say "DONE  $name (D-Fire val mAP50=$new_score, previous best was $prev_score)"

    if awk -v a="$new_score" -v b="$prev_score" 'BEGIN{exit !(a>b)}'; then
        say "  still climbing (+$(awk -v a=\"$new_score\" -v b=\"$prev_score\" 'BEGIN{printf \"%.5f\", a-b}')) -- extending further"
        prev_score="$new_score"
        idx=$((idx + 1))
        steps=$((steps + 1))
    else
        say "  PEAK FOUND: ratio=$ratio did not beat the previous point ($new_score <= $prev_score) -- stopping"
        break
    fi
done

say "================ RATIO EXTENSION SUMMARY (D-Fire val) ================"
say "  ratio=11        joint_r11_explore    0.84428  (from original ladder)"
for ((i=1; i<idx+1 && i<${#LADDER[@]}; i++)); do
    r="${LADDER[$i]}"
    n="joint_r${r}_explore"
    s=$(best_map50 "$n" 2>/dev/null || echo "n/a")
    say "  ratio=$r  $n  $s"
done
say "Remember: retention (FASDD val) has NOT been checked for these new"
say "ratios -- ratio=11 already showed retention dipping from its ratio=5.5"
say "peak (0.8027 -> 0.7948). Check retention on the winner before concluding"
say "anything, exactly as was done for ratio=11."
say "ALLDONE"
