#!/usr/bin/env bash
# LR ladder for joint training at ratio=16 -- the winner of the ratio search
# (D-Fire val 0.84693, beating both 11 and 22). lr0=1e-4/bblr=0.3 was never
# validated for this config, only carried over from the sequential D-Fire
# recipe. Same "extend until interior" discipline as fair_control.sh's LR
# ladder and joint_ratio_extend.sh's ratio search -- no fixed grid.
#
# Held constant vs the ratio search: 2 epochs/point (same step-budget
# reasoning as joint_ratio_ladder.sh -- the joint pool at ratio=16 is huge,
# ~333K images/epoch, so 2 epochs here is not under-training relative to the
# D-Fire-only ladder's 10-epoch/969-iter convention), bblr=0.3, freeze=0,
# warm-started from stage1_fasdd/best.pt, data=joint_r16.yaml (already staged).
set -u

ROOT="$HOME/repos/fire_detection"
cd "$ROOT" || exit 1
source venv/bin/activate

LOGDIR="$ROOT/runs/joint_lr_ladder"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/driver.log"
EPOCHS=2
BBLR=0.3
DATA="joint_r16.yaml"
WEIGHTS="runs/stage1_fasdd/weights/best.pt"

# Ordered LR ladder. The sweep starts at indices 1..3 (bracketing the
# carried-over 1e-4) and walks outward if an edge wins.
LADDER=(1e-5 3e-5 1e-4 3e-4 1e-3)
LO_START=1
HI_START=3
MAX_SWEEP=5

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
lrtag() { echo "${1//-/}"; }

best_map50() {
    local csv="runs/$1/results.csv"
    [ -f "$csv" ] || return 1
    awk -F, 'NR>1 && $8+0>m {m=$8+0} END{if(m>0) printf "%.5f", m}' "$csv"
}

train_run() {   # name  lr0
    local name=$1 lr=$2
    if [ -f "runs/$name/weights/best.pt" ]; then
        say "SKIP $name (best.pt exists, mAP50=$(best_map50 "$name"))"
        return 0
    fi
    say "START $name  data=$DATA  lr0=$lr  epochs=$EPOCHS"
    python -u scripts/train.py --stage 2 \
        --data "$DATA" \
        --weights "$WEIGHTS" \
        --freeze 0 --bblr "$BBLR" --lr0 "$lr" --epochs "$EPOCHS" --patience 0 \
        --workers 4 --save-period 0 \
        --name "$name" >> "$LOGDIR/${name}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "runs/$name/weights/best.pt" ]; then
        say "DONE  $name (mAP50=$(best_map50 "$name"))"
    else
        say "FAIL  $name exit=$rc -- see $LOGDIR/${name}.log"
    fi
}

say "waiting for any running train.py/ensemble_eval.py to finish (GPU handoff)..."
while pgrep -f 'scripts/trai[n]\.py' > /dev/null || pgrep -f 'ensemble_eva[l]\.py' > /dev/null; do
    sleep 30
done
say "GPU clear -- starting joint ratio=16 LR ladder"

lo=$LO_START
hi=$HI_START
for i in $(seq "$lo" "$hi"); do
    lr="${LADDER[$i]}"
    train_run "joint_r16_lr$(lrtag "$lr")" "$lr"
done

sweeps=0
while [ "$sweeps" -lt "$MAX_SWEEP" ]; do
    lo_score=$(best_map50 "joint_r16_lr$(lrtag "${LADDER[$lo]}")" 2>/dev/null || echo 0)
    hi_score=$(best_map50 "joint_r16_lr$(lrtag "${LADDER[$hi]}")" 2>/dev/null || echo 0)

    # find current interior winner among tested points
    best_idx=$lo
    best_score=$lo_score
    for i in $(seq "$lo" "$hi"); do
        s=$(best_map50 "joint_r16_lr$(lrtag "${LADDER[$i]}")" 2>/dev/null || echo 0)
        if awk -v a="$s" -v b="$best_score" 'BEGIN{exit !(a>b)}'; then
            best_score=$s
            best_idx=$i
        fi
    done

    if [ "$best_idx" -eq "$lo" ] && [ "$lo" -gt 0 ]; then
        lo=$((lo - 1))
        say "winner is at LOW edge (${LADDER[$best_idx]}) -- extending down to ${LADDER[$lo]}"
        train_run "joint_r16_lr$(lrtag "${LADDER[$lo]}")" "${LADDER[$lo]}"
        sweeps=$((sweeps + 1))
    elif [ "$best_idx" -eq "$hi" ] && [ "$hi" -lt $((${#LADDER[@]} - 1)) ]; then
        hi=$((hi + 1))
        say "winner is at HIGH edge (${LADDER[$best_idx]}) -- extending up to ${LADDER[$hi]}"
        train_run "joint_r16_lr$(lrtag "${LADDER[$hi]}")" "${LADDER[$hi]}"
        sweeps=$((sweeps + 1))
    else
        say "winner (${LADDER[$best_idx]}) is interior or ladder edge exhausted -- stopping"
        break
    fi
done

say "================ JOINT RATIO=16 LR LADDER SUMMARY ================"
for i in $(seq "$lo" "$hi"); do
    lr="${LADDER[$i]}"
    name="joint_r16_lr$(lrtag "$lr")"
    s=$(best_map50 "$name" 2>/dev/null || echo "n/a")
    say "  lr0=$lr  $name  D-Fire_val_mAP50=$s"
done
say "Next: D-Fire TEST + FASDD retention eval for the winning LR, exactly as"
say "was done for ratio=11 and ratio=16 at lr0=1e-4."
say "ALLDONE"
