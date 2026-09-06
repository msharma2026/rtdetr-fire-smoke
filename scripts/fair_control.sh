#!/usr/bin/env bash
# Fair-control rerun: fixes the defects that make the stage-2 factorial's
# effect sizes upper bounds rather than estimates.
#
# WRITTEN BY CLAUDE (2026-08-18). Not the project author's code -- edit or
# delete freely. See NOTES_BY_CLAUDE_2026-08-18.md section 3.2.
#
# Defect 1: every arm of the 2026-08-18 factorial peaked at epoch 29 or 30
#           of 30. Nothing had converged, so the gap between arms is whatever
#           it happened to be when the budget ran out.
# Defect 2: all four arms ran at lr0=3e-5, an LR tuned on FASDD-initialised
#           runs. The COCO control was handicapped by its competitor's LR.
#
# --- REVISION 2026-08-18 19:50 ---------------------------------------------
# v1 swept the COCO arm only, assuming 3e-5 was already the FASDD arm's tuned
# LR. The first sweep demolished that assumption: for the COCO arm, 3e-5
# scored 0.72556 against 3e-4's 0.80324 -- a 0.078 spread, larger than the
# entire pretraining effect being measured. Assuming an LR is exactly the
# error this rerun exists to correct, so BOTH arms are now swept, and v1's
# fixed 3-point grid is replaced by a ladder that keeps extending until the
# winner is INTERIOR to the tested range. v1 would have declared 3e-4 the
# COCO winner while 3e-4 sat at the edge of the grid -- "best of the three we
# tried" masquerading as an optimum.
#
# Phase A  LR ladder for each arm, 10 epochs per point, extended toward
#          whichever edge wins until the winner is bracketed (max 5 points).
# Phase B  60 epochs of both arms, each at its own measured best LR.
# Phase C  D-Fire test eval of both, plus FASDD val retention for the FASDD arm.
#
# Held constant: batch 16, imgsz 640, freeze=0, bblr=0.3 (the factorial's
# winning configuration), patience=0 so full curves stay comparable.
#
# Restartable: any run whose weights/best.pt exists is skipped, and a crashed
# run is retried with --resume. Logs live under runs/fair_control/ and NOT
# /tmp, which WSL wipes whenever the VM idles out.
#
# CAVEAT (stated, not fixed): a 10-epoch LR check rewards early-training speed,
# which is not identical to 60-epoch quality. It is used here because the
# alternative -- full-length runs per LR -- costs days. The 3e-5 vs 1e-4 vs
# 3e-4 ordering was stable across all 10 epochs, not just at the endpoint,
# which is the evidence that this proxy is not merely measuring warmup rate.
set -u

ROOT="$HOME/repos/fire_detection"
cd "$ROOT" || exit 1
source venv/bin/activate

LOGDIR="$ROOT/runs/fair_control"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/driver.log"
MAX_RETRY=3

LRCHK_EPOCHS=10
FULL_EPOCHS=60

# Ordered LR ladder. The sweep starts at indices 1..3 and walks outward.
LADDER=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3)
LO_START=1
HI_START=3
MAX_SWEEP=5   # cap on points per arm, so a runaway edge cannot loop forever

FASDD_W="runs/stage1_fasdd/weights/best.pt"
COCO_W="rtdetr-l.pt"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

lrtag() { echo "${1//-/}"; }   # 3e-5 -> 3e5, matching the v1 run names

# best mAP50 seen in a run's results.csv (column 8), empty if unavailable
best_map50() {
    local csv="runs/$1/results.csv"
    [ -f "$csv" ] || return 1
    awk -F, 'NR>1 && $8+0>m {m=$8+0} END{if(m>0) printf "%.5f", m}' "$csv"
}

train_run() {   # name  weights  lr0  epochs
    local name=$1 w=$2 lr=$3 ep=$4
    if [ -f "runs/$name/weights/best.pt" ]; then
        say "SKIP $name (best.pt exists, mAP50=$(best_map50 "$name"))"
        return 0
    fi
    local attempt=1 resume=""
    while [ "$attempt" -le "$MAX_RETRY" ]; do
        say "START $name (attempt $attempt) weights=$w lr0=$lr epochs=$ep"
        # shellcheck disable=SC2086
        python -u scripts/train.py --stage 2 \
            --weights "$w" --freeze 0 --bblr 0.3 \
            --lr0 "$lr" --epochs "$ep" --patience 0 \
            --workers 4 --save-period 10 \
            --name "$name" $resume >> "$LOGDIR/${name}.log" 2>&1
        local rc=$?
        if [ "$rc" -eq 0 ] && [ -f "runs/$name/weights/best.pt" ]; then
            say "DONE  $name (mAP50=$(best_map50 "$name"))"
            return 0
        fi
        say "FAIL  $name exit=$rc attempt=$attempt -- retrying with --resume"
        resume="--resume"
        attempt=$(( attempt + 1 ))
        sleep 30
    done
    say "GIVEUP $name after $MAX_RETRY attempts"
    return 1
}

SWEEP_BEST_LR=""
SWEEP_BEST_SCORE=""

sweep_arm() {   # arm_label  weights
    local arm=$1 w=$2
    local lo=$LO_START hi=$HI_START i s best_i best_s count

    say "--- LR ladder for '$arm' arm: ${LADDER[*]:$lo:$((hi-lo+1))} ---"
    for (( i=lo; i<=hi; i++ )); do
        train_run "lrchk_${arm}_$(lrtag "${LADDER[$i]}")" "$w" "${LADDER[$i]}" "$LRCHK_EPOCHS" || true
    done

    while :; do
        best_i=""; best_s=0
        for (( i=lo; i<=hi; i++ )); do
            s=$(best_map50 "lrchk_${arm}_$(lrtag "${LADDER[$i]}")" || echo 0)
            [ -z "$s" ] && s=0
            say "  [$arm] LR ${LADDER[$i]} -> best mAP50 $s"
            if awk -v a="$s" -v b="$best_s" 'BEGIN{exit !(a>b)}'; then
                best_s=$s; best_i=$i
            fi
        done
        if [ -z "$best_i" ]; then
            say "ABORT: no LR point produced a result for '$arm'"
            return 1
        fi

        count=$(( hi - lo + 1 ))
        if [ "$best_i" -eq "$lo" ] && [ "$lo" -gt 0 ] && [ "$count" -lt "$MAX_SWEEP" ]; then
            lo=$(( lo - 1 ))
            say "  [$arm] winner ${LADDER[$best_i]} is at the LOW edge -- extending down to ${LADDER[$lo]}"
            train_run "lrchk_${arm}_$(lrtag "${LADDER[$lo]}")" "$w" "${LADDER[$lo]}" "$LRCHK_EPOCHS" || true
            continue
        fi
        if [ "$best_i" -eq "$hi" ] && [ "$hi" -lt $(( ${#LADDER[@]} - 1 )) ] && [ "$count" -lt "$MAX_SWEEP" ]; then
            hi=$(( hi + 1 ))
            say "  [$arm] winner ${LADDER[$best_i]} is at the HIGH edge -- extending up to ${LADDER[$hi]}"
            train_run "lrchk_${arm}_$(lrtag "${LADDER[$hi]}")" "$w" "${LADDER[$hi]}" "$LRCHK_EPOCHS" || true
            continue
        fi

        if [ "$best_i" -eq "$lo" ] || [ "$best_i" -eq "$hi" ]; then
            say "  [$arm] WARNING: winner ${LADDER[$best_i]} still at an edge (ladder or MAX_SWEEP exhausted)"
        fi
        break
    done

    SWEEP_BEST_LR="${LADDER[$best_i]}"
    SWEEP_BEST_SCORE="$best_s"
    say "PHASE A winner [$arm]: lr0=$SWEEP_BEST_LR (mAP50 $SWEEP_BEST_SCORE)"
    return 0
}

say "================ PHASE A: LR ladders ($LRCHK_EPOCHS epochs per point) ================"

sweep_arm coco "$COCO_W"  || exit 1
COCO_LR="$SWEEP_BEST_LR"; COCO_LR_SCORE="$SWEEP_BEST_SCORE"

sweep_arm fasdd "$FASDD_W" || exit 1
FASDD_LR="$SWEEP_BEST_LR"; FASDD_LR_SCORE="$SWEEP_BEST_SCORE"

say "PHASE A complete: coco lr0=$COCO_LR ($COCO_LR_SCORE), fasdd lr0=$FASDD_LR ($FASDD_LR_SCORE)"
cat > "$LOGDIR/phase_a_result.json" <<JSON
{
  "coco":  {"best_lr": "$COCO_LR",  "best_map50": $COCO_LR_SCORE},
  "fasdd": {"best_lr": "$FASDD_LR", "best_map50": $FASDD_LR_SCORE},
  "lrchk_epochs": $LRCHK_EPOCHS
}
JSON

say "================ PHASE B: ${FULL_EPOCHS}-epoch arms ================"
train_run fair_fasdd_60ep "$FASDD_W" "$FASDD_LR" "$FULL_EPOCHS"
train_run fair_coco_60ep  "$COCO_W"  "$COCO_LR"  "$FULL_EPOCHS"

say "================ PHASE C: evaluation ================"
for n in fair_fasdd_60ep fair_coco_60ep; do
    if [ -f "runs/$n/weights/best.pt" ]; then
        say "--- D-Fire test eval: $n ---"
        python -u scripts/evaluate.py --weights "runs/$n/weights/best.pt" \
            >> "$LOGDIR/eval_${n}.log" 2>&1
        say "eval $n exit $?"
    else
        say "no weights for $n, skipping eval"
    fi
done

# FASDD retention for the FASDD-initialised arm: how much of stage 1 survived
if [ -f "runs/fair_fasdd_60ep/weights/best.pt" ]; then
    say "--- FASDD val retention: fair_fasdd_60ep ---"
    python -u - <<'PY' >> "$LOGDIR/retention_fair.log" 2>&1
from pathlib import Path
from ultralytics import RTDETR
ROOT = Path.home() / "repos/fire_detection"
m = RTDETR(str(ROOT / "runs/fair_fasdd_60ep/weights/best.pt"))
r = m.val(data=str(ROOT / "data/fasdd.yaml"), split="val", batch=16,
          imgsz=640, device="0", project=str(ROOT / "runs"),
          name="retention_fair_fasdd_60ep", exist_ok=True, plots=False)
print(f"FASDD val mAP50 {r.box.map50:.4f}  mAP50-95 {r.box.map:.4f}")
PY
    say "retention eval exit $?"
fi

say "ALLDONE"
