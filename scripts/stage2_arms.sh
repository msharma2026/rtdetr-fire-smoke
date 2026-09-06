#!/usr/bin/env bash
# Stage-2 factorial: {COCO, FASDD} starting weights x {freeze=10, freeze=0+bblr}
#
# Answers the two questions this project actually rests on:
#
#   1. Did FASDD pretraining help at all?  (fasdd_* vs coco_* rows)
#      Without this control, a good D-Fire number is indistinguishable from
#      published models trained on D-Fire directly, and the two-stage premise
#      -- ~20 h of stage 1 -- is unsupported.
#
#   2. Was freeze=10 right?  (frz10_* vs bblr_* rows)
#      freeze is the limit case of a backbone LR multiplier (mult=0), and our
#      sweep measured performance FALLING as the multiplier approached zero
#      (0.033 -> 0.4005 vs 0.3 -> 0.4244). freeze=10 sits further along that
#      losing direction than anything we tested.
#
# Everything else is held constant: lr0 3e-5, 30 epochs, batch 16, imgsz 640,
# patience 0 (full curves, so arms stay comparable). Each arm ~2.5 h.
set -u
cd "$HOME/repos/fire_detection" || exit 1
source venv/bin/activate

FASDD="runs/stage1_fasdd/weights/best.pt"
COCO="rtdetr-l.pt"
LOG=/tmp/stage2_arms.log

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_arm() {   # name  weights  freeze  bblr
    local name=$1 w=$2 frz=$3 bblr=$4
    if [ -f "runs/$name/weights/best.pt" ]; then
        say "SKIP $name (already done)"; return 0
    fi
    say "START $name  weights=$w freeze=$frz bblr=$bblr"
    python -u scripts/train.py --stage 2 \
        --weights "$w" --freeze "$frz" --bblr "$bblr" \
        --epochs 30 --patience 0 --workers 4 \
        --name "$name" >> "/tmp/s2_${name}.log" 2>&1
    say "END   $name (exit $?)"
}

# freeze=10 arms: bblr 0 disables the split (freeze already zeroes those grads)
run_arm fasdd_frz10  "$FASDD" 10 0
run_arm coco_frz10   "$COCO"  10 0
# freeze=0 arms: backbone trains at 0.3x, the probed optimum
run_arm fasdd_bblr03 "$FASDD" 0  0.3
run_arm coco_bblr03  "$COCO"  0  0.3

say "=== all arms done; evaluating on D-Fire test (4,306 images) ==="
for n in fasdd_frz10 coco_frz10 fasdd_bblr03 coco_bblr03; do
    if [ -f "runs/$n/weights/best.pt" ]; then
        say "--- eval $n ---"
        python -u scripts/evaluate.py --weights "runs/$n/weights/best.pt" \
            >> "/tmp/s2_eval_${n}.log" 2>&1
        say "eval $n exit $?"
    fi
done
say "ALLDONE"
