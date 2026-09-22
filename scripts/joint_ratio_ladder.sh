#!/usr/bin/env bash
# Joint-training ratio exploration: does mixing FASDD into the D-Fire
# fine-tune phase reduce the retention cost of sequential training?
#
# DESIGN, and why each choice:
#
#   Warm-start from stage1_fasdd/best.pt, not COCO. A from-scratch joint
#   pretrain over the combined pool would cost as much as stage 1 all over
#   again (potentially more, since the pool is bigger). Warm-starting from the
#   already-converged FASDD checkpoint and continuing on the JOINT pool
#   instead of D-Fire alone is the direct, cheap version of the actual
#   hypothesis: does exposure to both datasets during fine-tuning reduce
#   forgetting relative to D-Fire-only fine-tuning.
#
#   2 epochs per ratio, not 10. The LR ladder's "10 epochs" convention was
#   calibrated on D-Fire alone (~15K images/epoch, ~969 iters). The joint pool
#   is dominated by FASDD (85,783 images) plus oversampled D-Fire, so an
#   "epoch" here is 6-17x more iterations depending on ratio. Reusing 10
#   epochs blindly would cost ~13h for the ratio=5.5 point alone. 2 epochs at
#   these pool sizes gives MORE total gradient steps than the original
#   10-epoch/969-iter D-Fire ladder had (12,660-32,000 vs ~9,687), so this is
#   not under-training relative to that precedent -- it is matching step
#   budget instead of blindly copying an epoch count that meant something
#   different in a different context. That mismatch (assuming a number tuned
#   for one config transfers to another) is the exact failure mode this
#   project corrected three times already for LR, freeze and epoch count.
#
#   lr0=1e-4, bblr=0.3, freeze=0 -- the current best D-Fire recipe, reused
#   here ONLY as a first-pass default. It has NOT been validated for joint
#   training and the ratio ladder result should be treated as a first
#   checkpoint to review, not the final word -- the winning ratio deserves
#   its own short LR ladder before any longer confirmatory run, the same
#   escalation pattern used for stage 2's own recipe.
#
# Waits for any running ensemble_eval.py to finish before touching the GPU,
# so this can be launched immediately and queue behind it automatically.
set -u

ROOT="$HOME/repos/fire_detection"
cd "$ROOT" || exit 1
source venv/bin/activate

LOGDIR="$ROOT/runs/joint_ladder"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/driver.log"
EPOCHS=2
LR0=1e-4
RATIOS=(1 5.5 11)

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for any running ensemble_eval.py to finish (GPU handoff)..."
while pgrep -f 'ensemble_eval\.py' > /dev/null; do
    sleep 30
done
say "GPU clear -- starting joint ratio ladder"

best_map50() {
    local csv="runs/$1/results.csv"
    [ -f "$csv" ] || return 1
    awk -F, 'NR>1 && $8+0>m {m=$8+0} END{if(m>0) printf "%.5f", m}' "$csv"
}

for ratio in "${RATIOS[@]}"; do
    name="joint_r${ratio}_explore"
    say "--- staging ratio=$ratio ---"
    python -u scripts/prepare_joint_data.py --ratio "$ratio" >> "$LOGDIR/stage_r${ratio}.log" 2>&1

    if [ -f "runs/$name/weights/best.pt" ]; then
        say "SKIP $name (best.pt exists, D-Fire val mAP50=$(best_map50 "$name"))"
        continue
    fi

    say "START $name  data=joint_r${ratio}.yaml  epochs=$EPOCHS  lr0=$LR0"
    python -u scripts/train.py --stage 2 \
        --data "joint_r${ratio}.yaml" \
        --weights runs/stage1_fasdd/weights/best.pt \
        --freeze 0 --bblr 0.3 --lr0 "$LR0" --epochs "$EPOCHS" --patience 0 \
        --workers 4 --save-period 0 \
        --name "$name" >> "$LOGDIR/${name}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "runs/$name/weights/best.pt" ]; then
        say "DONE  $name (D-Fire val mAP50=$(best_map50 "$name"))"
    else
        say "FAIL  $name exit=$rc -- see $LOGDIR/${name}.log"
    fi
done

say "================ RATIO LADDER SUMMARY (D-Fire val, not test) ================"
for ratio in "${RATIOS[@]}"; do
    name="joint_r${ratio}_explore"
    s=$(best_map50 "$name" 2>/dev/null || echo "n/a")
    say "  ratio=$ratio  ${name}  best_val_mAP50=$s"
done
say "Compare against the sequential baseline: fair_fasdd/seedvar_fasdd_* runs"
say "at their own D-Fire val numbers before"
say "concluding anything -- this ladder used only 2 epochs and an UNVALIDATED"
say "LR for this config. This is a checkpoint to review, not a final answer."
say "ALLDONE"
