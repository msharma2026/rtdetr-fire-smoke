#!/usr/bin/env bash
# Logic test for fair_control.sh's sweep_arm ladder. Replicates the loop
# verbatim with train_run stubbed and best_map50 backed by a score table,
# then asserts the chosen LR and the set of points actually run.
set -u

LADDER=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3)
LO_START=1
HI_START=3
MAX_SWEEP=5

lrtag() { echo "${1//-/}"; }

declare -A SCORES
RUN_ORDER=""

say() { :; }   # silence

best_map50() {
    local n=$1
    echo "${SCORES[$n]:-0}"
}

train_run() {
    local name=$1
    RUN_ORDER="$RUN_ORDER $name"
}

SWEEP_BEST_LR=""

sweep_arm() {   # arm_label  weights
    local arm=$1 w=$2
    local lo=$LO_START hi=$HI_START i s best_i best_s count

    for (( i=lo; i<=hi; i++ )); do
        train_run "lrchk_${arm}_$(lrtag "${LADDER[$i]}")" "$w" "${LADDER[$i]}" 10 || true
    done

    while :; do
        best_i=""; best_s=0
        for (( i=lo; i<=hi; i++ )); do
            s=$(best_map50 "lrchk_${arm}_$(lrtag "${LADDER[$i]}")" || echo 0)
            [ -z "$s" ] && s=0
            if awk -v a="$s" -v b="$best_s" 'BEGIN{exit !(a>b)}'; then
                best_s=$s; best_i=$i
            fi
        done
        [ -z "$best_i" ] && return 1

        count=$(( hi - lo + 1 ))
        if [ "$best_i" -eq "$lo" ] && [ "$lo" -gt 0 ] && [ "$count" -lt "$MAX_SWEEP" ]; then
            lo=$(( lo - 1 ))
            train_run "lrchk_${arm}_$(lrtag "${LADDER[$lo]}")" "$w" "${LADDER[$lo]}" 10 || true
            continue
        fi
        if [ "$best_i" -eq "$hi" ] && [ "$hi" -lt $(( ${#LADDER[@]} - 1 )) ] && [ "$count" -lt "$MAX_SWEEP" ]; then
            hi=$(( hi + 1 ))
            train_run "lrchk_${arm}_$(lrtag "${LADDER[$hi]}")" "$w" "${LADDER[$hi]}" 10 || true
            continue
        fi
        break
    done
    SWEEP_BEST_LR="${LADDER[$best_i]}"
    return 0
}

fail=0
check() {  # label  expected_lr  expected_run_count
    local label=$1 exp_lr=$2 exp_n=$3
    local n; n=$(echo "$RUN_ORDER" | wc -w)
    if [ "$SWEEP_BEST_LR" = "$exp_lr" ] && [ "$n" -eq "$exp_n" ]; then
        echo "PASS  $label -> lr=$SWEEP_BEST_LR after $n runs"
    else
        echo "FAIL  $label -> got lr=$SWEEP_BEST_LR after $n runs (want $exp_lr / $exp_n)"
        echo "      runs:$RUN_ORDER"
        fail=1
    fi
}

reset() { SCORES=(); RUN_ORDER=""; SWEEP_BEST_LR=""; }

# --- Scenario 1: the real COCO data. 3e-4 wins at the high edge, extends to
# 1e-3, which scores lower -> stop, winner interior. Expect 4 runs.
reset
SCORES[lrchk_coco_3e5]=0.72556
SCORES[lrchk_coco_1e4]=0.79149
SCORES[lrchk_coco_3e4]=0.80324
SCORES[lrchk_coco_1e3]=0.77000
sweep_arm coco w
check "coco: peak interior after one extension" 3e-4 4

# --- Scenario 2: keeps climbing past 1e-3 -> extends twice, hits MAX_SWEEP=5
reset
SCORES[lrchk_coco_3e5]=0.60
SCORES[lrchk_coco_1e4]=0.70
SCORES[lrchk_coco_3e4]=0.80
SCORES[lrchk_coco_1e3]=0.85
SCORES[lrchk_coco_3e3]=0.90
sweep_arm coco w
check "coco: monotone up, stops at MAX_SWEEP" 3e-3 5

# --- Scenario 3: FASDD-style. 3e-5 wins at the low edge, extends down to
# 1e-5, which is worse -> stop. Expect 4 runs.
reset
SCORES[lrchk_fasdd_3e5]=0.83
SCORES[lrchk_fasdd_1e4]=0.81
SCORES[lrchk_fasdd_3e4]=0.75
SCORES[lrchk_fasdd_1e5]=0.79
sweep_arm fasdd w
check "fasdd: peak interior after extending down" 3e-5 4

# --- Scenario 4: 1e-5 keeps winning -> extends down, hits ladder floor
reset
SCORES[lrchk_fasdd_3e5]=0.70
SCORES[lrchk_fasdd_1e4]=0.60
SCORES[lrchk_fasdd_3e4]=0.50
SCORES[lrchk_fasdd_1e5]=0.75
sweep_arm fasdd w
check "fasdd: stops at ladder floor" 1e-5 4

# --- Scenario 5: winner already interior -> no extension, 3 runs
reset
SCORES[lrchk_coco_3e5]=0.70
SCORES[lrchk_coco_1e4]=0.85
SCORES[lrchk_coco_3e4]=0.80
sweep_arm coco w
check "winner already interior, no extension" 1e-4 3

# --- Scenario 6: everything zero (all runs failed) -> must not hang
reset
sweep_arm coco w && r=0 || r=1
if [ "$r" -eq 1 ]; then echo "PASS  all-failed returns nonzero"; else
    echo "FAIL  all-failed returned 0 (lr=$SWEEP_BEST_LR)"; fail=1; fi

exit $fail
