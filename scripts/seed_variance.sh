#!/usr/bin/env bash
# Run-to-run variance: 3 seeds x 2 arms, 20 epochs each, D-Fire test eval.
#
# WRITTEN BY CLAUDE (2026-08-19). Not the project author's code -- edit or
# delete freely. See NOTES_BY_CLAUDE_2026-08-18.md section 8.2.
#
# WHY: every number in this project is a single seed, so no comparison has an
# error bar. The fair-control rerun measured a +0.0328 mAP50 pretraining
# effect; `freeze=0` over `freeze=10` was +0.028; the retention edge that
# justifies the whole third-party-dataset plan is +0.0195. If run-to-run sigma
# is ~0.02, the second and third are noise and that work should be cancelled.
# This measures sigma once so every later comparison is interpretable.
#
# DESIGN
#   * 20 epochs: both arms are converged well before this (best epochs were 30
#     and 29, but val was flat from ~10 and ~18 respectively), and it keeps the
#     whole job to ~7.6 h. See notes 2.2.
#   * Each arm at ITS OWN laddered LR -- fasdd 1e-4, coco 3e-4 (notes 2.5).
#     Everything else held at the factorial's winning config.
#   * Seeds vary head init, data order, and augmentation sampling. train.py
#     previously hardcoded seed=0; a --seed flag was added for this.
#   * Arms are INTERLEAVED (fasdd s0, coco s0, fasdd s1, ...) so that an
#     interrupted job still leaves complete pairs rather than three of one arm.
#   * deterministic=False is left as-is, so this measures seed variance plus
#     GPU nondeterminism -- the practically relevant quantity, and the
#     conservative (larger) of the two.
#
# CAVEAT: n=3 gives a 2-dof sigma estimate, roughly a [0.5x, 2.9x] 95%
# interval. It reliably separates sigma~0.005 from sigma~0.02; it does NOT
# certify a tight CI on +0.0328. If a publication-grade interval is needed,
# extend SEEDS to 0..4 (~12.6 h).
set -u

ROOT="$HOME/repos/fire_detection"
cd "$ROOT" || exit 1
source venv/bin/activate

LOGDIR="$ROOT/runs/seed_variance"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/driver.log"
MAX_RETRY=3
EPOCHS=20
SEEDS=(0 1 2)

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

train_run() {   # name  weights  lr0  seed
    local name=$1 w=$2 lr=$3 sd=$4
    if [ -f "runs/$name/weights/best.pt" ]; then
        say "SKIP $name (best.pt exists)"; return 0
    fi
    local attempt=1 resume=""
    while [ "$attempt" -le "$MAX_RETRY" ]; do
        say "START $name (attempt $attempt) lr0=$lr seed=$sd epochs=$EPOCHS"
        # shellcheck disable=SC2086
        python -u scripts/train.py --stage 2 \
            --weights "$w" --freeze 0 --bblr 0.3 \
            --lr0 "$lr" --epochs "$EPOCHS" --patience 0 --seed "$sd" \
            --workers 4 --save-period 10 \
            --name "$name" $resume >> "$LOGDIR/${name}.log" 2>&1
        local rc=$?
        if [ "$rc" -eq 0 ] && [ -f "runs/$name/weights/best.pt" ]; then
            say "DONE  $name"; return 0
        fi
        say "FAIL  $name exit=$rc attempt=$attempt -- retrying with --resume"
        resume="--resume"; attempt=$(( attempt + 1 )); sleep 30
    done
    say "GIVEUP $name"; return 1
}

eval_run() {    # name
    local name=$1
    [ -f "$LOGDIR/eval_${name}.log" ] && grep -q 'mAP50-95' "$LOGDIR/eval_${name}.log" \
        && { say "SKIP eval $name (already evaluated)"; return 0; }
    if [ ! -f "runs/$name/weights/best.pt" ]; then
        say "no weights for $name, skipping eval"; return 1
    fi
    say "--- D-Fire test eval: $name ---"
    python -u scripts/evaluate.py --weights "runs/$name/weights/best.pt" \
        >> "$LOGDIR/eval_${name}.log" 2>&1
    say "eval $name exit $?"
}

say "================ SEED VARIANCE: ${#SEEDS[@]} seeds x 2 arms, $EPOCHS epochs ================"

for sd in "${SEEDS[@]}"; do
    train_run "seedvar_fasdd_s${sd}" "runs/stage1_fasdd/weights/best.pt" 1e-4 "$sd"
    eval_run  "seedvar_fasdd_s${sd}"
    train_run "seedvar_coco_s${sd}"  "rtdetr-l.pt"                       3e-4 "$sd"
    eval_run  "seedvar_coco_s${sd}"
    say "=== seed $sd pair complete ==="
done

say "================ AGGREGATING ================"
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json, re, statistics as st
from pathlib import Path

LOGDIR = Path.home() / "repos/fire_detection/runs/seed_variance"
SEEDS = [0, 1, 2]
pat = re.compile(r"mAP50-95\s+([0-9.]+)\s+mAP50\s+([0-9.]+)")

def read(name):
    f = LOGDIR / f"eval_{name}.log"
    if not f.exists():
        return None
    m = None
    for line in f.read_text(errors="ignore").splitlines():
        hit = pat.search(line)
        if hit:
            m = hit
    return (float(m.group(2)), float(m.group(1))) if m else None

arms = {}
for arm in ("fasdd", "coco"):
    rows = [(s, read(f"seedvar_{arm}_s{s}")) for s in SEEDS]
    got = [(s, v) for s, v in rows if v]
    arms[arm] = got
    print(f"\n{arm} arm:")
    for s, (m50, m) in got:
        print(f"  seed {s}: mAP50 {m50:.4f}  mAP50-95 {m:.4f}")
    if len(got) >= 2:
        for i, label in ((0, "mAP50"), (1, "mAP50-95")):
            vals = [v[i] for _, v in got]
            print(f"  {label:9} mean {st.mean(vals):.4f}  sd {st.stdev(vals):.4f}"
                  f"  range {max(vals)-min(vals):.4f}")

out = {"seeds": SEEDS, "epochs": 20,
       "arms": {a: {str(s): {"map50": v[0], "map": v[1]} for s, v in g}
                for a, g in arms.items()}}

if len(arms["fasdd"]) >= 2 and len(arms["coco"]) >= 2:
    print("\n=== pretraining effect, with error bar ===")
    for i, label in ((0, "mAP50"), (1, "mAP50-95")):
        f = [v[i] for _, v in arms["fasdd"]]
        c = [v[i] for _, v in arms["coco"]]
        nf, nc = len(f), len(c)
        d = st.mean(f) - st.mean(c)
        sp = (((nf-1)*st.stdev(f)**2 + (nc-1)*st.stdev(c)**2) / (nf+nc-2)) ** 0.5
        se = sp * (1/nf + 1/nc) ** 0.5
        # two-sided 95% t critical values by dof, enough for n<=5 per arm
        tcrit = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447,
                 7:2.365, 8:2.306}.get(nf+nc-2, 1.96)
        lo, hi = d - tcrit*se, d + tcrit*se
        print(f"  {label:9} diff {d:+.4f}  pooled sd {sp:.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT' if lo > 0 else 'NOT significant at 95%'}")
        out.setdefault("effect", {})[label] = {
            "diff": d, "pooled_sd": sp, "ci95": [lo, hi], "significant": lo > 0}

    sd50 = out["effect"]["mAP50"]["pooled_sd"]
    print("\n=== decision rule for existing claims (mAP50 sigma "
          f"= {sd50:.4f}) ===")
    for claim, val in (("pretraining effect (fair control)", 0.0328),
                       ("freeze=0 over freeze=10", 0.0280),
                       ("fasdd_frz10 retention edge", 0.0195)):
        n = val / sd50 if sd50 else float("inf")
        verdict = ("solid" if n >= 3 else
                   "marginal" if n >= 2 else "WITHIN NOISE -- do not claim")
        print(f"  {claim:36} {val:+.4f} = {n:4.1f} sigma  -> {verdict}")

(LOGDIR / "seed_variance.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {LOGDIR/'seed_variance.json'}")
PY

say "ALLDONE"
