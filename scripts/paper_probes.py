"""Probe each RT-DETR-paper deviation before committing to an 80-epoch run.

Paper (Table A) vs what stage 1 actually used:

  base LR            1e-4   vs 3e-4
  backbone LR        1e-5   vs 3e-4  (paper = base/10; Ultralytics has no
                                      per-group LR at all)
  weight decay       1e-4   vs 5e-4
  clip grad norm     0.1    vs 10.0  (hardcoded in Ultralytics)
  augmentation       color/expand/crop/flip/resize -- NO mosaic

Design notes, learned from earlier probes that produced meaningless numbers:

  * 20 epochs, not 14. With close_mosaic=10 a 14-epoch probe leaves only 4
    epochs of mosaic, so the earlier mosaic comparison silently became
    "4 epochs of mosaic vs none". At 20 epochs the production arm gets a
    real 10-epoch mosaic phase.
  * mosaic_never is compared against production settings (mosaic=1.0 +
    close_mosaic=10), i.e. the actual decision being made.
  * Every patched run asserts the patch is LIVE (param counts / attributes),
    because a patch that silently no-ops produces plausible numbers.
  * fasdd_big slice (3,000 imgs) so 20 epochs is not pure memorisation.

Each probe ~25-35 min; the suite is ~3 h. Cheap against a ~26 h retrain.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "repos/fire_detection"
sys.path.insert(0, str(ROOT / "scripts"))

EPOCHS = 20
BATCH = 16
OUT = ROOT / "runs/paper_probes"

# name -> (overrides, patch flags)
PROBES = {
    # control: exactly what stage 1 ran
    "baseline":      ({}, {}),
    # paper deviations, one at a time
    # The paper uses backbone LR = base/10 on COCO with a ResNet backbone.
    # That is their choice for their setup, not a law -- sweep it.
    "bblr_0.05":     ({}, {"bblr": 0.05}),
    "bblr_0.1":      ({}, {"bblr": 0.1}),
    "bblr_0.3":      ({}, {"bblr": 0.3}),
    # 0.05 and 0.1 traded places by metric (mAP50-95 favoured 0.05, mAP50
    # favoured 0.1), i.e. a plateau rather than a slope. These two probe the
    # open direction: does mAP50-95 keep climbing as the backbone slows, and
    # does the paper's ABSOLUTE backbone LR (1e-5) matter more than the ratio?
    # base lr0=3e-4, so mult 0.0333 -> exactly 1e-5.
    # The sweep came back non-monotonic (0.3 > 0.05 > 0.1, all >> 1.0), so the
    # unexplored direction is UPWARD, not down: 0.3 was the top of the range.
    # 0.033 is kept only to test whether the paper's ABSOLUTE backbone LR
    # (1e-5; = 3e-4 * 0.0333) matters more than the ratio.
    "bblr_0.5":      ({}, {"bblr": 0.5}),
    "bblr_0.033":    ({}, {"bblr": 0.0333}),
    # paper_all (0.3932) came in BELOW bblr_0.3 alone (0.4244) because it
    # bundles two changes this data rejects (clip 0.1, lr 1e-4). This is the
    # combination the measurements actually support: the two winning levers
    # only, everything else left at production values.
    "combo":         ({"weight_decay": 1e-4}, {"bblr": 0.3}),
    "gradclip":      ({}, {"clip": 0.1}),
    "wd_1e4":        ({"weight_decay": 1e-4}, {}),
    "mosaic_off":    ({"mosaic": 0.0}, {}),
    "lr_1e4":        ({"lr0": 1e-4}, {}),
    # everything the paper specifies, together
    "paper_all":     ({"weight_decay": 1e-4, "lr0": 1e-4, "mosaic": 0.0},
                      {"bblr": 0.1, "clip": 0.1}),
}


def run_one(name: str) -> None:
    import torch
    import torch._inductor.config as ind

    # DataLoader's default 'file_descriptor' sharing strategy drops connections
    # under this workload ("ConnectionResetError: [Errno 104]" inside
    # multiprocessing.resource_sharer at iteration 0). 'file_system' is the
    # standard workaround and is reproducible-failure-free here.
    torch.multiprocessing.set_sharing_strategy("file_system")

    over, flags = PROBES[name]
    ind.cpp.simdlen = 0
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    import patches
    patches.patch_hgblock_for_compile()
    patches.patch_mha_fast_path()
    patches.patch_rtdetr_loss_syncs()
    if "clip" in flags:
        assert patches.set_grad_clip(flags["clip"])

    from ultralytics import RTDETR

    args = dict(
        data=str(ROOT / "runs/exp/_cfg/fasdd_big_exp.yaml"),
        epochs=EPOCHS, imgsz=640, batch=BATCH, workers=4,
        optimizer="AdamW", lr0=3e-4, lrf=0.01, warmup_epochs=3.0,
        weight_decay=5e-4, close_mosaic=10, deterministic=False, seed=0,
        cache=False, plots=False, val=True, verbose=False,
        project=str(OUT), name=name, exist_ok=True,
        channels_last=True, compile="default", patience=10_000,
    )
    args.update(over)

    model = RTDETR("rtdetr-l.pt")
    state = {}

    def on_setup(trainer):
        state["fused"] = patches.use_fused_adamw(trainer)
        if "bblr" in flags:
            state["moved"] = patches.use_backbone_lr_multiplier(trainer, flags["bblr"])
            assert state["moved"] > 0, "backbone LR patch moved 0 params"
            lrs = sorted({g["lr"] for g in trainer.optimizer.param_groups})
            assert len(lrs) > 1, f"expected >1 distinct LR, got {lrs}"
            state["lrs"] = lrs
        if "clip" in flags:
            from ultralytics.engine.trainer import BaseTrainer
            assert getattr(BaseTrainer, "_clip_max_norm", None) == flags["clip"]
            state["clip"] = flags["clip"]

    model.add_callback("on_pretrain_routine_end", on_setup)
    t0 = time.time()
    model.train(**args)
    el = time.time() - t0

    import csv
    rows = list(csv.reader(open(OUT / name / "results.csv")))
    h = [x.strip() for x in rows[0]]
    rows = rows[1:]
    i50, i5095 = h.index("metrics/mAP50(B)"), h.index("metrics/mAP50-95(B)")
    best = max(float(r[i5095]) for r in rows)
    print("RESULT " + json.dumps({
        "name": name,
        "final_mAP50": round(float(rows[-1][i50]), 4),
        "final_mAP50_95": round(float(rows[-1][i5095]), 4),
        "best_mAP50_95": round(best, 4),
        "best_epoch": 1 + max(range(len(rows)), key=lambda k: float(rows[k][i5095])),
        "epochs": len(rows),
        "wall_min": round(el / 60, 1),
        "verified": state,
    }), flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name in PROBES:
        print(f"\n=== {name} ===", flush=True)
        # The FIRST spawned subprocess reliably dies in multiprocessing's
        # resource_sharer ("ConnectionResetError: [Errno 104]" at iteration 0)
        # while every later one runs fine -- a spawn race, not a config fault.
        # Retry once rather than lose a probe: losing the baseline would
        # invalidate every comparison in the suite.
        for attempt in (1, 2):
            r = subprocess.run([sys.executable, "-u", __file__, "--run", name],
                               capture_output=True, text=True, timeout=7200)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                results.append(json.loads(line[0][7:]))
                print(line[0], flush=True)
                break
            blob = r.stdout + r.stderr
            if attempt == 1 and "ConnectionResetError" in blob:
                print(f"  transient spawn failure, retrying {name}...", flush=True)
                time.sleep(20)
                continue
            print(f"FAILED {name}:", r.stdout[-1200:], r.stderr[-1200:], flush=True)
            break
        (OUT / "summary.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print(f"{'probe':>12} {'mAP50':>8} {'mAP50-95':>9} {'best':>8} {'@ep':>4} {'min':>6}")
    for r in results:
        print(f"{r['name']:>12} {r['final_mAP50']:>8} {r['final_mAP50_95']:>9} "
              f"{r['best_mAP50_95']:>8} {r['best_epoch']:>4} {r['wall_min']:>6}")
    base = next((r for r in results if r["name"] == "baseline"), None)
    if base:
        print(f"\nvs baseline (mAP50-95 {base['best_mAP50_95']}):")
        for r in results:
            if r is base:
                continue
            d = r["best_mAP50_95"] - base["best_mAP50_95"]
            print(f"  {r['name']:>12}: {d:+.4f}  ({d / base['best_mAP50_95'] * 100:+.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        run_one(sys.argv[2])
    else:
        main()
