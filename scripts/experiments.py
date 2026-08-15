"""Cheap hyperparameter probes, run before committing to the ~15h stage-1 job.

Four groups:

  overfit  -- 16 images, no val. The single most valuable check: if loss does
              not crater toward zero on 16 images the pipeline is broken
              somewhere (labels, class remap, loss, optimizer), and no amount
              of hyperparameter tuning will save the real run.

  lr       -- 3e-5 / 1e-4 / 3e-4 at fixed imgsz+batch. Our 1e-4 came from
              DETR-family convention, not measurement.

  imgsz    -- 640 / 960 / 1280 at FIXED batch=4. Batch is held constant on
              purpose: varying it alongside resolution would confound the two
              (and batch=4 is the only size that fits at 1280). 640 is what
              rtdetr-l.pt was pretrained at (verified in its train_args), so
              this also tests whether training away from that costs anything.

  freeze   -- freeze=0 vs freeze=10 on D-Fire.
              CAVEAT: the real question is whether freezing prevents
              catastrophic forgetting of FASDD features, which cannot be
              tested until stage 1 exists. This probe starts from COCO
              weights instead, so it only answers the weaker question "does
              freezing the backbone hinder adaptation to D-Fire". Treat it as
              directional, not decisive.

  imgsz_dfire, imgsz_dfire_long -- same as imgsz but on D-Fire, at 4 and 14
              epochs respectively. Added after the FASDD-only imgsz probe was
              found to structurally favour 640 regardless of any genuine
              detail-recovery effect, since FASDD's native images are already
              close to 640.

  epoch_budget -- close_mosaic=10 vs 0 on a LARGER FASDD slice
              (fasdd_big, N_TRAIN_BIG images) over BUDGET_EPOCHS epochs.
              Doubles as the epoch-count probe: where the mAP50 curve
              plateaus (or doesn't) is the signal for whether stage 1 needs
              its full 30 epochs.

  coslr    -- cos_lr=False (default, linear decay) vs True.

  mosaic_dfire -- mosaic=1.0 vs 0.0 on D-Fire, where the small-box concern
              (p10 box width ~26px native) actually applies.

Slices are small and fixed so every run in a group sees identical data.

Robustness, learned the hard way:

  * Each probe runs in its OWN subprocess with a timeout. An earlier version
    ran everything in one process and wedged for 59 minutes on a dataloader
    deadlock (workers asleep, GPU at 1%) after five probes had passed.
    Isolation means a hang costs one probe, not the suite.
  * The suite is resumable: a probe counts as done only when its results.csv
    holds every epoch, so a kill redoes only the incomplete run.
  * workers=4, not 8. The hang appeared at dataloader startup, and fewer
    worker processes is the cheapest mitigation under WSL.

Two traps a first draft fell into, both worth knowing about:

  * Gradient accumulation vs. tiny datasets. accumulate = round(nbs/batch),
    so at the default nbs=64 with batch=8 an optimizer step fires only every
    8 iterations. 16 images at batch 8 is 2 iterations/epoch, i.e. one step
    per 4 epochs -- a 60-epoch overfit probe took ~15 gradient steps total and
    looked like a broken pipeline when it was merely starved. The overfit
    probe therefore pins nbs=batch to force accumulate=1.

  * Warmup vs. short probes. nw = max(round(warmup_epochs * iters), 100).
    With the default warmup_epochs=3 a 3-epoch probe is 100% warmup, so an LR
    sweep would compare ramps that never reach their target LR. Probes use
    warmup_epochs=0.5 (still floored at 100 iterations) over 4 epochs.

Usage:
    python scripts/experiments.py            # run/resume the whole suite
    python scripts/experiments.py --run NAME # one probe (used internally)
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "repos/fire_detection"
DATA = ROOT / "data"
OUT = ROOT / "runs/exp"
CFG = OUT / "_cfg"
ROWS = OUT / "rows.jsonl"

SEED = 0
N_TRAIN, N_VAL = 1200, 300
EPOCHS = 4
LONG_EPOCHS = 14  # for probes that need room past initial-convergence effects
WARMUP = 0.5  # the 3.0 default would swallow a 4-epoch probe entirely
WORKERS = 4

# For probes about DATA VOLUME (epoch count, close_mosaic timing), N_TRAIN is
# too small: repeating the same 1,200 images for 20+ epochs mostly measures
# "how well does it memorise this slice", not anything transferable to the
# real 85,783-unique-image run. N_TRAIN_BIG trades probe cost for a slice
# large enough that the plateau it shows is closer to genuine capacity than
# to memorisation of a fixed set. Still far short of the real dataset -- this
# reduces the confound, it doesn't eliminate it.
N_TRAIN_BIG = 3000
BUDGET_EPOCHS = 20

# Probes that ran, failed to complete, and were deliberately abandoned rather
# than retried further. already_done() alone would keep re-attempting these
# on every suite run since they never reach their target epoch count.
# imgsz_1280_dfire_long: timed out twice (2700s, then 5400s); its two partial
# attempts disagreed with each other at matching epochs under identical
# config/seed -- judged unresolvable at this probe scale, not worth a third
# attempt (see DESIGN_DECISIONS.md sec 3.4). Its partial results.csv is kept
# on disk and still appears in summary.json/the notebook, just never rerun.
ABANDONED = {"imgsz_1280_dfire_long"}
TIMEOUT_S = 5400  # per probe. 2700s undershot imgsz_1280_dfire_long, which was
                  # ~300s/epoch x 14 epochs =~4200s; this leaves real margin.
CATEGORIES = ("bothFireAndSmoke", "fire", "neitherFireNorSmoke", "smoke")

PROBES: dict[str, dict] = {
    # nbs=batch forces accumulate=1 (2 steps/epoch); at the default nbs=64
    # this probe gets ~15 steps and proves nothing. lrf=1.0 holds the LR flat
    # so a decaying schedule cannot be blamed either.
    "overfit": dict(group="overfit", slice="overfit", epochs=200, imgsz=640,
                    batch=8, nbs=8, lr0=1e-4, lrf=1.0, warmup_epochs=0,
                    val=False, freeze=0),
    "lr_3e-05": dict(group="lr", slice="fasdd", epochs=EPOCHS, imgsz=960,
                     batch=8, lr0=3e-5, warmup_epochs=WARMUP, freeze=0),
    "lr_1e-04": dict(group="lr", slice="fasdd", epochs=EPOCHS, imgsz=960,
                     batch=8, lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    "lr_3e-04": dict(group="lr", slice="fasdd", epochs=EPOCHS, imgsz=960,
                     batch=8, lr0=3e-4, warmup_epochs=WARMUP, freeze=0),
    # Round 1 (3e-5/1e-4/3e-4) found 3e-4 winning on train loss, mAP50, AND
    # generalisation gap -- monotonically improving with no peak in sight.
    # Extended upward to find where it stops helping or starts destabilising.
    "lr_1e-03": dict(group="lr", slice="fasdd", epochs=EPOCHS, imgsz=960,
                     batch=8, lr0=1e-3, warmup_epochs=WARMUP, freeze=0),
    "lr_3e-03": dict(group="lr", slice="fasdd", epochs=EPOCHS, imgsz=960,
                     batch=8, lr0=3e-3, warmup_epochs=WARMUP, freeze=0),
    "imgsz_640": dict(group="imgsz", slice="fasdd", epochs=EPOCHS, imgsz=640,
                      batch=4, lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    "imgsz_960": dict(group="imgsz", slice="fasdd", epochs=EPOCHS, imgsz=960,
                      batch=4, lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    "imgsz_1280": dict(group="imgsz", slice="fasdd", epochs=EPOCHS, imgsz=1280,
                       batch=4, lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    # Round 1 ran the resolution sweep on FASDD, whose native images (median
    # 718x540) are already ~640 -- so it could only show 640 winning, and
    # never tested the actual hypothesis: D-Fire's larger native images
    # (median 1200x720) may benefit from higher imgsz. Same sweep, D-Fire data.
    "imgsz_640_dfire": dict(group="imgsz_dfire", slice="dfire", epochs=EPOCHS,
                            imgsz=640, batch=4, lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    "imgsz_960_dfire": dict(group="imgsz_dfire", slice="dfire", epochs=EPOCHS,
                            imgsz=960, batch=4, lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    "imgsz_1280_dfire": dict(group="imgsz_dfire", slice="dfire", epochs=EPOCHS,
                             imgsz=1280, batch=4, lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    "freeze_0": dict(group="freeze", slice="dfire", epochs=EPOCHS, imgsz=960,
                     batch=8, lr0=1e-5, warmup_epochs=WARMUP, freeze=0),
    "freeze_10": dict(group="freeze", slice="dfire", epochs=EPOCHS, imgsz=960,
                      batch=8, lr0=1e-5, warmup_epochs=WARMUP, freeze=10),
    # Both 4-epoch resolution sweeps (FASDD and D-Fire) had 640 winning on
    # every metric -- but a 4-epoch, fixed-budget comparison structurally
    # favours whichever imgsz needs no adaptation, since 640 is exactly what
    # rtdetr-l.pt was pretrained at (verified in its train_args) and AIFI's
    # position embeddings need epochs to adjust to a different feature-map
    # size at 960/1280. LONG_EPOCHS gives that adaptation cost room to pay
    # off (or not) before judging. D-Fire only, since that is where the
    # native-resolution hypothesis (median 1200x720) actually applies.
    "imgsz_640_dfire_long": dict(group="imgsz_dfire_long", slice="dfire",
                                 epochs=LONG_EPOCHS, imgsz=640, batch=4,
                                 lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    "imgsz_960_dfire_long": dict(group="imgsz_dfire_long", slice="dfire",
                                 epochs=LONG_EPOCHS, imgsz=960, batch=4,
                                 lr0=1e-4, warmup_epochs=WARMUP, freeze=0),
    "imgsz_1280_dfire_long": dict(group="imgsz_dfire_long", slice="dfire",
                                  epochs=LONG_EPOCHS, imgsz=1280, batch=4,
                                  lr0=1e-4, warmup_epochs=WARMUP, freeze=0),

    # Epoch budget + close_mosaic, combined: both questions are about
    # behaviour late in a longer run, so one pair of runs answers both --
    # does mAP50 still be improving at epoch 20 (epoch-count signal), and
    # does turning mosaic off for the last 10 epochs change the outcome
    # (close_mosaic signal). Uses fasdd_big (3,000 images), not the 1,200
    # slice, and the now-decided production config (imgsz=640, batch=16,
    # lr0=3e-4).
    "epoch_budget_cm10": dict(group="epoch_budget", slice="fasdd_big",
                              epochs=BUDGET_EPOCHS, imgsz=640, batch=16,
                              lr0=3e-4, warmup_epochs=WARMUP, freeze=0,
                              close_mosaic=10),
    "epoch_budget_cm0": dict(group="epoch_budget", slice="fasdd_big",
                             epochs=BUDGET_EPOCHS, imgsz=640, batch=16,
                             lr0=3e-4, warmup_epochs=WARMUP, freeze=0,
                             close_mosaic=0),

    # LR schedule shape: linear decay (Ultralytics default, cos_lr=False) vs
    # cosine. Purely about schedule shape over a training horizon, not data
    # volume, so the smaller fasdd slice is fine here.
    "coslr_off": dict(group="coslr", slice="fasdd", epochs=LONG_EPOCHS,
                      imgsz=640, batch=16, lr0=3e-4, warmup_epochs=WARMUP,
                      freeze=0, cos_lr=False),
    "coslr_on": dict(group="coslr", slice="fasdd", epochs=LONG_EPOCHS,
                     imgsz=640, batch=16, lr0=3e-4, warmup_epochs=WARMUP,
                     freeze=0, cos_lr=True),

    # Mosaic on D-Fire, where the small-box concern actually applies (p10 box
    # width ~26px native): does mosaic (which crams 4 images into one canvas,
    # shrinking already-small boxes further) hurt or help. No per-box-size
    # metric available from Ultralytics' results.csv -- overall mAP50 is the
    # practical signal here, imperfect for isolating small objects specifically.
    "mosaic_on_dfire": dict(group="mosaic_dfire", slice="dfire", epochs=LONG_EPOCHS,
                            imgsz=640, batch=16, lr0=3e-4, warmup_epochs=WARMUP,
                            freeze=0, mosaic=1.0),
    "mosaic_off_dfire": dict(group="mosaic_dfire", slice="dfire", epochs=LONG_EPOCHS,
                             imgsz=640, batch=16, lr0=3e-4, warmup_epochs=WARMUP,
                             freeze=0, mosaic=0.0),
}


def write_split(paths: list[str], dest: Path) -> Path:
    dest.write_text("".join(f"{p}\n" for p in paths))
    return dest


def make_yaml(name: str, train_txt: Path, val_txt: Path) -> Path:
    y = CFG / f"{name}.yaml"
    y.write_text(
        f"path: {CFG}\ntrain: {train_txt}\nval: {val_txt}\n\nnames:\n  0: fire\n  1: smoke\n"
    )
    return y


def build_slices() -> dict[str, Path]:
    """Fixed FASDD + D-Fire slices, plus a 16-image set for the overfit probe.
    Deterministic, so every probe (and every rerun) sees identical data."""
    CFG.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # FASDD: stratify by category so the slice keeps the real class balance
    train_lines = (DATA / "fasdd/train.txt").read_text().splitlines()
    val_lines = (DATA / "fasdd/val.txt").read_text().splitlines()
    ftrain = []
    for c in CATEGORIES:
        pool = [l for l in train_lines if Path(l).name.startswith(c + "_")]
        rng.shuffle(pool)
        ftrain += pool[: N_TRAIN // len(CATEGORIES)]
    fval = val_lines[:]
    rng.shuffle(fval)

    out = {"fasdd": make_yaml(
        "fasdd_exp",
        write_split(ftrain, CFG / "fasdd_train.txt"),
        write_split(fval[:N_VAL], CFG / "fasdd_val.txt"),
    )}

    # Larger FASDD slice for probes about data volume (see N_TRAIN_BIG above).
    # Independently stratified/shuffled from the pool, not a superset of
    # ftrain, so it's its own deterministic sample rather than ftrain padded.
    ftrain_big = []
    for c in CATEGORIES:
        pool = [l for l in train_lines if Path(l).name.startswith(c + "_")]
        rng.shuffle(pool)
        ftrain_big += pool[: N_TRAIN_BIG // len(CATEGORIES)]
    out["fasdd_big"] = make_yaml(
        "fasdd_big_exp",
        write_split(ftrain_big, CFG / "fasdd_big_train.txt"),
        write_split(fval[:N_VAL], CFG / "fasdd_big_val.txt"),
    )

    # overfit: 16 images WITH boxes (a background-only slice would trivially
    # reach zero loss and prove nothing)
    labelled = [l for l in ftrain if not Path(l).name.startswith("neither")][:16]
    out["overfit"] = make_yaml(
        "overfit",
        write_split(labelled, CFG / "overfit_train.txt"),
        write_split(labelled, CFG / "overfit_val.txt"),
    )

    d_train = (DATA / "dfire/train_split.txt").read_text().splitlines()
    d_val = (DATA / "dfire/val.txt").read_text().splitlines()
    rng.shuffle(d_train)
    rng.shuffle(d_val)
    out["dfire"] = make_yaml(
        "dfire_exp",
        write_split(d_train[:N_TRAIN], CFG / "dfire_train.txt"),
        write_split(d_val[:N_VAL], CFG / "dfire_val.txt"),
    )
    return out


def clear_caches() -> None:
    for c in DATA.rglob("*.cache"):
        c.unlink()


def already_done(name: str) -> bool:
    """Done only if results.csv holds every epoch -- makes the suite resumable."""
    csv = OUT / name / "results.csv"
    if not csv.exists():
        return False
    rows = [l for l in csv.read_text().splitlines()[1:] if l.strip()]
    return len(rows) >= PROBES[name]["epochs"]


def run_one(name: str) -> None:
    """Execute a single probe in this process. Invoked as a subprocess."""
    import torch
    from ultralytics import RTDETR

    cfg = dict(PROBES[name])
    group, slice_name = cfg.pop("group"), cfg.pop("slice")
    data = build_slices()[slice_name]

    clear_caches()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    row = {"name": name, "group": group, **cfg}
    try:
        RTDETR("rtdetr-l.pt").train(
            data=str(data), project=str(OUT), name=name, exist_ok=True,
            device="0", workers=WORKERS, optimizer="AdamW",
            deterministic=False, seed=SEED, cache=False, plots=False,
            patience=10_000, verbose=False, **cfg,
        )
        row["ok"] = True
    except Exception as e:  # noqa: BLE001 - a failed probe must not kill the suite
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    row["wall_s"] = round(time.time() - t0, 1)
    row["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    with ROWS.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[{name}] ok={row['ok']} {row['wall_s']}s {row['peak_vram_gb']}GB", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_slices()

    # rows.jsonl accumulates across resumed passes; drop rows we are redoing
    existing = {}
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["name"]] = r

    for name in PROBES:
        if name in ABANDONED:
            # Unconditional, unlike already_done() -- must not depend on
            # results.csv surviving, since the file this was meant to
            # protect was itself deleted by a run that started before this
            # check existed (rmtree fires before "starting" is printed).
            print(f"[{name}] SKIP (abandoned, will not retry)", flush=True)
            continue
        if already_done(name):
            print(f"[{name}] SKIP (already complete)", flush=True)
            continue
        shutil.rmtree(OUT / name, ignore_errors=True)  # drop any partial run
        existing.pop(name, None)
        print(f"[{name}] starting (timeout {TIMEOUT_S}s)", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-u", __file__, "--run", name],
                timeout=TIMEOUT_S, check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"[{name}] TIMEOUT after {TIMEOUT_S}s -- skipping", flush=True)
            with ROWS.open("a") as fh:
                fh.write(json.dumps({"name": name, "group": PROBES[name]["group"],
                                     "ok": False, "error": "timeout"}) + "\n")

    rows = {}
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["name"]] = r  # last write per probe wins
    ordered = [rows[n] for n in PROBES if n in rows]
    (OUT / "summary.json").write_text(json.dumps(ordered, indent=2))
    clear_caches()
    print(f"\nwrote {OUT / 'summary.json'} "
          f"({sum(bool(r.get('ok')) for r in ordered)}/{len(PROBES)} ok)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run a single probe by name")
    a = ap.parse_args()
    if a.run:
        run_one(a.run)
    else:
        main()
