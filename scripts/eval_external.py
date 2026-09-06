"""Evaluate the project's checkpoints on an EXTERNAL fire/smoke dataset.

WRITTEN BY CLAUDE (2026-08-20). Not the project author's code -- edit or
delete freely. See NOTES_BY_CLAUDE_2026-08-18.md section 3.1.

Answers the generalization question that no D-Fire number can: is
`fasdd_frz10`'s FASDD-retention advantage real transferable ability, or
memorised FASDD annotation conventions?

Runs three guards BEFORE evaluating, because each corresponds to a way this
project (or the published literature around it) has already been burned:

  1. LABEL-ID AUDIT. FASDD uses 0=fire/1=smoke, D-Fire uses the reverse, and
     neither ships a classes.txt. A silent remap produces a plausible-looking
     generalization result that is entirely wrong. Pass --expect-fire and
     --expect-smoke with the dataset's published instance counts and this
     refuses to run unless they match.

  2. CONTAMINATION CHECK. Most community fire/smoke datasets are repackaged
     D-Fire or FASDD. Evaluating on training images yields a spectacular and
     meaningless result. Perceptual (average) hashes of the candidate images
     are compared against every training image.

  3. FLOOR CHECK. A test where every model fails cannot rank models. If the
     best checkpoint lands below --floor mAP50, the comparison is reported as
     non-discriminative rather than as a finding.

Usage:
    python scripts/eval_external.py --data-dir datasets/FASDD_UAV \
        --name fasdd_uav --expect-fire 36308 --expect-smoke 17222

    # if the external set orders classes the other way:
    python scripts/eval_external.py --data-dir ... --remap 0:1,1:0
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "repos/fire_detection"
RUNS = ROOT / "runs"

# Checkpoints spanning the questions we care about. Missing ones are skipped.
DEFAULT_CKPTS = [
    ("stage1_only",   "stage1_fasdd/weights/best.pt"),      # FASDD only, no D-Fire
    ("fasdd_frz10",   "fasdd_frz10/weights/best.pt"),       # retention champion
    ("fasdd_bblr03",  "fasdd_bblr03/weights/best.pt"),      # old D-Fire champion
    ("best_s0",       "seedvar_fasdd_s0/weights/best.pt"),  # current best recipe
    ("best_s1",       "seedvar_fasdd_s1/weights/best.pt"),
    ("best_s2",       "seedvar_fasdd_s2/weights/best.pt"),
    ("coco_s0",       "seedvar_coco_s0/weights/best.pt"),   # D-Fire alone baseline
    ("coco_s1",       "seedvar_coco_s1/weights/best.pt"),
    ("coco_s2",       "seedvar_coco_s2/weights/best.pt"),
]
ARCHIVE = RUNS / "_archive/stage2_arms_20260818"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(root: Path):
    """Return [(image_path, label_path)] for a YOLO-layout dataset."""
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    pairs = []
    for img in imgs:
        # standard YOLO layout: .../images/... <-> .../labels/....txt
        parts = list(img.parts)
        lbl = None
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                lbl = Path(*parts).with_suffix(".txt")
                break
        if lbl is None:
            lbl = img.with_suffix(".txt")
        pairs.append((img, lbl))
    return pairs


def audit_labels(pairs, remap):
    counts, missing, malformed = Counter(), 0, 0
    for _, lbl in pairs:
        if not lbl.exists():
            missing += 1
            continue
        for line in lbl.read_text(errors="ignore").splitlines():
            f = line.split()
            if not f:
                continue
            if len(f) < 5:
                malformed += 1
                continue
            try:
                cid = int(float(f[0]))
            except ValueError:
                malformed += 1
                continue
            counts[remap.get(cid, cid)] += 1
    return counts, missing, malformed


def ahash(path, size=8):
    """64-bit average hash. Inlined to avoid adding an imagehash dependency."""
    from PIL import Image
    try:
        im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    except Exception:
        return None
    px = list(im.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p >= avg:
            bits |= 1 << i
    return bits


def training_hashes(cache: Path):
    """Hashes of every image this project trained on (D-Fire + FASDD)."""
    if cache.exists():
        return set(json.loads(cache.read_text()))
    hashes = set()
    for split in (ROOT / "data/dfire", ROOT / "data/fasdd"):
        if not split.exists():
            continue
        for img in split.rglob("*"):
            if img.suffix.lower() not in IMG_EXT:
                continue
            if "test" in img.parts:      # D-Fire test is held out, not trained on
                continue
            h = ahash(img)
            if h is not None:
                hashes.add(h)
    cache.write_text(json.dumps(sorted(hashes)))
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="external dataset root")
    p.add_argument("--name", required=True, help="short label for outputs")
    p.add_argument("--remap", default="", help='e.g. "0:1,1:0" to swap class ids')
    p.add_argument("--expect-fire", type=int, default=None)
    p.add_argument("--expect-smoke", type=int, default=None)
    p.add_argument("--floor", type=float, default=0.25,
                   help="below this mAP50 the test is called non-discriminative")
    p.add_argument("--skip-dedup", action="store_true")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    remap = {}
    if args.remap:
        for pair in args.remap.split(","):
            a, b = pair.split(":")
            remap[int(a)] = int(b)

    root = Path(args.data_dir)
    if not root.is_absolute():
        root = ROOT / root
    if not root.exists():
        raise SystemExit(f"no such dataset dir: {root}")

    out = RUNS / f"external_{args.name}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== dataset: {root} ===", flush=True)
    pairs = find_pairs(root)
    print(f"{len(pairs)} images found")
    if not pairs:
        raise SystemExit("no images found -- check the directory layout")

    # ---- guard 1: label id audit -----------------------------------------
    print("\n=== label-id audit ===", flush=True)
    counts, missing, malformed = audit_labels(pairs, remap)
    for cid in sorted(counts):
        print(f"  class {cid}: {counts[cid]:,} instances")
    print(f"  labels missing: {missing}   malformed lines: {malformed}")
    if remap:
        print(f"  (remap applied: {remap})")

    ok = True
    if args.expect_fire is not None:
        got = counts.get(0, 0)
        ok &= got == args.expect_fire
        print(f"  expect fire(id 0) = {args.expect_fire:,} -> got {got:,} "
              f"{'OK' if got == args.expect_fire else 'MISMATCH'}")
    if args.expect_smoke is not None:
        got = counts.get(1, 0)
        ok &= got == args.expect_smoke
        print(f"  expect smoke(id 1) = {args.expect_smoke:,} -> got {got:,} "
              f"{'OK' if got == args.expect_smoke else 'MISMATCH'}")
    if not ok:
        raise SystemExit(
            "\nABORT: instance counts do not match the published totals.\n"
            "Either the class ids are ordered differently (try --remap 0:1,1:0)\n"
            "or this is not the dataset you think it is. Do NOT evaluate until\n"
            "this resolves -- a silent remap looks exactly like a real result.")

    # ---- guard 2: contamination ------------------------------------------
    if not args.skip_dedup:
        print("\n=== contamination check ===", flush=True)
        train = training_hashes(RUNS / "train_image_hashes.json")
        print(f"  {len(train):,} training-image hashes")
        hits = sum(1 for img, _ in pairs if ahash(img) in train)
        pct = 100.0 * hits / len(pairs)
        print(f"  exact perceptual-hash collisions: {hits} ({pct:.2f}%)")
        if pct > 1.0:
            print("  WARNING: >1% overlap with training data. Results are"
                  " contaminated and must not be reported as generalization.")

    # ---- evaluate ---------------------------------------------------------
    yaml_path = out / f"{args.name}.yaml"
    yaml_path.write_text(
        f"path: {root}\nval: .\n\nnames:\n  0: fire\n  1: smoke\n")

    from ultralytics import RTDETR
    results = {}
    print("\n=== evaluating ===", flush=True)
    for label, rel in DEFAULT_CKPTS:
        w = RUNS / rel
        if not w.exists():
            w = ARCHIVE / rel          # older arms live in the archive
        if not w.exists():
            print(f"  {label:14} SKIP (no checkpoint)")
            continue
        m = RTDETR(str(w))
        r = m.val(data=str(yaml_path), split="val", batch=args.batch,
                  imgsz=args.imgsz, device="0", project=str(out),
                  name=label, exist_ok=True, plots=False, verbose=False)
        results[label] = {"map50": round(float(r.box.map50), 4),
                          "map": round(float(r.box.map), 4)}
        print(f"  {label:14} mAP50 {r.box.map50:.4f}  mAP50-95 {r.box.map:.4f}",
              flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))

    if not results:
        raise SystemExit("no checkpoints evaluated")

    # ---- guard 3: floor ---------------------------------------------------
    best = max(r["map50"] for r in results.values())
    print(f"\n=== summary ({args.name}) ===")
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]["map50"]):
        print(f"  {label:14} {r['map50']:.4f}  {r['map']:.4f}")
    if best < args.floor:
        print(f"\nNON-DISCRIMINATIVE: best model scores {best:.4f}, below the"
              f" {args.floor} floor.\nEvery model is failing, so differences"
              " between them are not interpretable. Report this as a domain-gap"
              " finding, not as a model ranking.")
    else:
        print(f"\nDiscriminative: best {best:.4f} clears the {args.floor} floor.")
        print("Key contrasts (see notes 9.1 for the relevant sigma):")
        print("  fasdd_frz10 vs fasdd_bblr03 -> is the retention edge real ability?")
        print("  best_s* vs coco_s*          -> does pretraining transfer?")
    print(f"\nwrote {out/'results.json'}")


if __name__ == "__main__":
    main()
