"""Stage FASDD + D-Fire into an Ultralytics-compatible layout.

Two problems this solves:
  1. Ultralytics finds labels by swapping the last "/images/" in an image path
     for "/labels/". FASDD ships labels under annotations/YOLO_CV/labels, which
     is not a sibling of images/, so nothing would resolve.
  2. FASDD and D-Fire disagree on class ids (FASDD 0=fire 1=smoke,
     D-Fire 0=smoke 1=fire). D-Fire labels are rewritten to FASDD order.

Source data is never modified: images are symlinked, only D-Fire labels are
rewritten into new files.
"""

import random
from pathlib import Path

ROOT = Path.home() / "repos/fire_detection"
DATA = ROOT / "data"
FASDD_SRC = ROOT / "datasets/archive/FASDD_CV/FASDD_CV"
DFIRE_SRC = ROOT / "datasets/D-Fire"

NAMES = {0: "fire", 1: "smoke"}  # canonical order == FASDD order
VAL_FRACTION = 0.10
SEED = 0


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def write_yaml(path: Path, root: Path, train: str, val: str, test: str | None = None) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
    test_line = f"test: {test}\n" if test else ""
    path.write_text(
        f"path: {root}\ntrain: {train}\nval: {val}\n{test_line}\nnames:\n{names}\n"
    )


def stage_fasdd() -> None:
    out = DATA / "fasdd"
    out.mkdir(parents=True, exist_ok=True)
    link(FASDD_SRC / "images", out / "images")
    link(FASDD_SRC / "annotations/YOLO_CV/labels", out / "labels")

    # Ignore FASDD's official 50/33/17 split -- stage 1 is feature-learning for
    # D-Fire fine-tuning, not a benchmark to match against published FASDD
    # numbers, so train/val gets all the data. No FASDD test set: D-Fire's own
    # test set is the only number that matters for this project, and a
    # FASDD-native test set doesn't earn its keep as a diagnostic on top of
    # val (which already flags overfitting/divergence during stage-1 training).
    # Stratify by category (encoded in the filename prefix) so each split
    # keeps the same fire/smoke/both/neither balance.
    img_dir = out / "images"
    CATEGORIES = ("bothFireAndSmoke", "fire", "neitherFireNorSmoke", "smoke")
    splits = {"train": [], "val": []}
    rng = random.Random(SEED)
    for cat in CATEGORIES:
        names = sorted(p.name for p in img_dir.iterdir() if p.name.startswith(cat + "_"))
        rng.shuffle(names)
        n_val = round(len(names) * VAL_FRACTION)
        splits["val"] += names[:n_val]
        splits["train"] += names[n_val:]

    for split, names in splits.items():
        names = sorted(names)
        (out / f"{split}.txt").write_text("".join(f"{img_dir / n}\n" for n in names))
        print(f"  fasdd {split:5} {len(names):6} images")

    stale_test = out / "test.txt"
    if stale_test.exists():
        stale_test.unlink()

    write_yaml(DATA / "fasdd.yaml", out, "train.txt", "val.txt")


def remap_labels(src_dir: Path, dst_dir: Path) -> int:
    """Copy YOLO labels swapping class ids 0<->1, preserving box coords verbatim."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in src_dir.iterdir():
        if src.suffix != ".txt":
            continue
        out = []
        for line in src.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cid = parts[0]
            if cid not in ("0", "1"):
                raise ValueError(f"unexpected class id {cid!r} in {src}")
            out.append(" ".join(["1" if cid == "0" else "0", *parts[1:]]))
        # empty label files are background images -- keep them empty, but present
        (dst_dir / src.name).write_text("".join(f"{line}\n" for line in out))
        n += 1
    return n


def stage_dfire() -> None:
    out = DATA / "dfire"
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        link(DFIRE_SRC / split / "images", out / split / "images")
        n = remap_labels(DFIRE_SRC / split / "labels", out / split / "labels")
        print(f"  dfire {split:5} {n:6} labels remapped (0<->1)")

    # D-Fire ships no val split -- carve one deterministically out of train
    imgs = sorted(p.name for p in (DFIRE_SRC / "train/images").iterdir())
    random.Random(SEED).shuffle(imgs)
    k = round(len(imgs) * VAL_FRACTION)
    val, train = sorted(imgs[:k]), sorted(imgs[k:])
    base = out / "train/images"
    (out / "val.txt").write_text("".join(f"{base / n}\n" for n in val))
    (out / "train_split.txt").write_text("".join(f"{base / n}\n" for n in train))
    print(f"  dfire train/val split: {len(train)} / {len(val)} (seed={SEED})")

    write_yaml(DATA / "dfire.yaml", out, "train_split.txt", "val.txt", "test/images")


if __name__ == "__main__":
    print("staging FASDD...")
    stage_fasdd()
    print("staging D-Fire...")
    stage_dfire()
    print(f"\nwrote {DATA / 'fasdd.yaml'} and {DATA / 'dfire.yaml'}")
