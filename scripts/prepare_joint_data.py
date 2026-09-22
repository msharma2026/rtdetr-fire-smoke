"""Stage a joint FASDD+D-Fire training pool, D-Fire oversampled by --ratio.

Reads the ALREADY-STAGED, ALREADY-REMAPPED files that prepare_data.py
produces (data/fasdd/train.txt, data/dfire/train_split.txt) rather than
touching datasets/ again -- both lists are already in canonical class order
(0=fire, 1=smoke), so this cannot reintroduce the class-id bug that bit both
prepare_data.py's own history (D-Fire test mAP50 collapsed to 0.013) and this
project's first ensemble-eval attempt today (0.10 vs the known 0.836) when
raw D-Fire labels were read directly instead.

Run scripts/prepare_data.py FIRST. This script fails loudly if it hasn't been.

--ratio is how many times each D-Fire train image is duplicated in the pool.
FASDD is 85,783 train images, D-Fire train is ~15,499 -- ratio=1 leaves the
natural ~5.5:1 FASDD-dominated imbalance, ratio=5.5 reaches parity, and higher
ratios make D-Fire the majority. This is the exact "how much to oversample"
question left open after the joint-training design discussion; running a
short ladder across ratios (mirroring scripts/fair_control.sh's LR ladder) is
the intended next step, not picking one value by assumption.

Validates on D-Fire val throughout training (same target metric stage 2
already uses), so joint-training results are directly comparable to the
existing sequential-training numbers. FASDD retention should be checked
separately post-hoc with scripts/retention_seedvar.py's approach, exactly as
was done for the sequentially-trained seeds.
"""

import argparse
from pathlib import Path

DATA = Path.home() / "repos/fire_detection/data"
NAMES = {0: "fire", 1: "smoke"}


def read_lines(p: Path) -> list[str]:
    if not p.exists():
        raise SystemExit(
            f"{p} does not exist -- run scripts/prepare_data.py first "
            f"(this script only combines its output, it doesn't stage raw data)"
        )
    return [l for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratio", type=float, required=True,
                    help="D-Fire train duplication factor (1 = natural imbalance, "
                         "~5.5 = parity with FASDD train count)")
    args = ap.parse_args()

    fasdd = read_lines(DATA / "fasdd/train.txt")
    dfire = read_lines(DATA / "dfire/train_split.txt")

    n_dup = round(len(dfire) * args.ratio)
    # cycle rather than random-resample: every D-Fire image appears an equal
    # number of times (+/-1), no image is systematically over- or under-seen
    dfire_oversampled = [dfire[i % len(dfire)] for i in range(n_dup)]

    pool = fasdd + dfire_oversampled
    tag = f"r{args.ratio:g}"
    out_dir = DATA / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = out_dir / f"train_{tag}.txt"
    train_file.write_text("".join(f"{l}\n" for l in pool))

    names = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
    yaml_path = DATA / f"joint_{tag}.yaml"
    yaml_path.write_text(
        f"path: {DATA}\n"
        f"train: joint/train_{tag}.txt\n"
        f"val: dfire/val.txt\n"
        f"\nnames:\n{names}\n"
    )

    print(f"FASDD train images:        {len(fasdd):>7,}")
    print(f"D-Fire train images:       {len(dfire):>7,}")
    print(f"D-Fire duplicated to:      {n_dup:>7,}  (ratio={args.ratio:g})")
    print(f"combined pool:             {len(pool):>7,}  "
          f"({100*len(fasdd)/len(pool):.1f}% FASDD / "
          f"{100*n_dup/len(pool):.1f}% D-Fire)")
    print(f"\nwrote {train_file}")
    print(f"wrote {yaml_path}")
    print(f"\ntrain with: python scripts/train.py --stage 2 --data joint_{tag}.yaml ...")


if __name__ == "__main__":
    main()
