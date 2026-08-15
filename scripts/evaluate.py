"""Evaluate a trained checkpoint on D-Fire's held-out test set.

D-Fire test (4,306 images) is the only number that matters for this project --
it is the generalization target, untouched by either training stage.

    python scripts/evaluate.py                      # stage-2 best.pt
    python scripts/evaluate.py --weights <path>     # any checkpoint
    python scripts/evaluate.py --stage 1            # stage-1 model, for the
                                                    # before/after comparison
"""

import argparse
from pathlib import Path

from ultralytics import RTDETR

ROOT = Path.home() / "repos/fire_detection"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
STAGE_RUN = {1: "stage1_fasdd", 2: "stage2_dfire"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=None)
    p.add_argument("--stage", type=int, choices=(1, 2), default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    weights = args.weights or RUNS / STAGE_RUN[args.stage] / "weights/best.pt"
    if not Path(weights).exists():
        raise SystemExit(f"no checkpoint at {weights}")

    print(f"evaluating {weights} on D-Fire test")
    m = RTDETR(str(weights))
    r = m.val(
        data=str(DATA / "dfire.yaml"),
        split="test",
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(RUNS),
        name=f"eval_{Path(weights).parent.parent.name}_dfire_test",
        exist_ok=True,
        plots=True,
    )

    print("\n--- D-Fire test ---")
    print(f"mAP50-95 {r.box.map:.4f}   mAP50 {r.box.map50:.4f}")
    for i, name in enumerate(("fire", "smoke")):
        print(f"  {name:6} mAP50-95 {r.box.maps[i]:.4f}")


if __name__ == "__main__":
    main()
