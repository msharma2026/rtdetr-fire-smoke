"""Is the mAP50-95 ceiling annotation-limited or model-limited?

If loose/ambiguous ground-truth boxes are the cap, AP should fall off a cliff
as the IoU threshold rises, and it should fall faster for smoke (no crisp
boundary) than for fire. If the model simply localises poorly, both classes
should degrade at a similar, gentler rate.

Reads the STAGED test copy (data/dfire/test), never datasets/D-Fire -- the raw
copy uses the opposite class order and silently collapses mAP ~8x.
"""
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import RTDETR

ROOT = Path.home() / "repos/fire_detection"
IMG_DIR = ROOT / "data/dfire/test/images"
LBL_DIR = ROOT / "data/dfire/test/labels"
CKPT = ROOT / "runs/joint_r16_explore/weights/best.pt"
CONF = 0.001
BATCH = 16
NAMES = {0: "fire", 1: "smoke"}
THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


def load_gt(label_path, w, h):
    boxes, labels = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c, xc, yc, bw, bh = float(f[0]), *map(float, f[1:5])
            if not (0 <= xc <= 1.05 and 0 <= yc <= 1.05):
                continue
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
            boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
            labels.append(int(c))
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    files = sorted(IMG_DIR.glob("*.jpg"))
    print(f"{len(files)} test images, ckpt {CKPT.name}", flush=True)
    model = RTDETR(str(CKPT))

    preds, targets = [], []
    for i in range(0, len(files), BATCH):
        chunk = files[i:i + BATCH]
        results = model.predict(chunk, conf=CONF, verbose=False, device="0", half=True)
        for f, r in zip(chunk, results):
            b = r.boxes
            preds.append({
                "boxes": b.xyxy.cpu().float(),
                "scores": b.conf.cpu().float(),
                "labels": b.cls.cpu().long(),
            })
            w, h = Image.open(f).size
            targets.append(load_gt(LBL_DIR / f"{f.stem}.txt", w, h))
        if i % (BATCH * 50) == 0:
            print(f"  {i + len(chunk)}/{len(files)}", flush=True)

    print("\nAP by IoU threshold (per class):", flush=True)
    print(f"{'IoU':>6} {'fire':>8} {'smoke':>8} {'mean':>8}")
    rows = {}
    for thr in THRESHOLDS:
        m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                 iou_thresholds=[thr], class_metrics=True)
        m.update(preds, targets)
        r = m.compute()
        per = r["map_per_class"]
        fire, smoke = float(per[0]), float(per[1])
        rows[thr] = (fire, smoke)
        print(f"{thr:>6.2f} {fire:>8.4f} {smoke:>8.4f} {(fire + smoke) / 2:>8.4f}",
              flush=True)

    f50, s50 = rows[0.50]
    print("\nretained fraction of AP50:")
    print(f"{'IoU':>6} {'fire':>8} {'smoke':>8}")
    for thr in THRESHOLDS:
        f, s = rows[thr]
        print(f"{thr:>6.2f} {f / f50:>8.1%} {s / s50:>8.1%}")


if __name__ == "__main__":
    main()
