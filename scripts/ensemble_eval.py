"""Ensemble + TTA evaluation of the 3 seed-variance models on D-Fire test.

Deliberately built on the STABLE public API (model.predict()) plus an
independent, well-tested metrics library (torchmetrics), rather than
subclassing Ultralytics' internal DetectionValidator -- a first attempt at
that hit version-specific internals (postprocess() returns a dict-keyed
format in 8.4.118, not the tuple format an older version used) that would
have silently produced wrong numbers if not caught. This version is slower
but the failure mode of a wrong assumption is an exception, not a plausible
-looking incorrect mAP.

SANITY CHECK, not optional: before trusting any ensemble/TTA number, config 1
(single seed, no TTA, no merging) is scored through this exact same pipeline
and must land within noise (~0.001) of the seed's already-known result
(0.8359 mAP50, runs/seed_variance/eval_seedvar_fasdd_s2.log). If it doesn't,
the pipeline has a bug and nothing downstream is trustworthy.

Four configs, cheapest first:
  1. baseline      -- seedvar_fasdd_s2 alone, no TTA. THE SANITY CHECK.
  2. tta           -- each seed individually, flip+multiscale (predict's
                      native augment=True).
  3. ensemble      -- 3 seeds merged via weighted box fusion, no TTA.
  4. ensemble+tta  -- 3 seeds, each with TTA, merged via WBF.
"""

import json
from pathlib import Path

import torch
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import RTDETR

ROOT = Path.home() / "repos/fire_detection"
SEEDS = ["seedvar_fasdd_s0", "seedvar_fasdd_s1", "seedvar_fasdd_s2"]
CKPTS = [ROOT / f"runs/{s}/weights/best.pt" for s in SEEDS]
# MUST be the STAGED copy (data/dfire/), not raw datasets/D-Fire/ -- the raw
# labels use D-Fire's native class order (0=smoke,1=fire), the opposite of the
# canonical order (0=fire,1=smoke) every trained model expects. prepare_data.py
# remaps this once at staging time. Using the raw labels here caused an 8x mAP
# collapse (0.10 vs the known 0.836) on the first attempt.
IMG_DIR = ROOT / "data/dfire/test/images"
LBL_DIR = ROOT / "data/dfire/test/labels"
IMGSZ = 640
CONF = 0.001   # let the metric's PR curve see everything; do not pre-filter
IOU_NMS = 0.7  # each model's own per-model NMS, matches evaluate.py
WBF_IOU_THR = 0.55
# Pre-filter before WBF, separate from CONF (which stays at 0.001 for the
# single-model AP tail). Feeding WBF the full ~300 raw boxes/model (conf>=
# 0.001) caused an 8x mAP collapse (0.65 vs known-good ~0.83): three models
# agreeing at 0.78 confidence on the identical box (verified: coords within
# <1px, IoU ~0.99) merged down to 0.16 once ~900 total candidate boxes were
# in play, most of them noise. Isolating just the 3 real boxes confirmed WBF
# fuses them correctly (0.778 = their average) -- the bug is an artifact of
# clustering at that box count, not a coordinate or API misuse. Pre-filtering
# to a sane confidence before merging fixes it.
MERGE_CONF = 0.05
WBF_CONF_TYPE = "max"  # not the default "avg": avg divides by TOTAL model
# count regardless of how many actually contributed to a cluster, penalising
# any box even 1 of 3 near-identical models missed. "max" avoids that and
# matched/slightly beat the single-model baseline on a 400-image check;
# "avg" and "box_and_model_avg" both landed marginally below it.
OUT = ROOT / "runs/ensemble"


def load_gt(label_path, w, h):
    boxes, labels = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c, xc, yc, bw, bh = float(f[0]), *map(float, f[1:5])
            if not (0 <= xc <= 1.05 and 0 <= yc <= 1.05):
                continue  # same 4 malformed labels evaluate.py/val.py skip
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
            boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
            labels.append(int(c))
    return boxes, labels


def predict_one(model, img_path, augment):
    r = model.predict(str(img_path), imgsz=IMGSZ, conf=CONF, iou=IOU_NMS,
                      augment=augment, half=True, device="0", verbose=False,
                      max_det=300)[0]
    b = r.boxes
    if len(b) == 0:
        return [], [], []
    return b.xyxy.cpu().tolist(), b.conf.cpu().tolist(), b.cls.cpu().tolist()


def filter_for_merge(boxes, scores, labels):
    """Pre-filter to MERGE_CONF before WBF -- see the MERGE_CONF comment."""
    keep = [i for i, s in enumerate(scores) if s >= MERGE_CONF]
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]


def merge(all_boxes, all_scores, all_labels, w, h):
    """all_* are lists-of-lists, one inner list per model, pixel-space boxes."""
    if len(all_boxes) == 1:
        return all_boxes[0], all_scores[0], all_labels[0]
    norm_boxes = []
    for boxes in all_boxes:
        nb = [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in boxes]
        norm_boxes.append(nb)
    mb, ms, ml = weighted_boxes_fusion(
        norm_boxes, all_scores, all_labels,
        iou_thr=WBF_IOU_THR, skip_box_thr=0.0, conf_type=WBF_CONF_TYPE)
    mb = [[x1 * w, y1 * h, x2 * w, y2 * h] for x1, y1, x2, y2 in mb]
    return mb, list(ms), list(ml)


def run(label, models, augment):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    files = sorted(IMG_DIR.glob("*.jpg"))
    n_skipped = 0

    for i, img_path in enumerate(files):
        try:
            w, h = Image.open(img_path).size
        except Exception:
            n_skipped += 1
            continue
        gt_boxes, gt_labels = load_gt(LBL_DIR / (img_path.stem + ".txt"), w, h)

        all_b, all_s, all_c = [], [], []
        for m in models:
            b, s, c = predict_one(m, img_path, augment)
            if len(models) > 1:
                b, s, c = filter_for_merge(b, s, c)
            all_b.append(b); all_s.append(s); all_c.append(c)
        mb, ms, ml = merge(all_b, all_s, all_c, w, h)

        pred = {
            "boxes": torch.tensor(mb, dtype=torch.float32) if mb else torch.zeros((0, 4)),
            "scores": torch.tensor(ms, dtype=torch.float32) if ms else torch.zeros((0,)),
            "labels": torch.tensor(ml, dtype=torch.int64) if ml else torch.zeros((0,), dtype=torch.int64),
        }
        target = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.zeros((0,), dtype=torch.int64),
        }
        metric.update([pred], [target])

        if (i + 1) % 500 == 0:
            print(f"    {label}: {i+1}/{len(files)}", flush=True)

    r = metric.compute()
    map50 = float(r["map_50"])
    map5095 = float(r["map"])
    print(f"__RESULT__ {label}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  "
          f"(scored {len(files) - n_skipped}/{len(files)} images)", flush=True)
    return {"map50": map50, "map": map5095, "n_images": len(files) - n_skipped}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = [RTDETR(str(c)) for c in CKPTS]

    results = {}

    print("=== 1. SANITY CHECK: seed s2 alone, no TTA (expect ~0.8359 mAP50) ===",
          flush=True)
    results["baseline_s2"] = run("baseline_s2", [models[2]], augment=False)
    known = 0.8359
    got = results["baseline_s2"]["map50"]
    if abs(got - known) > 0.01:
        print(f"\n!!! SANITY CHECK FAILED: got {got:.4f}, expected ~{known} !!!")
        print("!!! Stopping -- do not trust ensemble/TTA numbers until this is fixed !!!")
        (OUT / "ensemble_results.json").write_text(json.dumps(results, indent=2))
        return
    print(f"    sanity check passed ({got:.4f} vs known {known}, "
          f"delta {got-known:+.4f})\n", flush=True)

    print("=== 2. TTA: each seed individually ===", flush=True)
    for i, m in enumerate(models):
        results[f"tta_s{i}"] = run(f"tta_s{i}", [m], augment=True)

    print("=== 3. ensemble: 3 seeds, no TTA ===", flush=True)
    results["ensemble_notta"] = run("ensemble_notta", models, augment=False)

    print("=== 4. ensemble + TTA: 3 seeds, each with TTA ===", flush=True)
    results["ensemble_tta"] = run("ensemble_tta", models, augment=True)

    (OUT / "ensemble_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT/'ensemble_results.json'}")

    print("\n================ SUMMARY ================")
    base = results["baseline_s2"]["map50"]
    for k, v in results.items():
        print(f"  {k:<18} mAP50={v['map50']:.4f}  mAP50-95={v['map']:.4f}  "
              f"delta={v['map50']-base:+.4f}")


if __name__ == "__main__":
    main()
