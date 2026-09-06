"""FASDD val retention for the 3 seed-variance FASDD-arm checkpoints.

WRITTEN BY CLAUDE (2026-08-20). Completes the retention picture for the
current best recipe (20 ep, lr 1e-4) and gives it an error bar, which none of
the earlier single-run retention numbers have.

Baseline for comparison: stage 1 alone scores 0.8038 on this same FASDD val
split; the COCO-init arms, which never saw FASDD, score ~0.556.
"""
import json
import statistics as st
from pathlib import Path

from ultralytics import RTDETR

ROOT = Path.home() / "repos/fire_detection"
OUT = ROOT / "runs/seed_variance/retention_seedvar.json"

results = {}
for seed in (0, 1, 2):
    w = ROOT / f"runs/seedvar_fasdd_s{seed}/weights/best.pt"
    if not w.exists():
        print(f"seed {seed}: MISSING {w}")
        continue
    m = RTDETR(str(w))
    r = m.val(data=str(ROOT / "data/fasdd.yaml"), split="val", batch=16,
              imgsz=640, device="0", project=str(ROOT / "runs"),
              name=f"retention_seedvar_fasdd_s{seed}", exist_ok=True,
              plots=False, verbose=False)
    results[seed] = {"map50": round(float(r.box.map50), 4),
                     "map": round(float(r.box.map), 4)}
    print(f"__SEED__ {seed} FASDD val mAP50 {r.box.map50:.4f} "
          f"mAP50-95 {r.box.map:.4f}", flush=True)

if len(results) >= 2:
    for key, label in (("map50", "mAP50"), ("map", "mAP50-95")):
        vals = [v[key] for v in results.values()]
        print(f"__AGG__ {label:9} mean {st.mean(vals):.4f} "
              f"sd {st.stdev(vals):.4f} range {max(vals)-min(vals):.4f}")
    base = 0.8038  # stage-1 alone, FASDD val
    drop = base - st.mean([v["map50"] for v in results.values()])
    print(f"__AGG__ drop from stage-1 baseline {base}: -{drop:.4f}")

OUT.write_text(json.dumps(results, indent=2))
print(f"wrote {OUT}")
