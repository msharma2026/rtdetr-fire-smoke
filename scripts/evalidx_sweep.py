"""RT-DETR decoder early-exit sweep: accuracy and latency vs decoder depth.

RT-DETR trains every decoder layer with its own bbox/score head (deep
supervision), so stopping early returns a trained prediction rather than an
unfinished intermediate. DeformableTransformerDecoder.forward breaks at
self.eval_idx during eval, so decoder depth is a no-retraining inference knob.

eval_idx=k uses layers 0..k, i.e. k+1 of 6.
"""
import time

import torch
from ultralytics import RTDETR

from pathlib import Path

ROOT = Path.home() / "repos/fire_detection"
CKPT = ROOT / "runs/joint_r16_explore/weights/best.pt"
DATA = ROOT / "data/dfire.yaml"
WARMUP, ITERS = 20, 60


def latency(model_module, eval_idx):
    """Forward-only batch-1 latency at a given decoder depth, fp16."""
    model_module.model[-1].decoder.eval_idx = eval_idx
    m = model_module.eval().half().cuda()
    x = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.half)
    with torch.no_grad():
        for _ in range(WARMUP):
            m(x)
        torch.cuda.synchronize()
        lat = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            m(x)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return lat[len(lat) // 2]


def main() -> None:
    rows = []
    for k in range(6):
        model = RTDETR(str(CKPT))
        dec = model.model.model[-1].decoder
        dec.eval_idx = k
        assert dec.eval_idx == k, "eval_idx did not stick"

        res = model.val(data=str(DATA), split="test", batch=16, imgsz=640,
                        device="0", half=True, plots=False, verbose=False,
                        project=str(ROOT / "runs"), name=f"evalidx_{k}",
                        exist_ok=True)
        # re-assert: the validator must not have rebuilt the module
        used = model.model.model[-1].decoder.eval_idx
        ms = latency(model.model, k)
        rows.append((k, k + 1, float(res.box.map50), float(res.box.map), ms, used))
        print(f"__ROW__ eval_idx={k} layers={k+1} mAP50={res.box.map50:.4f} "
              f"mAP50-95={res.box.map:.4f} median_ms={ms:.2f} used_idx={used}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    base50, base95, base_ms = rows[-1][2], rows[-1][3], rows[-1][4]
    print("\n__SUMMARY__")
    print(f"{'layers':>7} {'mAP50':>8} {'d mAP50':>9} {'mAP50-95':>9} "
          f"{'ms':>7} {'speedup':>8}")
    for k, n, m50, m95, ms, _ in rows:
        print(f"{n:>7} {m50:>8.4f} {m50 - base50:>+9.4f} {m95:>9.4f} "
              f"{ms:>7.2f} {base_ms / ms:>7.2f}x")


if __name__ == "__main__":
    main()
