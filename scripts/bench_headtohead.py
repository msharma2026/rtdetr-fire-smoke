"""Head-to-head latency: our RT-DETR-L vs Venancio's YOLOv5s/YOLOv5l.

WRITTEN BY CLAUDE (2026-09-06). Not the project author's code -- edit or
delete freely.

Both models measured on the SAME machine, SAME images, SAME protocol, so the
comparison is free of the cross-paper confounds that made every published
D-Fire number unusable (different hardware, different batch size, unstated
whether NMS is included).

Two numbers per model, because they answer different questions:

  FORWARD-ONLY  raw nn.Module on a preprocessed tensor. Pure architecture
                cost, no framework overhead. Fair architectural comparison.
  END-TO-END    numpy image -> final boxes, each model through its OWN native
                pipeline (letterbox -> forward -> NMS/decode). What a
                deployment actually pays, and the only number a "real-time"
                claim may honestly rest on.

Why end-to-end matters here specifically: RT-DETR is NMS-free by design,
YOLOv5 is not. That difference exists ONLY in the end-to-end number, and it
also makes RT-DETR's latency independent of how many objects are in frame,
which forward-only timing cannot show.

Run at batch 1 -- deployment processes one frame at a time. Batch-16
throughput divided by 16 is NOT latency and is the most common way this claim
gets faked.

Reports p95/p99, not just mean: a real-time claim is about worst case. A model
averaging 20 ms with a 60 ms tail drops frames.
"""

import json
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path.home() / "repos/fire_detection"
YOLOV5 = Path.home() / "yolov5-up"
OUT = ROOT / "runs/headtohead"
IMGSZ = 640
WARMUP = 30
ITERS = 200

RTDETR_CKPT = ROOT / "runs/seedvar_fasdd_s2/weights/best.pt"
V5S = Path.home() / "dfire-models/yolov5s.pt"
V5L = Path.home() / "dfire-models/yolov5l.pt"


def load_images(n=40):
    """Real D-Fire test images, PRE-LETTERBOXED to exactly 640x640.

    Critical for fairness: ultralytics' predict() letterboxes rectangularly
    (a 1200x720 frame becomes ~640x384, ~40% fewer pixels) while the yolov5
    path here uses auto=False and pads to a full square. Feeding both an
    already-square 640x640 image removes that asymmetry -- otherwise RT-DETR
    is timed on materially less work than YOLOv5.
    """
    sys.path.insert(0, str(YOLOV5))
    from utils.augmentations import letterbox
    d = ROOT / "datasets/D-Fire/test/images"
    files = sorted(d.glob("*.jpg"))[:n]
    imgs = []
    for f in files:
        im = cv2.imread(str(f))
        if im is not None:
            imgs.append(letterbox(im, IMGSZ, stride=32, auto=False)[0])
    assert all(i.shape[:2] == (IMGSZ, IMGSZ) for i in imgs), "not square"
    return imgs


def pct(lat_ms):
    s = sorted(lat_ms)
    return {
        "mean_ms": round(st.mean(s), 2),
        "p50_ms": round(st.median(s), 2),
        "p95_ms": round(s[int(0.95 * len(s))], 2),
        "p99_ms": round(s[int(0.99 * len(s))], 2),
        "fps_mean": round(1000 / st.mean(s), 1),
        "fps_p99": round(1000 / s[int(0.99 * len(s))], 1),
    }


# ---------------------------------------------------------------- forward-only
def bench_forward(module, half):
    dtype = torch.half if half else torch.float
    module.eval()   # RT-DETR computes auxiliary decoder heads in train mode
    x = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda", dtype=dtype)
    with torch.no_grad():
        for _ in range(WARMUP):
            module(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    lat = []
    with torch.no_grad():
        for _ in range(ITERS):
            t0 = time.perf_counter()
            module(x)
            torch.cuda.synchronize()   # CUDA is async: without this we time
            lat.append((time.perf_counter() - t0) * 1000)   # the launch only
    r = pct(lat)
    r["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    return r


# ---------------------------------------------------------------- yolov5 e2e
def bench_yolov5_e2e(weights, images, half):
    sys.path.insert(0, str(YOLOV5))
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    model = DetectMultiBackend(str(weights), device=torch.device("cuda"),
                               fp16=half)
    model.warmup(imgsz=(1, 3, IMGSZ, IMGSZ))

    def one(im0):
        im = letterbox(im0, IMGSZ, stride=model.stride, auto=False)[0]
        im = im.transpose((2, 0, 1))[::-1]          # BGR->RGB, HWC->CHW
        im = np.ascontiguousarray(im)
        t = torch.from_numpy(im).to("cuda")
        t = t.half() if half else t.float()
        t /= 255.0
        t = t[None]
        pred = model(t)
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=300)
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(t.shape[2:], det[:, :4], im0.shape).round()
        return pred

    for i in range(WARMUP):
        one(images[i % len(images)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    lat, ndet = [], []
    for i in range(ITERS):
        im0 = images[i % len(images)]
        t0 = time.perf_counter()
        pred = one(im0)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        ndet.append(sum(len(d) for d in pred))
    r = pct(lat)
    r["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    r["mean_detections"] = round(st.mean(ndet), 2)
    return r, model.model


# ---------------------------------------------------------------- rtdetr e2e
def bench_rtdetr_e2e(images, half):
    from ultralytics import RTDETR
    m = RTDETR(str(RTDETR_CKPT))
    # prime the predictor
    for i in range(WARMUP):
        m.predict(images[i % len(images)], imgsz=IMGSZ, device="0",
                  half=half, verbose=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    lat, ndet = [], []
    for i in range(ITERS):
        im0 = images[i % len(images)]
        t0 = time.perf_counter()
        res = m.predict(im0, imgsz=IMGSZ, device="0", half=half, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        ndet.append(len(res[0].boxes))
    r = pct(lat)
    r["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    r["mean_detections"] = round(st.mean(ndet), 2)
    return r, m.model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = load_images()
    print(f"{len(images)} real D-Fire test images loaded\n")
    results = {}

    for half in (False, True):
        tag = "fp16" if half else "fp32"
        print(f"================ {tag} ================", flush=True)

        # --- RT-DETR (ours)
        r_e2e, rt_mod = bench_rtdetr_e2e(images, half)
        rt_mod = rt_mod.half() if half else rt_mod.float()
        r_fwd = bench_forward(rt_mod, half)
        results[f"rtdetr-l_{tag}"] = {"e2e": r_e2e, "forward": r_fwd}
        print(f"  rtdetr-l   e2e {r_e2e['mean_ms']:>6.2f} ms "
              f"(p99 {r_e2e['p99_ms']:>6.2f})  fwd {r_fwd['mean_ms']:>6.2f} ms "
              f"| {r_e2e['fps_mean']:>5.1f} FPS | dets {r_e2e['mean_detections']}",
              flush=True)
        del rt_mod
        torch.cuda.empty_cache()

        # --- YOLOv5s / YOLOv5l (Venancio)
        for name, w in (("yolov5s", V5S), ("yolov5l", V5L)):
            r_e2e, mod = bench_yolov5_e2e(w, images, half)
            r_fwd = bench_forward(mod, half)
            results[f"{name}_{tag}"] = {"e2e": r_e2e, "forward": r_fwd}
            print(f"  {name:<10} e2e {r_e2e['mean_ms']:>6.2f} ms "
                  f"(p99 {r_e2e['p99_ms']:>6.2f})  fwd {r_fwd['mean_ms']:>6.2f} ms "
                  f"| {r_e2e['fps_mean']:>5.1f} FPS | dets {r_e2e['mean_detections']}",
                  flush=True)
            del mod
            torch.cuda.empty_cache()
        print()

    (OUT / "headtohead.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT/'headtohead.json'}")

    print("\n================ SUMMARY (batch 1, end-to-end) ================")
    print(f"{'model':<12} {'prec':<5} {'mean':>8} {'p95':>8} {'p99':>8} "
          f"{'FPS':>7} {'FPS@p99':>8}  30FPS?")
    for k, v in results.items():
        name, tag = k.rsplit("_", 1)
        e = v["e2e"]
        ok = "YES" if e["p99_ms"] <= 33.3 else "no (tail)"
        print(f"{name:<12} {tag:<5} {e['mean_ms']:>7.2f}ms {e['p95_ms']:>7.2f}ms "
              f"{e['p99_ms']:>7.2f}ms {e['fps_mean']:>7.1f} {e['fps_p99']:>8.1f}  {ok}")


if __name__ == "__main__":
    main()
