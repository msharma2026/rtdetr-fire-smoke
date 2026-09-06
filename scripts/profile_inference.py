"""Inference hardware profile: what would it take to run this in real time?

WRITTEN BY CLAUDE (2026-08-20). Not the project author's code -- edit or
delete freely.

This runs BEFORE the real-time benchmark, to establish which regime the model
is in at batch 1. That determines how (and whether) results extrapolate to
other hardware:

  * COMPUTE-BOUND  -> latency scales with the target device's effective
                      FLOPS, so a TFLOPS ratio predicts other hardware.
  * LAUNCH-BOUND   -> latency is set by kernel count and CPU dispatch speed.
                      TFLOPS ratios MISPREDICT badly; a weaker GPU with a fast
                      CPU may match a stronger one, and TensorRT (which fuses
                      kernels) buys more than its FLOPs math suggests.

This project has already measured the TRAINING path as launch-bound (7,443
launches/iter, 26% of step time; RT-DETR-X at batch 8 beat RT-DETR-L at batch
16 per iteration). Batch 1 should be worse, since the same launch count is
amortised over one image.

The tell is throughput vs batch size: if images/sec rises steeply with batch,
the GPU is idling between launches and we are launch-bound.

Measures, per config: batch-1 latency percentiles (mean/median/p95/p99 --
p99 is what a real-time claim actually rests on), peak inference VRAM, and
concurrent GPU utilisation.
"""

import json
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path.home() / "repos/fire_detection"
CKPT = ROOT / "runs/seedvar_fasdd_s2/weights/best.pt"   # current best model
OUT = ROOT / "runs/inference_profile"
IMGSZ = 640
WARMUP = 50
ITERS = 300


def sample_gpu(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            samples.append(int(r.stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(0.05)


def bench(model, batch, half, iters=ITERS):
    dtype = torch.half if half else torch.float
    x = torch.randn(batch, 3, IMGSZ, IMGSZ, device="cuda", dtype=dtype)

    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    stop_evt, gpu = threading.Event(), []
    threading.Thread(target=sample_gpu, args=(stop_evt, gpu), daemon=True).start()

    lat = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()      # CUDA is async; without this we time
            lat.append(time.perf_counter() - t0)   # the launch, not the work
    stop_evt.set()
    time.sleep(0.2)

    lat_ms = sorted(t * 1000 for t in lat)
    return {
        "batch": batch,
        "precision": "fp16" if half else "fp32",
        "mean_ms": round(st.mean(lat_ms), 2),
        "median_ms": round(st.median(lat_ms), 2),
        "p95_ms": round(lat_ms[int(0.95 * len(lat_ms))], 2),
        "p99_ms": round(lat_ms[int(0.99 * len(lat_ms))], 2),
        "min_ms": round(lat_ms[0], 2),
        "fps_batch1_equiv": round(1000 * batch / st.mean(lat_ms), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "gpu_util_mean": round(st.mean(gpu), 1) if gpu else None,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not CKPT.exists():
        raise SystemExit(f"no checkpoint at {CKPT}")

    from ultralytics import RTDETR
    wrapper = RTDETR(str(CKPT))
    net = wrapper.model.to("cuda").eval()
    for p in net.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params/1e6:.1f}M   fp16 weights: {n_params*2/1e6:.0f} MB")
    print(f"checkpoint: {CKPT.name}   imgsz {IMGSZ}\n")

    results = []

    # batch 1, both precisions -- the deployment-relevant latency
    for half in (False, True):
        if half:
            net.half()
        else:
            net.float()
        r = bench(net, 1, half)
        results.append(r)
        print(f"batch 1 {r['precision']}: mean {r['mean_ms']} ms  "
              f"p95 {r['p95_ms']}  p99 {r['p99_ms']}  "
              f"{r['fps_batch1_equiv']} FPS  "
              f"VRAM {r['peak_vram_gb']} GB  util {r['gpu_util_mean']}%",
              flush=True)

    # batch scaling at fp16 -- the launch-bound tell
    print("\nbatch scaling (fp16) -- rising img/s means launch-bound:")
    net.half()
    for b in (1, 2, 4, 8, 16):
        r = bench(net, b, True, iters=150)
        results.append(r)
        print(f"  batch {b:<3} {r['mean_ms']:>7.2f} ms/iter  "
              f"{r['fps_batch1_equiv']:>7.1f} img/s  "
              f"VRAM {r['peak_vram_gb']:>5.2f} GB  util {r['gpu_util_mean']}%",
              flush=True)

    (OUT / "inference_profile.json").write_text(json.dumps(results, indent=2))

    b1 = next(r for r in results if r["batch"] == 1 and r["precision"] == "fp16")
    b16 = next(r for r in results if r["batch"] == 16 and r["precision"] == "fp16")
    scale = b16["fps_batch1_equiv"] / b1["fps_batch1_equiv"]

    print("\n=== verdict ===")
    print(f"batch-1 fp16: {b1['mean_ms']} ms mean / {b1['p99_ms']} ms p99 "
          f"= {b1['fps_batch1_equiv']} FPS")
    print(f"throughput gain from batch 1 -> 16: {scale:.2f}x")
    if scale > 2.0:
        print("LAUNCH-BOUND at batch 1: the GPU idles between kernels.")
        print("  -> TFLOPS ratios will MISPREDICT other hardware.")
        print("  -> CPU single-thread speed and kernel fusion (TensorRT) matter")
        print("     more than the target GPU's peak throughput.")
    else:
        print("COMPUTE-BOUND at batch 1: latency should scale with effective FLOPS,")
        print("  so a TFLOPS ratio is a reasonable predictor for other hardware.")
    print(f"\ninference VRAM at batch 1: {b1['peak_vram_gb']} GB "
          f"(vs 13.3 GB for training at batch 16)")
    print(f"wrote {OUT/'inference_profile.json'}")


if __name__ == "__main__":
    main()
