"""Inference optimization sweep for RT-DETR-L, batch 1, 640x640.

WRITTEN BY CLAUDE (2026-09-06). Not the project author's code -- edit or
delete freely.

Context: eager inference was measured launch-bound (fp16 gave 0.98x, i.e.
nothing; batch 1->16 scaled 4.84x; GPU sat at 52% of max SM clock). CUDA
graphs via torch.compile(mode="reduce-overhead") then gave 2.0x median and
4.3x on p99.

The point of this sweep is a specific prediction: once CUDA graphs remove the
dispatch bottleneck, the workload should become COMPUTE-bound, and fp16 --
which bought exactly nothing in eager mode -- should suddenly start paying.
If fp16 still gives ~1.0x under CUDA graphs, the model is bottlenecked on
something else again and the diagnosis needs revisiting.

Also tests TF32 (inductor explicitly warns it is available but disabled) and
channels_last (measured +9.0% during training).

Every config is timed identically: static 640x640 input, warmup discarded,
torch.cuda.synchronize() around each call, median + p95 + p99 over 120 iters.
p99 matters most -- the eager tail (2.4x median) was the entire reason this
model failed a real-time bar.
"""

import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from patches import patch_hgblock_for_compile  # noqa: E402

ROOT = Path.home() / "repos/fire_detection"
CKPT = str(ROOT / "runs/seedvar_fasdd_s2/weights/best.pt")
OUT = ROOT / "runs/inference_opt"
IMGSZ = 640
ITERS = 120
WARMUP = 40


def fresh_model(half=False, chlast=False):
    from ultralytics import RTDETR
    m = RTDETR(CKPT).model.cuda().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if half:
        m = m.half()
    if chlast:
        m = m.to(memory_format=torch.channels_last)
    return m


def bench(module, half=False, chlast=False):
    x = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda",
                    dtype=torch.half if half else torch.float)
    if chlast:
        x = x.to(memory_format=torch.channels_last)
    with torch.no_grad():
        for _ in range(WARMUP):
            module(x)
        torch.cuda.synchronize()
        lat = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            module(x)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    s = sorted(lat)
    return {
        "median_ms": round(st.median(s), 2),
        "p95_ms": round(s[int(0.95 * len(s))], 2),
        "p99_ms": round(s[int(0.99 * len(s))], 2),
        "min_ms": round(s[0], 2),
        "fps_median": round(1000 / st.median(s), 1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patch_hgblock_for_compile()   # required: dynamo mistraces HGBlock without it
    results = {}

    configs = [
        # (label, half, channels_last, compile_mode, tf32)
        ("eager fp32",                    False, False, None,              False),
        ("cudagraphs fp32",               False, False, "reduce-overhead", False),
        ("cudagraphs fp32 + TF32",        False, False, "reduce-overhead", True),
        ("cudagraphs fp16",               True,  False, "reduce-overhead", False),
        ("cudagraphs fp16 + channels_last", True, True,  "reduce-overhead", False),
    ]

    base = None
    print(f"{'config':<34}{'median':>9}{'p95':>9}{'p99':>9}{'FPS':>8}{'vs base':>9}")
    for label, half, chlast, mode, tf32 in configs:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        if tf32:
            torch.set_float32_matmul_precision("high")
        else:
            torch.set_float32_matmul_precision("highest")
        try:
            m = fresh_model(half, chlast)
            if mode:
                m = torch.compile(m, mode=mode)
                with torch.no_grad():   # trigger compilation outside timing
                    x = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda",
                                    dtype=torch.half if half else torch.float)
                    if chlast:
                        x = x.to(memory_format=torch.channels_last)
                    m(x)
            r = bench(m, half, chlast)
            results[label] = r
            if base is None:
                base = r["median_ms"]
            print(f"{label:<34}{r['median_ms']:>8.2f}ms{r['p95_ms']:>8.2f}ms"
                  f"{r['p99_ms']:>8.2f}ms{r['fps_median']:>8.1f}"
                  f"{base/r['median_ms']:>8.2f}x", flush=True)
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{label:<34} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)

    (OUT / "inference_opt.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT/'inference_opt.json'}")

    if "cudagraphs fp32" in results and "cudagraphs fp16" in results:
        a = results["cudagraphs fp32"]["median_ms"]
        b = results["cudagraphs fp16"]["median_ms"]
        print(f"\nfp16 speedup UNDER cuda graphs: {a/b:.2f}x")
        print("  (eager fp16 was 0.98x -- if this is >1.15x, removing the "
              "dispatch\n   bottleneck made the model compute-bound, as predicted)")


if __name__ == "__main__":
    main()
