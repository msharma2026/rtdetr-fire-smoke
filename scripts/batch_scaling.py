"""Measure whether training is CUDA-launch-overhead-bound or GPU-compute-bound.

The distinction decides whether a bigger batch (and therefore whether
gradient checkpointing, which buys VRAM to enable one) is worth anything:

  * If throughput in IMAGES/sec rises materially with batch size, the CPU
    cannot issue kernels fast enough and each launch is being amortised over
    more images -- bigger batch is a real win, and checkpointing is a
    legitimate way to afford it.
  * If images/sec is roughly flat, the GPU is already saturated, a bigger
    batch buys nothing, and checkpointing would be pure recompute overhead.

Measures images/sec (NOT it/s -- that trivially falls with batch size and
tells you nothing) and samples GPU utilisation concurrently.

Deliberately runs each batch size in its own subprocess, sequentially, with
the GPU otherwise idle: concurrent GPU work corrupted earlier measurements in
this project.
"""

import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `patches`

ROOT = Path.home() / "repos/fire_detection"
OUT = ROOT / "runs/batch_scaling"
# 24 is deliberately excluded: extrapolating the reliable production fit
# (batch 4 -> 3.42GB, batch 16 -> 12.4GB, i.e. ~0.75GB/image + ~0.4GB fixed)
# puts batch 24 at ~18.4GB, well past the 16GB card. Testing it would just
# buy an OOM. 8/12/16 all fit and are enough to establish the trend.
BATCHES = (8, 12, 16)
IMGSZ = 640
WARMUP_ITERS = 30   # skip: cudnn autotune / allocator warmup distort early iters
TIMED_ITERS = 120   # measured window
COMPILE_WARMUP_ITERS = 90  # torch.compile traces/codegens on first iterations
WORKERS = 4
COMPILE_MODES = (False, "default", "reduce-overhead")
COMPILE_BATCH = 16  # compare compile modes at the production batch size


def sample_gpu(stop_evt, samples):
    """Poll GPU utilisation while training runs."""
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            v = r.stdout.strip().splitlines()
            if v:
                samples.append(int(v[0]))
        except Exception:
            pass
        time.sleep(0.5)


def run_one(batch: int, compile_mode=False, ckpt=False, channels_last=False,
            cudnn_bench=False, ckpt_segments=None) -> dict:
    """Time a fixed number of real training iterations at a given batch size.

    compile_mode: False | "default" | "reduce-overhead" -- passed to
    Ultralytics' `compile` arg (torch.compile, inductor backend).
    """
    import torch
    from ultralytics import RTDETR

    if compile_mode:
        # Upstream HGBlock.forward uses a self-referential generator that
        # TorchDynamo mistraces (channel-count error). Verified numerically
        # identical in eager mode; see scripts/patches.py.
        from patches import patch_hgblock_for_compile
        patch_hgblock_for_compile()

    # torch.compile traces + generates kernels on the first iterations, which
    # can take minutes. The default 30-iteration warmup is nowhere near enough
    # to get past that, and timing it would measure compilation, not steady
    # state.
    warmup = WARMUP_ITERS if not compile_mode else COMPILE_WARMUP_ITERS

    # cuDNN autotunes the fastest conv algorithm per input shape. Ultralytics
    # leaves this off (torch_utils.py:685 has it commented out as an AutoBatch
    # workaround) -- but our shapes are fixed (imgsz 640, rect=False), which is
    # exactly the case benchmark mode is for.
    if cudnn_bench:
        torch.backends.cudnn.benchmark = True

    model = RTDETR("rtdetr-l.pt")
    # Use the trainer so the measured path is the real training path, not a
    # hand-rolled loop that might miss loss/optimizer overhead.
    trainer = model._smart_load("trainer")(overrides={
        "data": str(ROOT / "data/fasdd.yaml"),
        "imgsz": IMGSZ, "batch": batch, "epochs": 1, "workers": WORKERS,
        "optimizer": "AdamW", "lr0": 3e-4, "deterministic": False, "seed": 0,
        "cache": False, "plots": False, "val": False, "verbose": False,
        "project": str(OUT), "name": f"b{batch}", "exist_ok": True,
        "warmup_epochs": 0,  # keep LR schedule out of the timing
        "model": "rtdetr-l.pt",
        "compile": compile_mode,
        # NHWC: what cuDNN's tensor-core conv kernels want natively. NCHW
        # forces internal transposes on every conv. Untested until now.
        "channels_last": channels_last,
    })
    trainer._setup_train()
    trainer.model.train()

    # _setup_train builds a validator + 9,531-image val dataloader regardless
    # of val=False. Unused here and it inflates memory, so drop it before the
    # timed window (an earlier version of this script left it in place and
    # drove the box into swap).
    trainer.test_loader = None
    trainer.validator = None

    n_segs = 0
    if ckpt:
        from patches import enable_segment_checkpointing
        # ckpt_segments lets us checkpoint a SUBSET. The segments differ wildly
        # in value: (0,3) and (4,7) hold 320x320/160x160 activations and are
        # cheap convs to recompute; (8,12) contains the AIFI transformer on a
        # 20x20 grid -- expensive to recompute, almost no memory saved.
        n_segs = len(enable_segment_checkpointing(
            trainer.model, ckpt_segments if ckpt_segments else "auto"))
    torch.cuda.empty_cache()

    loader = trainer.train_loader
    it = iter(loader)
    stop_evt, gpu_samples = threading.Event(), []
    torch.cuda.reset_peak_memory_stats()

    n_done = 0
    t_start = None
    try:
        for i in range(warmup + TIMED_ITERS):
            try:
                batch_data = next(it)
            except StopIteration:
                it = iter(loader)
                batch_data = next(it)

            if i == warmup:  # begin timing after warmup
                torch.cuda.synchronize()
                t_start = time.time()
                threading.Thread(target=sample_gpu, args=(stop_evt, gpu_samples),
                                 daemon=True).start()

            batch_data = trainer.preprocess_batch(batch_data)
            # CRITICAL: real training runs the forward under autocast (amp=True).
            # Without this the loop runs FP32, roughly doubling activation
            # memory and measuring a precision production never uses -- which
            # is exactly how an earlier version hit 16GB at batch 16 versus
            # production's 12.4GB, then thrashed.
            with torch.autocast("cuda", enabled=bool(trainer.amp)):
                loss, _ = trainer.model(batch_data)
            trainer.scaler.scale(loss.sum()).backward()
            trainer.optimizer_step()
            if i >= warmup:
                n_done += 1
    except torch.cuda.OutOfMemoryError as e:
        stop_evt.set()
        return {"batch": batch, "oom": True, "error": str(e)[:200]}

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    stop_evt.set()
    time.sleep(0.6)

    imgs = n_done * batch
    return {
        "batch": batch,
        "iters": n_done,
        "elapsed_s": round(elapsed, 2),
        "it_per_s": round(n_done / elapsed, 3),
        "images_per_s": round(imgs / elapsed, 2),
        "gpu_util_mean": round(statistics.mean(gpu_samples), 1) if gpu_samples else None,
        "gpu_util_max": max(gpu_samples) if gpu_samples else None,
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "amp": bool(trainer.amp),
        "compile": compile_mode if compile_mode else "off",
        "ckpt": bool(ckpt),
        # proves checkpointing actually ran; a previous patch targeted a method
        # RT-DETR never calls and silently measured an unmodified model
        "ckpt_segment_calls": getattr(trainer.model, "ckpt_segment_calls", 0),
        "n_segments": n_segs,
    }


def compile_sweep() -> None:
    """Compare torch.compile modes at the production batch size.

    ~26% of the GPU sits idle in kernel-launch gaps at batch 16 (see the batch
    sweep). compile fuses kernels, and "reduce-overhead" adds CUDA graphs that
    replay a whole launch sequence with one CPU call -- both target that gap
    WITHOUT adding recompute, unlike gradient checkpointing.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in COMPILE_MODES:
        label = mode if mode else "off"
        print(f"\n=== compile={label} (batch {COMPILE_BATCH}) ===", flush=True)
        r = subprocess.run(
            [sys.executable, "-u", __file__, "--run", str(COMPILE_BATCH), str(label)],
            capture_output=True, text=True, timeout=3600,
        )
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            results.append(json.loads(line[0][len("RESULT "):]))
            print(line[0], flush=True)
        else:
            print(f"FAILED compile={label}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}", flush=True)

    (OUT / "compile_sweep.json").write_text(json.dumps(results, indent=2))
    ok = [r for r in results if not r.get("oom")]
    print("\n" + "=" * 74)
    print(f"{'compile':>16} {'img/s':>9} {'GPU% mean':>10} {'GPU% max':>9} {'VRAM GB':>8}")
    for r in results:
        if r.get("oom"):
            print(f"{r.get('compile', '?'):>16} {'OOM':>9}")
            continue
        print(f"{r['compile']:>16} {r['images_per_s']:>9} {r['gpu_util_mean']:>10} "
              f"{r['gpu_util_max']:>9} {r['peak_vram_gb']:>8}")
    if len(ok) >= 2:
        base = next((r for r in ok if r["compile"] == "off"), ok[0])
        print(f"\nvs compile=off ({base['images_per_s']} img/s):")
        for r in ok:
            if r is base:
                continue
            print(f"  {r['compile']:>16}: {r['images_per_s'] / base['images_per_s']:.2f}x")


def opt_sweep() -> None:
    """Untested optimisations, each against the batch-16 baseline.

    channels_last : NHWC is what cuDNN tensor-core conv kernels want natively.
    cudnn_bench   : autotunes conv algorithms per shape; Ultralytics disables
                    it as an AutoBatch workaround we do not need (fixed shapes).
    selective ckpt: only the high-resolution, cheap-to-recompute segments --
                    the earlier sweep checkpointed everything including the
                    expensive AIFI transformer, which is bad value.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    configs = [
        ("baseline",            dict(batch=16)),
        ("channels_last",       dict(batch=16, channels_last=True)),
        ("cudnn_bench",         dict(batch=16, cudnn_bench=True)),
        ("chlast+cudnn",        dict(batch=16, channels_last=True, cudnn_bench=True)),
        ("ckpt_hires_b16",      dict(batch=16, ckpt=True, ckpt_segments=[(0, 3), (4, 7)])),
        ("ckpt_hires_b32",      dict(batch=32, ckpt=True, ckpt_segments=[(0, 3), (4, 7)])),
        ("all_b32",             dict(batch=32, ckpt=True, ckpt_segments=[(0, 3), (4, 7)],
                                     channels_last=True, cudnn_bench=True)),
    ]
    results = []
    for label, kw in configs:
        print(f"\n=== {label} ===", flush=True)
        args = [sys.executable, "-u", __file__, "--run-kw", json.dumps(kw)]
        r = subprocess.run(args, capture_output=True, text=True, timeout=2400)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            row = json.loads(line[0][len("RESULT "):])
            row["label"] = label
            results.append(row)
            print(line[0], flush=True)
        else:
            print(f"FAILED {label}:\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}", flush=True)

    (OUT / "opt_sweep.json").write_text(json.dumps(results, indent=2))
    ok = [r for r in results if not r.get("oom")]
    print("\n" + "=" * 82)
    print(f"{'config':>18} {'batch':>6} {'img/s':>8} {'GPU%':>6} {'VRAM':>7} {'segs':>5}")
    for r in ok:
        print(f"{r['label']:>18} {r['batch']:>6} {r['images_per_s']:>8} "
              f"{r['gpu_util_mean']:>6} {r['peak_vram_gb']:>7} {r['n_segments']:>5}")
    base = next((r for r in ok if r["label"] == "baseline"), None)
    if base:
        print(f"\nvs baseline ({base['images_per_s']} img/s):")
        for r in ok:
            if r is base:
                continue
            d = r["images_per_s"] / base["images_per_s"]
            print(f"  {r['label']:>18}: {d:.3f}x   -> stage 1 ~{17.5 / d:.1f}h")


def ckpt_sweep() -> None:
    """The payoff test: does segment checkpointing buy throughput via bigger batch?

    Verified separately: ~50% activation memory freed, gradients correct
    (cosine 0.9999999 vs baseline). That should make batch 24/32 fit where
    they previously would not. The open question is whether the extra images
    per launch outweigh the recompute cost.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    configs = [(16, False), (16, True), (24, True), (32, True)]
    results = []
    for batch, ck in configs:
        label = f"batch {batch} {'+ckpt' if ck else 'baseline'}"
        print(f"\n=== {label} ===", flush=True)
        r = subprocess.run(
            [sys.executable, "-u", __file__, "--run", str(batch), "off", "1" if ck else "0"],
            capture_output=True, text=True, timeout=2400,
        )
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            row = json.loads(line[0][len("RESULT "):])
            results.append(row)
            print(line[0], flush=True)
            if row.get("ckpt") and not row.get("ckpt_segment_calls"):
                print("  !! ckpt requested but never executed -- result is meaningless",
                      flush=True)
        else:
            print(f"FAILED {label}:\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}", flush=True)

    (OUT / "ckpt_sweep.json").write_text(json.dumps(results, indent=2))
    ok = [r for r in results if not r.get("oom")]
    print("\n" + "=" * 78)
    print(f"{'batch':>6} {'ckpt':>6} {'img/s':>9} {'GPU%':>6} {'VRAM GB':>8} {'segcalls':>9}")
    for r in results:
        if r.get("oom"):
            print(f"{r['batch']:>6} {str(r.get('ckpt')):>6} {'OOM':>9}")
            continue
        print(f"{r['batch']:>6} {str(r['ckpt']):>6} {r['images_per_s']:>9} "
              f"{r['gpu_util_mean']:>6} {r['peak_vram_gb']:>8} {r['ckpt_segment_calls']:>9}")
    base = next((r for r in ok if r["batch"] == 16 and not r["ckpt"]), None)
    if base:
        print(f"\nvs production (batch 16, no ckpt, {base['images_per_s']} img/s):")
        for r in ok:
            if r is base:
                continue
            delta = r["images_per_s"] / base["images_per_s"]
            hrs = 17.5 / delta
            print(f"  batch {r['batch']:>2} ckpt={str(r['ckpt']):>5}: {delta:.2f}x"
                  f"   -> stage 1 ~{hrs:.1f}h (from 17.5h)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for b in BATCHES:
        print(f"\n=== batch {b} ===", flush=True)
        r = subprocess.run([sys.executable, "-u", __file__, "--run", str(b)],
                           capture_output=True, text=True, timeout=1800)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            results.append(json.loads(line[0][len("RESULT "):]))
            print(line[0], flush=True)
        else:
            print(f"FAILED batch {b}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}", flush=True)

    (OUT / "scaling.json").write_text(json.dumps(results, indent=2))
    ok = [r for r in results if not r.get("oom")]
    print("\n" + "=" * 74)
    print(f"{'batch':>6} {'img/s':>9} {'it/s':>8} {'GPU% mean':>10} {'GPU% max':>9} {'VRAM GB':>8}")
    for r in results:
        if r.get("oom"):
            print(f"{r['batch']:>6} {'OOM':>9}")
            continue
        print(f"{r['batch']:>6} {r['images_per_s']:>9} {r['it_per_s']:>8} "
              f"{r['gpu_util_mean']:>10} {r['gpu_util_max']:>9} {r['peak_vram_gb']:>8}")
    if len(ok) >= 2:
        base = ok[0]
        print(f"\nthroughput vs batch {base['batch']} (decides whether bigger batch is worth anything):")
        for r in ok[1:]:
            ratio = r["images_per_s"] / base["images_per_s"]
            print(f"  batch {r['batch']:>2}: {ratio:.2f}x img/s")
        print("\ninterpretation: ~1.0x across the sweep => GPU-saturated, bigger batch")
        print("(and therefore checkpointing) buys nothing. Materially >1.0x =>")
        print("launch-overhead-bound, and batch 32-48 via checkpointing is worth building.")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        mode = sys.argv[3] if len(sys.argv) > 3 else "off"
        cm = False if mode in ("off", "False", "false") else mode
        ck = len(sys.argv) > 4 and sys.argv[4] == "1"
        print("RESULT " + json.dumps(run_one(int(sys.argv[2]), cm, ck)), flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--compile-sweep":
        compile_sweep()
    elif len(sys.argv) > 2 and sys.argv[1] == "--run-kw":
        kw = json.loads(sys.argv[2])
        if kw.get("ckpt_segments"):
            kw["ckpt_segments"] = [tuple(t) for t in kw["ckpt_segments"]]
        print("RESULT " + json.dumps(run_one(**kw)), flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--ckpt-sweep":
        ckpt_sweep()
    elif len(sys.argv) > 1 and sys.argv[1] == "--opt-sweep":
        opt_sweep()
    else:
        main()
