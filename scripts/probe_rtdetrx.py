"""RT-DETR-X vs RT-DETR-L: VRAM and throughput probe.

WRITTEN BY CLAUDE (2026-08-18). Not the project author's code -- edit or
delete freely. See NOTES_BY_CLAUDE_2026-08-18.md.

Settles OPEN_ITEMS.md 3.1 / 4.7, deferred since before stage 1 on a
one-sentence estimate that X "roughly doubles" epoch time. That estimate
assumed compute scales to wall-clock, which is exactly what this
dispatch-bound pipeline does not do. X is 2.05x the params but only 1.14x
the top-level layers -- mostly wider, not deeper -- so in a launch-bound
regime it should issue roughly the same number of kernels doing more work
each, filling idle GPU rather than adding proportional wall-clock.

The unchecked counterweight is VRAM: L uses ~11-12GB at batch 16, and X may
not fit on the 16GB card at all. That is what this measures.

Method is lifted from scripts/batch_scaling.py:run_one() so the numbers are
comparable to the existing measurements: the real Ultralytics trainer (not a
hand-rolled loop), autocast on, validator dropped before the timed window,
warmup discarded, each config in its own subprocess with the GPU otherwise
idle.

Deliberately run EAGER (compile off) for both models: compile warmup is
minutes and the question here is relative cost, which a matched eager
comparison answers in a fraction of the time.
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
OUT = ROOT / "runs/probe_rtdetrx"
IMGSZ = 640
WORKERS = 4
WARMUP_ITERS = 30
TIMED_ITERS = 60
# Card is 16.38GB. A config whose measured peak exceeds this is treated as
# NOT fitting even if it ran without raising -- see the skip logic in main().
VRAM_FIT_GB = 15.0


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
        time.sleep(0.25)


def run_one(model_name: str, batch: int) -> dict:
    import torch
    from ultralytics import RTDETR

    model = RTDETR(model_name)
    trainer = model._smart_load("trainer")(overrides={
        "data": str(ROOT / "data/fasdd.yaml"),
        "imgsz": IMGSZ, "batch": batch, "epochs": 1, "workers": WORKERS,
        "optimizer": "AdamW", "lr0": 3e-4, "deterministic": False, "seed": 0,
        "cache": False, "plots": False, "val": False, "verbose": False,
        "project": str(OUT), "name": f"{Path(model_name).stem}_b{batch}",
        "exist_ok": True, "warmup_epochs": 0,
        "model": model_name, "compile": False,
    })
    trainer._setup_train()
    trainer.model.train()
    # _setup_train builds a val dataloader regardless of val=False; unused
    # here and it inflates memory.
    trainer.test_loader = None
    trainer.validator = None
    torch.cuda.empty_cache()

    n_params = sum(p.numel() for p in trainer.model.parameters())

    loader = trainer.train_loader
    it = iter(loader)
    stop_evt, gpu_samples = threading.Event(), []
    torch.cuda.reset_peak_memory_stats()
    n_done, t_start = 0, None

    try:
        for i in range(WARMUP_ITERS + TIMED_ITERS):
            try:
                batch_data = next(it)
            except StopIteration:
                it = iter(loader)
                batch_data = next(it)

            if i == WARMUP_ITERS:
                torch.cuda.synchronize()
                t_start = time.time()
                threading.Thread(target=sample_gpu,
                                 args=(stop_evt, gpu_samples),
                                 daemon=True).start()

            batch_data = trainer.preprocess_batch(batch_data)
            with torch.autocast("cuda", enabled=bool(trainer.amp)):
                loss, _ = trainer.model(batch_data)
            trainer.scaler.scale(loss.sum()).backward()
            trainer.optimizer_step()
            if i >= WARMUP_ITERS:
                n_done += 1
    except torch.cuda.OutOfMemoryError as e:
        stop_evt.set()
        return {"model": model_name, "batch": batch, "oom": True,
                "params_m": round(n_params / 1e6, 1), "error": str(e)[:160]}

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    stop_evt.set()
    time.sleep(0.6)

    return {
        "model": model_name,
        "batch": batch,
        "oom": False,
        "params_m": round(n_params / 1e6, 1),
        "iters": n_done,
        "elapsed_s": round(elapsed, 2),
        "images_per_s": round(n_done * batch / elapsed, 2),
        "ms_per_iter": round(1000 * elapsed / n_done, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "gpu_util_mean": round(statistics.mean(gpu_samples), 1) if gpu_samples else None,
        "amp": bool(trainer.amp),
    }


def main() -> None:
    # child mode: one config per process, GPU otherwise idle
    if len(sys.argv) > 2:
        res = run_one(sys.argv[1], int(sys.argv[2]))
        print("__RESULT__" + json.dumps(res))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    configs = [("rtdetr-l.pt", 16), ("rtdetr-x.pt", 16), ("rtdetr-x.pt", 8)]
    results = []

    for model_name, batch in configs:
        # Skip the smaller batch only if the larger one GENUINELY fit.
        #
        # "No OutOfMemoryError" is NOT evidence of fitting on WSL. Under
        # WDDM, allocations past physical VRAM spill silently into host RAM
        # over PCIe instead of raising: rtdetr-x at batch 16 reported 17.94GB
        # peak on a 16.38GB card, never raised, and ran 5.2x slower per
        # iteration. An earlier version of this loop keyed the skip on
        # `not r["oom"]` and therefore skipped the only X config that could
        # actually have run. Judge fit on measured peak VRAM instead.
        if model_name == "rtdetr-x.pt" and batch == 8 and any(
                r["model"] == "rtdetr-x.pt" and not r.get("oom")
                and r.get("peak_vram_gb", 99) < VRAM_FIT_GB for r in results):
            print(f"\n=== SKIP {model_name} batch {batch} "
                  f"(a larger batch fit under {VRAM_FIT_GB}GB) ===")
            continue

        print(f"\n=== {model_name} batch {batch} ===", flush=True)
        p = subprocess.run([sys.executable, "-u", __file__, model_name, str(batch)],
                           capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("__RESULT__")]
        if not line:
            print(p.stdout[-3000:])
            print(p.stderr[-3000:])
            results.append({"model": model_name, "batch": batch,
                            "error": "no result line"})
            continue
        res = json.loads(line[0][len("__RESULT__"):])
        results.append(res)
        print(json.dumps(res, indent=2), flush=True)

    (OUT / "probe_rtdetrx.json").write_text(json.dumps(results, indent=2))

    print("\n================ SUMMARY ================")
    base = next((r for r in results
                 if r.get("model") == "rtdetr-l.pt" and not r.get("oom")), None)
    for r in results:
        if r.get("oom"):
            print(f"{r['model']:14} b{r['batch']:<3} OOM ({r.get('params_m')}M params)")
            continue
        if "images_per_s" not in r:
            print(f"{r['model']:14} b{r['batch']:<3} FAILED: {r.get('error')}")
            continue
        rel = ""
        if base and r is not base:
            rel = (f"  ({r['images_per_s'] / base['images_per_s']:.2f}x img/s, "
                   f"{base['ms_per_iter'] and r['ms_per_iter'] / base['ms_per_iter']:.2f}x time/iter)")
        print(f"{r['model']:14} b{r['batch']:<3} "
              f"{r['params_m']:>6.1f}M  "
              f"{r['peak_vram_gb']:>5.2f} GB  "
              f"{r['images_per_s']:>7.2f} img/s  "
              f"{r['ms_per_iter']:>6.1f} ms/iter  "
              f"util {r['gpu_util_mean']}%{rel}")
    print(f"\nwrote {OUT / 'probe_rtdetrx.json'}")


if __name__ == "__main__":
    main()
