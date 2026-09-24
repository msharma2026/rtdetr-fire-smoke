"""Best-vs-best latency on a REPRESENTATIVE frame sample, with repeats.

Two things it fixes relative to bench_headtohead.py:

1. Frame sample. load_images() takes sorted(glob)[:40], which on D-Fire is
   0/40 positives -- every frame empty. NMS cost scales with candidate boxes,
   so an all-empty sample is YOLO's best case and understates RT-DETR. This
   samples to match the test set's own positive rate (~53%).

2. Repeats. YOLOv5l's p50 spans 3.13-6.44 ms across identical runs, so a
   single run cannot separate it from RT-DETR. Every config is repeated and
   the spread is reported alongside the central value.

Every model gets fp16 + TF32 + torch.compile(reduce-overhead) -- CUDA graphs
for all, not just RT-DETR. One config per process; warmup is wall-clock and
starts after compilation.

  python bench_optimized.py [repeats]     # driver
  python bench_optimized.py <config> run  # worker
"""
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "repos/fire_detection"
sys.path.insert(0, str(ROOT / "scripts"))

IMG = ROOT / "datasets/D-Fire/test/images"
LBL = ROOT / "datasets/D-Fire/test/labels"
CONFIGS = ["yolov5s_best", "yolov5l_best", "rtdetr_best", "rtdetr_best_tuned"]
GFLOPS = {"yolov5s": 15.8, "yolov5l": 107.7, "rtdetr": 105.3}
WARM_SECONDS, ITERS, N_IMGS = 8.0, 250, 40


def representative(n=N_IMGS):
    """n frames drawn to match the test set's own positive rate."""
    import numpy as np
    rng = np.random.default_rng(12345)
    files = sorted(IMG.glob("*.jpg"))

    def inst(f):
        p = LBL / f"{f.stem}.txt"
        return 0 if not p.exists() else len(
            [l for l in p.read_text().splitlines() if l.strip()])

    counts = {f: inst(f) for f in files}
    pos = [f for f in files if counts[f] > 0]
    neg = [f for f in files if counts[f] == 0]
    k = round(n * len(pos) / len(files))
    sel = list(rng.choice(pos, k, replace=False)) + \
          list(rng.choice(neg, n - k, replace=False))
    rng.shuffle(sel)
    return sel, k / n, sum(counts[f] for f in sel) / n


def worker(cfg):
    import cv2
    import numpy as np
    import torch
    import torch._dynamo
    sys.path.insert(0, str(ROOT / "scripts"))
    from bench_headtohead import pct, IMGSZ, YOLOV5, RTDETR_CKPT
    sys.path.insert(0, str(YOLOV5))
    from utils.augmentations import letterbox

    torch._dynamo.config.cache_size_limit = 64
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    files, pos_rate, density = representative()
    imgs = []
    for f in files:
        im = cv2.imread(str(f))
        if im is not None:
            imgs.append(letterbox(im, IMGSZ, stride=32, auto=False)[0])

    family = cfg.split("_")[0]

    def prep(im):
        return torch.from_numpy(np.ascontiguousarray(
            im.transpose((2, 0, 1))[::-1])).to("cuda").half().div_(255.0)[None]

    if family.startswith("yolov5"):
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression
        m = DetectMultiBackend(str(Path.home() / f"dfire-models/{family}.pt"),
                               device=torch.device("cuda"), fp16=True)
        m.warmup(imgsz=(1, 3, IMGSZ, IMGSZ))
        fwd = torch.compile(m, mode="reduce-overhead", dynamic=False)

        def run(im):
            o = fwd(prep(im))
            while isinstance(o, (list, tuple)):
                o = o[0]
            return non_max_suppression(o, 0.25, 0.45, max_det=300)
    else:
        from ultralytics import RTDETR
        from ultralytics.utils import ops
        import patches
        patches.patch_hgblock_for_compile()
        mod = RTDETR(str(RTDETR_CKPT)).model.cuda().eval().half()
        if cfg.endswith("tuned"):
            h = mod.model[-1]
            h.num_queries, h.decoder.eval_idx = 100, 3
        fwd = torch.compile(mod, mode="reduce-overhead", dynamic=False)

        def run(im):
            o = fwd(prep(im))
            o = o[0] if isinstance(o, (list, tuple)) else o
            return ops.xywh2xyxy(o[..., :4])[o[..., 4] > 0.25]

    with torch.no_grad():
        run(imgs[0])
        torch.cuda.synchronize()
        end = time.perf_counter() + WARM_SECONDS
        warm = 0
        while time.perf_counter() < end:
            run(imgs[warm % len(imgs)])
            warm += 1
        torch.cuda.synchronize()
        lat, ndet = [], []
        for k in range(ITERS):
            im = imgs[k % len(imgs)]
            t0 = time.perf_counter()
            out = run(im)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
            ndet.append(sum(len(d) for d in out) if isinstance(out, list) else len(out))

    r = pct(lat)
    r.update(config=cfg, family=family, pos_rate=round(pos_rate, 3),
             density=round(density, 2), dets=round(sum(ndet) / len(ndet), 3))
    print("__JSON__" + json.dumps(r))


def driver(reps):
    files, pos_rate, density = representative()
    print(f"sample: {len(files)} frames, {pos_rate:.0%} positive, "
          f"{density:.2f} GT instances/frame\n", flush=True)

    acc = {c: [] for c in CONFIGS}
    for i in range(reps):
        for cfg in CONFIGS:
            out = subprocess.run([sys.executable, __file__, cfg, "run"],
                                 capture_output=True, text=True, cwd=str(ROOT))
            ln = [l for l in out.stdout.splitlines() if l.startswith("__JSON__")]
            if not ln:
                print(f"  FAILED {cfg}: {out.stderr.strip()[-200:]}")
                continue
            r = json.loads(ln[0][len("__JSON__"):])
            acc[cfg].append(r)
            print(f"  rep{i+1} {cfg:<18} p50={r['p50_ms']:6.2f} "
                  f"p99={r['p99_ms']:6.2f} dets={r['dets']}", flush=True)

    print(f"\n__BEST-VS-BEST, {reps} reps, representative sample__")
    print(f"{'config':<18} {'p50 med':>8} {'p50 rng':>13} {'p99 med':>8} "
          f"{'FPS':>7} {'p99/p50':>8} {'dets':>6}")
    for cfg in CONFIGS:
        rs = acc[cfg]
        if not rs:
            continue
        p50 = [r["p50_ms"] for r in rs]
        p99 = [r["p99_ms"] for r in rs]
        mp50, mp99 = st.median(p50), st.median(p99)
        print(f"{cfg:<18} {mp50:>8.2f} {min(p50):>6.2f}-{max(p50):<6.2f} "
              f"{mp99:>8.2f} {1000/mp50:>7.1f} {mp99/mp50:>8.2f} "
              f"{rs[0]['dets']:>6.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        worker(sys.argv[1])
    else:
        driver(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
