"""Load the fire detector in its fast inference configuration.

WRITTEN BY CLAUDE (2026-09-06). Not the project author's code -- edit or
delete freely. See NOTES_BY_CLAUDE_2026-08-18.md section 10.

Eager inference is launch-bound: fp16 alone buys 0.98x (nothing), the GPU idles
at 52% of max SM clock, and p99 is 2.4x the median. CUDA graphs remove the
dispatch bottleneck, which makes the model compute-bound -- and only THEN does
fp16 pay (2.94x). Combined: 6.5x faster, p99/median 2.4x -> 1.19x, with mAP50
unchanged (0.8359 fp32 vs 0.8365 fp16, inside the 0.0008 seed noise floor).

    from scripts.load_optimized import load
    model = load()
    boxes = model(tensor)          # static 1x3x640x640, half, on cuda

CONSTRAINTS -- violating these silently loses the speedup:
  * Input MUST be a static 640x640 shape. CUDA graphs capture fixed shapes;
    rectangular/dynamic inference forces recapture or falls back to eager.
  * First call compiles for 27-42 s. Warm it before serving traffic.
  * Keep the model in eval(); train mode computes auxiliary decoder heads.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_CKPT = ROOT / "runs/seedvar_fasdd_s2/weights/best.pt"


def load(ckpt=None, half=True, compile_mode="reduce-overhead", warmup=True):
    """Return a compiled, fp16, eval-mode RT-DETR ready for 640x640 batches."""
    from patches import patch_hgblock_for_compile
    from ultralytics import RTDETR

    # Required: dynamo mistraces HGBlock's self-referential generator and
    # feeds the original x to every block. Verified bit-identical in eager.
    patch_hgblock_for_compile()

    # Inductor warns this is available but off by default; worth 1.69x on its
    # own for the fp32 path.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = RTDETR(str(ckpt or DEFAULT_CKPT)).model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if half:
        model = model.half()
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
        if warmup:
            x = torch.randn(1, 3, 640, 640, device="cuda",
                            dtype=torch.half if half else torch.float)
            with torch.no_grad():
                for _ in range(3):
                    model(x)
            torch.cuda.synchronize()
    return model


if __name__ == "__main__":
    import statistics as st
    import time

    m = load()
    x = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.half)
    with torch.no_grad():
        for _ in range(30):
            m(x)
        torch.cuda.synchronize()
        lat = []
        for _ in range(100):
            t0 = time.perf_counter()
            m(x)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    s = sorted(lat)
    print(f"median {st.median(s):.2f} ms  p99 {s[int(0.99*len(s))]:.2f} ms  "
          f"{1000/st.median(s):.1f} FPS")
