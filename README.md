# Fire & Smoke Detection with RT-DETR

Two-stage RT-DETR detector for fire and smoke, pretrained on **FASDD** and
fine-tuned on **D-Fire**: ran sequentially to measure domain shift.

The idea is coarse-to-fine transfer: FASDD (95k images) is large and varied
enough to learn general fire/smoke features, while D-Fire (21.5k images) is
smaller, curated, and contains ~9.8k negatives — good for sharpening
discrimination and suppressing false positives. Final evaluation is on D-Fire's
held-out test set.

---

## Results

D-Fire held-out test set (4,306 images), RTX 4080.

| | mAP50 | mAP50-95 | fire mAP50 | smoke mAP50 |
|---|---|---|---|---|
| **RT-DETR-L (this repo)** | **0.8352 ± 0.0008** | **0.4854 ± 0.0010** | 0.797 | 0.875 |

Mean ± sd over 3 seeds. FASDD pretraining is worth **+0.0386 mAP50**
(95% CI [+0.0253, +0.0520]).

### Against the D-Fire authors' own models

[Venâncio et al.](https://github.com/pedbrgs/Fire-Detection), who built the
D-Fire dataset, released YOLOv5s/YOLOv5l fire detectors. Their published
weights, re-evaluated here on the same test split rather than quoted:

| model | params | mAP50 | mAP50-95 | median | p99 | FPS |
|---|---|---|---|---|---|---|
| **RT-DETR-L (this repo)** | 32.8M | **0.835** | **0.486** | 6.59 ms | **7.46 ms** | 152 |
| YOLOv5l | 46.1M | 0.797 | 0.468 | 7.44 ms | 15.69 ms | 134 |
| YOLOv5s | 7.0M | 0.785 | 0.445 | **4.46 ms** | 7.47 ms | **224** |

Latency is batch-1, end-to-end (letterbox → forward → decode/NMS), 640×640,
fp16 + CUDA graphs, RTX 4080 — measured identically for all three, since
published latency figures are rarely comparable across papers.

RT-DETR is NMS-free, so its tail stays flat as detections accumulate: on the
busiest test frames latency rises **17%** against YOLOv5l's **59%**. That is
why it holds a 7.46 ms p99 while being 1.13× its own median, where YOLOv5l
sits at 2.1× its own.

Reproduce: `scripts/bench_headtohead.py` (latency), `scripts/evaluate.py`
(accuracy).

---

## Setup

Requires an NVIDIA GPU with CUDA. Developed on an RTX 4080 (16 GB) under
WSL2 (Ubuntu), Python 3.12, PyTorch 2.5.1+cu121, Ultralytics 8.4.118.

```bash
python -m venv venv
source venv/bin/activate
pip install ultralytics torch torchvision
pip install pandas jupyter matplotlib     # optional: experiment notebook
```

### Datasets

Download and extract both into `datasets/`:

| dataset | images | source |
|---|---|---|
| [FASDD](https://github.com/mmic-lcl/Datasets-and-benchmark-code) | 95,314 | Fire and Smoke Detection Dataset (CV subset) |
| [D-Fire](https://github.com/gaiasd/DFireDataset) | 21,527 | 17,221 train / 4,306 test |

Expected layout:

```
datasets/
├── D-Fire/
│   ├── train/{images,labels}/
│   └── test/{images,labels}/
└── archive/FASDD_CV/FASDD_CV/
    ├── images/
    └── annotations/YOLO_CV/labels/
```

Then build the staged copies Ultralytics expects:

```bash
python scripts/prepare_data.py
```

This symlinks images (no duplication), remaps D-Fire's class IDs, and writes
the split files and `data/*.yaml`. It never modifies `datasets/`.

> **Note:** the two datasets use **opposite class IDs** — FASDD is
> `0=fire, 1=smoke`, D-Fire is `0=smoke, 1=fire`. Neither ships a `classes.txt`
> and D-Fire's README doesn't state the order, so this is silent if missed.
> `prepare_data.py` normalises everything to FASDD's convention.

---

## Training

```bash
# Stage 1 — pretrain on FASDD (~13.6 h on an RTX 4080)
python scripts/train_watchdog.py --stage 1

# Stage 2 — fine-tune on D-Fire (~1.3 h at 20 epochs)
python scripts/train_watchdog.py --stage 2
```

`train_watchdog.py` wraps `train.py` and restarts from `last.pt` if the process
dies, which matters for a run this long. To run directly without supervision:

```bash
python scripts/train.py --stage 1
```

Useful flags: `--epochs`, `--batch`, `--imgsz`, `--lr0`, `--freeze`,
`--bblr` (backbone LR multiplier), `--seed`, `--lrf`, `--mosaic`, `--warmup`,
`--resume`, `--no-compile`, `--fraction` (train on a slice, for smoke tests).

### Evaluation

```bash
python scripts/evaluate.py --stage 2      # FASDD+D-Fire model on D-Fire test
python scripts/evaluate.py --stage 1      # FASDD-only, for the before/after
```

Both score against D-Fire's untouched 4,306-image test set and report
per-class mAP.

---

## Configuration

| | stage 1 (FASDD) | stage 2 (D-Fire) |
|---|---|---|
| images | 85,783 train / 9,531 val | 15,499 train / 1,722 val |
| epochs | 80 (converges by ~33) | 20 |
| imgsz / batch | 640 / 16 | 640 / 16 |
| optimizer | AdamW, `lr0=3e-4` | AdamW, `lr0=1e-4` |
| backbone | trained | unfrozen at 0.3× LR (`--bblr 0.3`) |

Most of these were chosen by measurement rather than convention — LR ladders
per arm, a resolution sweep on both datasets, and probes for `close_mosaic`,
`cos_lr`, and mosaic. See [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md).

Three settings changed after wider measurement and are worth calling out:

- **`freeze=10` was wrong.** Unfreezing the backbone at 0.3× LR beats freezing
  it by +0.028 mAP50 under both initialisations.
- **Stage 2 wants `lr0=1e-4`, not `3e-5`.** The original value was carried over
  from stage-1 tuning; on a COCO-initialised arm it costs 0.078 mAP50. Both
  arms were re-tuned with a ladder extended until the winner was bracketed.
- **20 epochs, not 30+.** Ultralytics anneals LR over the *total* run length,
  so a longer run's mid-schedule checkpoint never finishes its anneal — 60
  epochs scores 0.0137 *below* 20 (~17σ). Longer is actively worse here.

---

## Training throughput

RT-DETR training here is **CPU-dispatch-bound, not GPU-bound**: profiling shows
~7,443 kernel launches and ~163 stream synchronisations per iteration,
together more than half of each step. Only changes that reduce CPU work help.

| config | img/s | stage 1 |
|---|---|---|
| stock Ultralytics defaults | 44.5 | ~16.3 h |
| `+ channels_last + cudnn.benchmark` | 50.4 | ~14.6 h |
| `+ torch.compile` | **54.5** | **~13.6 h** |

`torch.compile` needs a small patch: Ultralytics' `HGBlock.forward` uses a
generator that reads `y[-1]` while `list.extend` appends to it, which
TorchDynamo mistraces (verified bit-identical when rewritten as a loop). See
`scripts/patches.py`.

Things that did **not** help *for training*, all measured: larger batch,
gradient checkpointing, more dataloader workers, bf16, and
`compile="reduce-overhead"` (13× slower — CUDA graph capture is skipped
because the loss keeps fragments on CPU).

> That last result is training-specific and does **not** generalise. Inference
> has no loss, so CUDA graphs capture cleanly there and are the single largest
> speedup available — see below.

## Inference latency

Stock eager inference is **CPU-dispatch-bound**, not compute-bound: fp16 buys
0.98× (nothing), the GPU idles at 52% of its max SM clock, and p99 runs 2.4×
the median. Fixing dispatch first makes precision matter again.

| config | median | p99 | speedup |
|---|---|---|---|
| eager fp32 | 33.92 ms | 52.43 ms | 1.00× |
| `+ torch.compile(mode="reduce-overhead")` | 15.26 ms | 17.55 ms | 2.22× |
| `+ TF32` | 9.05 ms | 11.05 ms | 3.75× |
| `+ fp16` | **5.19 ms** | **6.16 ms** | **6.54×** |

Batch-1, 640×640, forward only, RTX 4080. fp16 contributes 2.94× *here* versus
0.98× in eager mode — removing the launch bottleneck is what exposes the
arithmetic. Accuracy is unaffected (0.8359 fp32 → 0.8365 fp16, inside the
0.0008 seed noise).

```python
from scripts.load_optimized import load
model = load()          # fp16 + CUDA graphs, ~160 FPS, warm on first call
```

Requires static 640×640 input (CUDA graphs capture fixed shapes) and costs
27–42 s of compilation on the first call.

---

## Repository layout

```
scripts/
├── prepare_data.py      # stage datasets, remap class IDs, build splits
├── train.py             # stage 1 / stage 2 training
├── train_watchdog.py    # auto-restart wrapper for long runs
├── evaluate.py          # D-Fire test-set evaluation
├── experiments.py       # hyperparameter probe suite
├── batch_scaling.py     # training throughput / VRAM benchmarks
├── patches.py           # runtime patches to Ultralytics
├── build_notebook.py    # regenerate the experiment notebook
├── fair_control.sh      # LR ladder per arm, then matched 60-epoch runs
├── seed_variance.sh     # 3 seeds x 2 arms, for error bars
├── stage2_arms.sh       # {COCO,FASDD} x {freeze,bblr} factorial
├── test_ladder.sh       # unit tests for the LR-ladder logic (no GPU)
├── load_optimized.py    # fp16 + CUDA graphs inference config
├── optimize_inference.py # inference optimization sweep
├── profile_inference.py # batch-1 latency / VRAM / utilisation
├── bench_headtohead.py  # matched-protocol comparison vs external models
├── probe_rtdetrx.py     # RT-DETR-X viability probe
├── eval_external.py     # external-dataset eval with label/contamination audits
└── retention_seedvar.py # FASDD retention for the seed-variance checkpoints
DESIGN_DECISIONS.md      # full reasoning, measurements, dead ends
notebooks/experiments.ipynb
data/                    # generated by prepare_data.py (safe to delete)
datasets/                # raw downloads (never modified)
```

---

## Notes

- `data/` is fully regenerable; `datasets/` is the only thing worth backing up.
- WSL2 defaults to 50% of system RAM. Training was OOM-killed until
  `.wslconfig` was raised to `memory=20GB, swap=24GB`.
- `patches.py` modifies Ultralytics at runtime rather than editing
  `site-packages`, so it survives reinstalls.
