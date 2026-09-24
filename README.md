# Fire & Smoke Detection with RT-DETR

RT-DETR detector for fire and smoke, pretrained on **FASDD** and adapted to
**D-Fire**.

The idea is coarse-to-fine transfer: FASDD (95k images) is large and varied
enough to learn general fire/smoke features, while D-Fire (21.5k images) is
smaller, curated, and contains ~9.8k negatives — good for sharpening
discrimination and suppressing false positives. Final evaluation is on D-Fire's
held-out test set.

Stage 2 is now **joint** rather than sequential: instead of fine-tuning on
D-Fire alone, it continues from the FASDD checkpoint on a *pooled* set of both
datasets with D-Fire oversampled 16×. Fine-tuning on D-Fire alone costs FASDD
accuracy (0.8038 → 0.7550) — domain shift. Training on the pool instead holds
FASDD at **0.7901** at the same D-Fire accuracy, for about the same cost 
(2.59 h against 2.95 h). Note the epoch counts are not comparable — the pool
is 21× larger than D-Fire alone, so its 2 epochs are 41,720 gradient steps
against the sequential recipe's 19,380 over 20. It is a better model for the
same budget.

The sequential recipe is kept because it is the controlled arm the pretraining
ablation is measured against.

---

## Results

D-Fire held-out test set (4,306 images), RTX 4080.

| stage 2 recipe | D-Fire mAP50 | D-Fire mAP50-95 | FASDD val (retention) | train time |
|---|---|---|---|---|
| **joint pool, D-Fire ×16** | **0.8368** | **0.4866** | **0.7901** | 2.59 h |
| sequential, D-Fire only | 0.8352 ± 0.0008 | 0.4854 ± 0.0010 | 0.7550 | 2.95 h |

Sequential row is mean ± sd over 3 seeds.
FASDD pretraining is worth **+0.0386 mAP50** (95% CI [+0.0253, +0.0520]),
measured on the sequential arm.

The joint row is a **single run**, so its +0.0016 D-Fire edge sits at ~2σ of
the sequential arm's seed noise and should be read as *matching* sequential
accuracy, not beating it. The retention difference is the part that is not
marginal: every sequential configuration measured, including one tuned
specifically to protect FASDD (`freeze=10`, 0.7835), lands below the joint
run's 0.7901.

### Against the D-Fire authors' own models

[Venâncio et al.](https://github.com/pedbrgs/Fire-Detection), who built the
D-Fire dataset, released YOLOv5s/YOLOv5l fire detectors. Their published
weights, re-evaluated here on the same test split rather than quoted:

| model | params | GFLOPs | mAP50 | mAP50-95 | p50 | p99 | FPS |
|---|---|---|---|---|---|---|---|
| **RT-DETR-L (this repo)** | **32.0M** | 105.3 | **0.837** | **0.486** | **5.94 ms** | **7.21 ms** | 168 |
| YOLOv5l | 46.1M | 107.7 | 0.797 | 0.468 | 6.08 ms | 9.95 ms | 165 |
| YOLOv5s | 7.0M | 15.8 | 0.785 | 0.445 | **3.19 ms** | 6.79 ms | **314** |

**Every model at its own fastest configuration** — fp16 + TF32 +
`torch.compile(mode="reduce-overhead")`, i.e. CUDA graphs for all three, not
just ours. RT-DETR additionally runs 100 queries and 4 decoder layers, which
is separately measured as accuracy-neutral (+0.0011 mAP50) and has no YOLO
equivalent. Batch 1, 640×640, RTX 4080; median of 5 runs, each config in its
own process. **FPS is derived from p50, not the mean** — YOLOv5s's tail is
heavy enough (p99/p50 = 2.13) that its mean understates typical throughput by
roughly a third.

Two measurement choices that materially change the numbers:

- **Frames are sampled to match the test set's 52% positive rate.** Taking the
  first 40 files instead gives 0/40 positives, which leaves NMS nothing to
  suppress and understates any NMS-free advantage. On empty frames YOLOv5s
  measures 1.70 ms; on representative frames, 3.19 ms.
- **Each config runs in a separate process.** CUDA-graph capture reserves an
  allocator pool that slows anything measured afterwards in the same process —
  enough to move YOLOv5s between 248 and 123 FPS purely by reordering.

Against YOLOv5l — the comparable model at 107.7 vs 105.3 GFLOPs — RT-DETR wins
on accuracy, parameters, median and tail. The tail is the decisive one:
**7.21 ms vs 9.95 ms p99**, winning in all 5 runs individually, with a
p99/p50 ratio of 1.21 against 1.64. RT-DETR also returns ~45% more detections
per frame (1.99 vs 1.37) at that latency.

YOLOv5s remains ~1.9× faster on the median and is the right choice if 5 points
of mAP50 are affordable. Note its p99 (6.79 ms) is close to RT-DETR's despite
the median gap, and its run-to-run spread is 2.22–4.21 ms against RT-DETR's
5.90–6.39.

Reproduce: `scripts/bench_optimized.py` (latency), `scripts/evaluate.py`
(accuracy). `scripts/bench_headtohead.py` holds the older unoptimised
comparison.

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

# Stage 2 (joint, current recipe) — build the pooled set, then train on it
python scripts/prepare_joint_data.py --ratio 16
python scripts/train.py --stage 2 --data joint_r16.yaml \
    --weights runs/stage1_fasdd/weights/best.pt \
    --freeze 0 --bblr 0.3 --lr0 1e-4 --epochs 2

# Stage 2 (sequential, the ablation control) — D-Fire only, ~2.9 h at 20 epochs
python scripts/train_watchdog.py --stage 2
```

`prepare_joint_data.py --ratio 16` writes a pool of 333,767 entries — FASDD's
85,783 plus D-Fire's 15,499 repeated 16× — so D-Fire is ~74% of what the model
sees. The ratio was chosen by a ladder that extended until the peak was
interior (1 → 5.5 → 11 → **16** → 22, where 22 declined), and `lr0=1e-4` by a
second ladder at the winning ratio (3e-5 scored lower; 3e-4 diverged to NaN).

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

| | stage 1 (FASDD) | stage 2 joint | stage 2 sequential |
|---|---|---|---|
| train images | 85,783 | 333,767 pooled | 15,499 |
| val | 9,531 (FASDD) | 1,722 (D-Fire) | 1,722 (D-Fire) |
| epochs | 80 (converges by ~33) | 2 | 20 |
| imgsz / batch | 640 / 16 | 640 / 16 | 640 / 16 |
| optimizer | AdamW, `lr0=3e-4` | AdamW, `lr0=1e-4` | AdamW, `lr0=1e-4` |
| backbone | trained | unfrozen at 0.3× LR | unfrozen at 0.3× LR |

Both stage-2 variants start from the same stage-1 checkpoint. 2 epochs on the
pool is more gradient steps than 20 on D-Fire alone, since the pool is ~21×
larger — the epoch counts are not comparable, the step budgets are.

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
├── prepare_joint_data.py # build the pooled FASDD + oversampled D-Fire set
├── train.py             # stage 1 / stage 2 training
├── train_watchdog.py    # auto-restart wrapper for long runs
├── evaluate.py          # D-Fire test-set evaluation
├── experiments.py       # hyperparameter probe suite
├── batch_scaling.py     # training throughput / VRAM benchmarks
├── patches.py           # runtime patches to Ultralytics
├── build_notebook.py    # regenerate the experiment notebook
├── fair_control.sh      # LR ladder per arm, then matched 60-epoch runs
├── joint_ratio_ladder.sh # oversampling-ratio search for the joint pool
├── joint_ratio_extend.sh # extends that search until the peak is interior
├── joint_lr_ladder.sh   # LR ladder at the winning ratio
├── ensemble_eval.py     # weighted box fusion across seeds, via torchmetrics
├── render_video.py      # draw predictions onto a video
├── seed_variance.sh     # 3 seeds x 2 arms, for error bars
├── stage2_arms.sh       # {COCO,FASDD} x {freeze,bblr} factorial
├── test_ladder.sh       # unit tests for the LR-ladder logic (no GPU)
├── test_dynamo_trace.py # shows which Ultralytics blocks torch.compile breaks
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
