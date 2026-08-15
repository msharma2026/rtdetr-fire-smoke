# Fire/smoke detection — design decisions and reasoning

Two-stage RT-DETR detector: pretrain on FASDD (~95 k images), fine-tune on
D-Fire (21.5 k images).

This document records *why* each decision was made.
---

## 1. The datasets

### 1.1 What arrived

| | images | labels | native size (median) | notes |
|---|---|---|---|---|
| **FASDD_CV** | 95,314 | 95,314 | 718×540 | COCO/TDML/VOC/YOLO annotation formats |
| **D-Fire** | 21,527 | 21,527 | 1200×720 | train 17,221 / test 4,306 |

Both ship YOLO-format labels, which is why Ultralytics was viable with no
conversion step.


### 1.2 The class-ID mismatch

**FASDD uses `0=fire, 1=smoke`. D-Fire uses `0=smoke, 1=fire`.**

Neither dataset ships a `classes.txt` or `data.yaml`, and D-Fire's GitHub README
corrupts every label — the model would learn "fire" from smoke pixels and vice
versa, with nothing in the loss curve to indicate a problem.

**Resolution:** FASDD's order is canonical (`0: fire, 1: smoke`). D-Fire's labels
are rewritten to match. After swapping, published D-Fire figures matched.

### 1.3 Splits

FASDD ships a 50/33/17 train/val/test split. This was discarded.

**Reasoning:** stage 1 is feature-learning for a later fine-tune. Reserving a
third of the data for validation buys nothing here, so a conventional split
gives more training signal for the same eval cost. Re-split **stratified by
category** (fire / smoke / both / neither) so each split preserves the class
balance — a naive random split could skew categories between train and val.

**Then: does FASDD need a test set at all?** 10% from FASDD was folded into
train rather than split between train and val, because val's job (checkpoint
selection, overfitting detection) already saturates at 9,531 images: metric
noise falls as `1/√N`, so 9,531 → 14,297 buys only ~18% less noise — unlikely
to change which checkpoint gets picked.

**Final:**

| | train | val | test |
|---|---|---|---|
| FASDD | 85,783 (90%) | 9,531 (10%) | — |
| D-Fire | 15,499 | 1,722 | 4,306 (untouched) |

---

## 2. Model and framework

**RT-DETR-L via Ultralytics.** Both datasets already carry YOLO-format labels,
so this needs only two `data.yaml` files and no annotation conversion. RT-DETR
is anchor-free and NMS-free, and `rtdetr-l.pt` gives COCO-pretrained weights to
start from.

---

## 3. Training configuration

### 3.1 `optimizer="auto"` silently discards `lr0`

Ultralytics' default optimizer selection **ignores the learning rate you set**.

```
optimizer='auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' ...
optimizer: AdamW(lr=0.001667, momentum=0.9)
```

Stage 2's entire premise is a *lower* LR than stage 1. On `auto`, stage 2 would
have restarted D-Fire at whatever LR the heuristic picked, washing out the
FASDD features that stage 1 spent ~15 h learning — quietly making the two-stage
design pointless.

**Resolution:** `optimizer="AdamW"` is pinned explicitly. Confirmed working —
later runs log `AdamW(lr=0.0001)`.

### 3.2 Learning rate

`lr0=3e-4` (stage 1), `3e-5` (stage 2) — updated from the original `1e-4`/`1e-5`
after probing. 1e-4 was the DETR-family convention; a 5-point sweep
(3e-5 through 3e-3) found 3e-4 winning on train loss, mAP50, and
generalisation gap, with 1e-3 already behind and 3e-3 clearly unstable.
Ultralytics' auto-heuristic, for reference, picks ~1.7e-3, tuned for YOLO + SGD
and high for a DETR.

Stage 2 keeps the same 10× ratio to stage 1 it always had (now 3e-5, was
1e-5) — the same fine-tuning-safety-margin logic as before, **not**
independently probed. The LR sweep ran on FASDD only.

### 3.3 `deterministic=False`

Ultralytics defaults to `deterministic=True`, but RT-DETR uses ops with no
deterministic CUDA kernel (`grid_sampler_2d_backward`, `cumsum`), so PyTorch
warns and silently falls back anyway — paying a speed cost for determinism it
cannot deliver. Disabled, with `seed=0` for reproducibility.

### 3.4 Resolution and batch size

Peak reserved VRAM on the RTX 4080 (16 GB):

| imgsz | batch | peak VRAM | verdict |
|---|---|---|---|
| 640 | 16 | 12.4 GB | fits — **final choice, both stages** |
| 960 | 8 | 12.1 GB | fits |
| 960 | 10 | **15.9 GB** | too tight — <0.5 GB headroom |
| 1280 | 4 | 10.5 GB | fits |

Memory scales close to `fixed + per_image × batch` (~0.75 GB/image plus
~0.4 GB fixed at 640), so untested combinations can be projected from the
table — batch 24 at 640 lands near 18.4 GB and will not fit. Every figure
above is measured.

**640 resolution was chosen.** 640 beat 960 and 1280 on every metric — train
loss, mAP50, generalisation gap — on FASDD, and on D-Fire at both 4 and 14
epochs. 640/960's 14-epoch D-Fire runs both plateaued by epoch ~4-5 with no
further movement in either direction for the remaining 9-10 epochs, and
640's plateau sat above 960's throughout. 1280 could not be resolved: two
attempts at completing 14 epochs both timed out, and the two partial runs
disagreed with each other substantially at matching epochs under identical
config and seed.

### 3.5 Gradient accumulation

Ultralytics derives `accumulate = round(nbs / batch)` with `nbs=64`. At batch 16
that means **4 mini-batches are summed before each optimizer step**, holding the
effective batch at 64 (regardless of VRAM).

Mechanically: gradients *add* rather than overwrite, so skipping `zero_grad()`
between mini-batches lets them accumulate; the weights don't move until the
step, so every mini-batch in a group is evaluated against identical weights.

One consequence, discovered the hard way: changing `--batch` silently
changes *accumulation*, not the effective batch — and on very small datasets
that can starve a run of gradient steps entirely.

### 3.6 Backbone freezing (stage 2)

RT-DETR-L's module structure, read from the live model:

| modules | role | params |
|---|---|---|
| `model.0`–`model.9` | HGNetv2 CNN backbone | ~13.5 M (41%) |
| `model.10`–`model.27` | AIFI encoder + RepC3 fusion neck | ~12 M |
| `model.28` | `RTDETRDecoder` + heads | ~7.5 M |

`freeze=10` pins exactly the backbone. Stage 2 trains on a dataset
5.5× smaller than stage 1, and continued full-network training risks
catastrophic forgetting — overwriting general features with D-Fire-specific
ones. A low LR *slows* that drift; freezing *prevents* it for the
frozen layers. Since FASDD and D-Fire are the same task, low-level features
should transfer, so freezing them costs little — and 59% of the network (neck
+ decoder, the parts that turn features into detections) stays trainable.

### 3.7 Checkpoints

Each stage writes to its own run directory; stage 2 *reads* stage 1's `best.pt`
but never overwrites it. Both survive independently, so the FASDD-only and
FASDD+D-Fire models can be compared directly on the same D-Fire test set
(`evaluate.py --stage 1` / `--stage 2`). `best.pt` is the best-validation epoch,
not the last.

---

## 4. Hyperparameter probes

Rather than tune on the full ~15 h run, cheap probes run on fixed 1,200-image
train / 300-image validation slices, so runs within a group are directly
comparable. See `notebooks/experiments.ipynb` for the charts and
`scripts/experiments.py` for the definitions.

This approach was inspired by the Karpathy-style workflow.

### 4.1 Probe definitions

| group | varies | held fixed | question |
|---|---|---|---|
| `overfit` | — | 16 images, 200 epochs | does the pipeline learn *at all*? |
| `lr` | 3e-5 / 1e-4 / 3e-4 / 1e-3 / 3e-3 | imgsz 960, batch 8, FASDD | is 1e-4 in the right range? |
| `imgsz` | 640 / 960 / 1280 | **batch 4**, FASDD | does resolution help on FASDD, and what does it cost? |
| `imgsz_dfire` | 640 / 960 / 1280 | **batch 4**, D-Fire, 4 epochs | does resolution help where native detail is larger? |
| `imgsz_dfire_long` | 640 / 960 / 1280 | as above, 14 epochs | does a longer budget change the imgsz ranking? |
| `freeze` | 0 / 10 | imgsz 960, batch 8, D-Fire | does freezing hinder adaptation? |

Batch is deliberately **fixed at 4** across every resolution probe so `imgsz` is
the only variable (and 4 is the only batch that fits at 1280). The `lr` probe
started at 3 points and was extended to 5 after the first pass found 3e-4 beating
1e-4 with no sign of having peaked. The `imgsz_dfire` / `imgsz_dfire_long` groups
were added after the first `imgsz` pass ran on FASDD only, whose native resolution
is already close to 640 — that probe could show 640 winning but could never test
the actual hypothesis (D-Fire's larger native images having real detail to recover).

### 4.2 Results

See `notebooks/experiments.ipynb` for the charts, generated from the real
`results.csv` of each run. Headline numbers:

**Learning rate** (FASDD, 4 epochs, final-epoch mAP50):

| lr0 | mAP50 | note |
|---|---|---|
| 3e-5 | 0.0059 | |
| 1e-4 (current `train.py`) | 0.0151 | |
| **3e-4** | **0.0654** | best on loss, mAP50, *and* generalisation gap simultaneously |
| 1e-3 | 0.0490 | close behind 3e-4 |
| 3e-3 | 0.0035 | collapses — worse than 3e-5, classic too-high-LR shape |

3e-4 looks like a real local optimum in this range — 1e-3 is already worse
and 3e-3 is clearly unstable.

**Resolution** — 640 wins on FASDD (4 epochs) and on D-Fire (4 and 14 epochs),
960 also fully resolved and consistently behind 640, 1280 unresolved due
to measurement noise.


### 4.3 LR schedule shape (`cos_lr`)

Linear decay (Ultralytics default) vs cosine, 14 epochs, otherwise identical.

| | final mAP50 | epochs won |
|---|---|---|
| `cos_lr=False` (linear, default) | 0.6102 | 11/14 |
| `cos_lr=True` (cosine) | 0.6079 | 3/14 |

**Finding: no meaningful difference** (0.0023 apart at the end). **Decision:
keep the default**, no change to `train.py`.


### 4.4 Mosaic on D-Fire

Mosaic stitches 4 training images into one 2×2 composite canvas with random
scale/crop, remapping boxes. It is a strong regulariser (more objects, scales
and contexts per sample) but shrinks every object roughly 2× linearly, which
is why it was worth testing against D-Fire's small boxes (p10 width ≈26 px
native).

| epoch | 5 | 9 | 12 | 14 |
|---|---|---|---|---|
| mosaic on | 0.110 | 0.525 | 0.608 | 0.621 |
| mosaic off | 0.424 | 0.575 | 0.637 | 0.636 |
| gap | **+0.314** | +0.050 | +0.029 | **+0.015** |

Mosaic-off wins 12/14 epochs and finishes ahead, but the endpoint misleads:
**the gap is collapsing**, from +0.314 at epoch 5 to +0.015 at epoch 14 —
mosaic-on is steadily catching up, which is the textbook
mosaic pattern (harder early, pays off later) and precisely why
`close_mosaic` exists as a concept. Stage 2 runs 30 epochs, more than double
this probe. The final +0.015 is also below the noise floor established in
§4.3.

**Decision: no change — mosaic stays on.**

---

## 5. Open questions

- **Epoch count for stage 1.** 30 epochs ≈ 15 h. The probe was still
  improving at epoch 19-20 on a 3,000-image slice, so short budgets are clearly
  wrong, but it cannot settle 20 vs 30 at 85,783 images/epoch. `patience=15`
  is the practical safeguard.
- **Whether stage 2 helps at all.** The entire two-stage premise is untested
  until `evaluate.py --stage 1` vs `--stage 2` runs on D-Fire test.
- **Backbone freezing** (`freeze=10`). The probe could only test
  "does freezing hinder adaptation", not the actual question — whether it
  prevents forgetting FASDD features — because that requires a stage-1
  checkpoint to exist.

---

## 6. Throughput: what actually limits training speed

Investigated because GPU utilisation looked low mid-run.

### 6.1 The bottleneck is CUDA launch overhead

Profiling one training step directly (`torch.profiler`, batch 16, ~356 ms
wall) gives the breakdown:

| cost | calls/iter | CPU time | share |
|---|---|---|---|
| `cudaStreamSynchronize` | 163 | **99.9 ms** | 28% |
| `cudaLaunchKernel` | **7,443** | **91.7 ms** | 26% |
| everything else | | ~164 ms | 46% |

**Over half of every step is CPU dispatch overhead, not compute.** That single
fact explains why batch size, checkpointing and worker count all did nothing —
each addresses GPU compute or memory, while the constraint is the serial CPU
dispatch chain.

Everything else was ruled out by measurement:

| signal | value | rules out |
|---|---|---|
| `wa` (iowait) | **0%** | disk I/O |
| dataloader workers | **3.7% CPU each** | data loading |
| Hungarian matcher | scipy C++ solver, ~0.02-0.1ms/call (~1-3% of iteration) | DETR matching cost |
| main process | **106% CPU = exactly one core pegged** | — |
| GPU | 27-93%, oscillating | — |
| idle cores | 23 of 24 | — |

One saturated core feeding a starved GPU is the signature: PyTorch training is
a single-threaded Python loop issuing async GPU work, and DETR-family models
execute many *small* ops (6 decoder layers, multi-scale deformable attention,
many reshapes) rather than few large ones, so per-op Python and dispatcher
cost becomes the rate limiter. This cannot be spread across the 23 idle cores
— Python bytecode is serialised by the GIL, and op N+1 depends on op N.
Nothing is misconfigured: more workers, `cache='disk'`, or faster storage each
change nothing.

### 6.2 Batch-size scaling (`scripts/batch_scaling.py`)

Fixed iteration count per batch size, so bigger batches process proportionally
more images. Measured in **images/sec** (it/s falls trivially with batch size
and tells you nothing):

| batch | img/s | GPU% mean | GPU% max | VRAM |
|---|---|---|---|---|
| 8 | 32.51 | 51.4 | 76 | 7.2 GB |
| 12 | 41.05 | 67.2 | 92 | 10.09 GB |
| **16 (production)** | **45.78** | **74.0** | **95** | **13.33 GB** |

**1.41x throughput from batch 8 to 16.**

### 6.3 Why gradient checkpointing is not worth building

The same data answers it. Marginal returns are collapsing as the GPU fills:

- batch 8 -> 12: +50% batch for **+26%** throughput
- batch 12 -> 16: +33% batch for **+11.5%** throughput

Production already runs batch 16 at **74% mean / 95% peak** utilisation, so
only ~26% headroom remains. Pricing checkpointing against that: if it bought
batch 32 and utilisation reached ~88%, raw throughput would be
`45.78 x 88/74 ~= 54 img/s` — but checkpointing adds a full forward
recompute (+33% standard), giving `54 / 1.33 ~= 41 img/s`, **worse than batch
16**. Even optimistic selective checkpointing (+10%) yields ~49 img/s,
a ~7% gain for 2-3 hours of work carrying silent-wrong-gradient risk.

**Checkpointing arrives too late to matter here: batch 16 already captured
most of what batch scaling had to offer.** Also relevant: Ultralytics has
**no** gradient checkpointing in any version, so it would be written from
scratch against `_predict_once`, not adapted.

### 6.4 Optimizations applied

| change | img/s | vs baseline | notes |
|---|---|---|---|
| baseline | 44.48 | — | |
| `cudnn.benchmark=True` | 45.18 | 1.016x | autotunes conv algorithms; Ultralytics disables it only as an AutoBatch workaround we do not use |
| `channels_last=True` | 48.47 | 1.090x | NHWC is what cuDNN tensor-core kernels want; NCHW forces a transpose around every conv |
| **both** | **50.36** | **1.132x** | applied |
| **+ `compile="default"`** | **54.47** | **1.225x** | applied; also drops peak VRAM 13.5 -> 11.2 GB |

`channels_last` is bit-exact on CUDA (Ultralytics' "numerically wrong" caveat
in-source refers to MPS). `cudnn.benchmark` only selects among mathematically
equivalent algorithms. `compile` fuses kernels, so results differ by normal
floating-point reassociation.

**Unblocking `compile` took two patches** in `scripts/patches.py`:

1. `HGBlock.forward` uses `y.extend(m(y[-1]) for m in self.m)` — a generator
   reading `y[-1]` while `extend` appends to it. TorchDynamo mistraces it and
   feeds the original `x` to every block: *expected input[16, 128, 80, 80] to
   have 96 channels*.
   Rewritten as an explicit loop, verified **bit-for-bit identical**
2. TorchInductor's **vectorised CPU** codegen emits invalid C++
   (`'Vectorized<bool>' has no member named 'cast'`). That path is only reached
   because RT-DETR's loss `.item()`s data-dependent GT counts and graph-breaks
   a fragment to CPU. `torch._inductor.config.cpp.simdlen = 0` forces scalar
   CPU codegen and sidesteps it; GPU kernels come from Triton regardless.

---

## 7. Deferred experiments: the stage-2 A/B plan

Several questions are unanswerable by cheap probes but *are* cheaply
answerable once stage 1 exists, because the two stages have wildly asymmetric
cost:

| | images | ~cost at 640/batch16 | A/B-able? |
|---|---|---|---|
| Stage 1 (FASDD) | 85,783 | ~15 h | No — too expensive to fork |
| Stage 2 (D-Fire) | 15,499 | **~2.6 h** | **Yes** |

Once stage 1 has produced a checkpoint *once*, it can be forked into multiple
stage-2 variants for a couple of hours each. Crucially those are scored on
**D-Fire's real 4,306-image test set** rather than the probes' 300-image val
slice — ~14× more evaluation data, cutting measurement noise roughly 3.7×, on
top of 13× more training data, which removes the small-slice variance that makes
single-run probe differences hard to trust.

Worth running, in rough priority order:

1. **`freeze=10` vs `freeze=0`**
2. **mosaic on vs off**
3. **imgsz 1280 vs 640 for stage 2**
4. **`close_mosaic` timing** — only interesting if (2) shows mosaic mattering.

Also worth considering, not yet probed at all:

- **`patience` interaction with epoch count** — if stage 1 early-stops well
  before 30, that is itself the answer to the epoch-budget question.
- **Small-object breakdown.** The mosaic and resolution questions are really
  about small boxes specifically, but `results.csv` only exposes aggregate
  mAP.
