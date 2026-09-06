# RT-DETR-L — Fire & Smoke Detection

Two-class object detector (`fire`, `smoke`) built on RT-DETR-L, pretrained on
**FASDD** and fine-tuned on **D-Fire**.

- **Architecture:** RT-DETR-L (32.8M parameters, NMS-free)
- **Input:** 640×640 RGB
- **Classes:** `0: fire`, `1: smoke`
- **Framework:** Ultralytics 8.4.118 / PyTorch 2.5.1+cu121

---

## Results

D-Fire held-out test set (4,306 images):

| | mAP50 | mAP50-95 | fire mAP50 | smoke mAP50 |
|---|---|---|---|---|
| **RT-DETR-L** | **0.8352 ± 0.0008** | **0.4854 ± 0.0010** | 0.797 | 0.875 |

Mean ± standard deviation over 3 seeds. The same architecture initialised from
COCO instead of FASDD, and fine-tuned identically with its own tuned learning
rate, scores **0.7965 ± 0.0083** — so FASDD pretraining is worth **+0.0386
mAP50**, 95% CI [+0.0253, +0.0520].

Run-to-run variance differs by initialisation and is worth knowing before
reading any comparison: **σ = 0.0008** for FASDD-initialised runs, **σ =
0.0083** for COCO-initialised ones.

### Compared with the D-Fire authors' own detectors

[Venâncio et al.](https://github.com/pedbrgs/Fire-Detection), who built the
D-Fire dataset, released YOLOv5s/YOLOv5l fire detectors. Their published
weights, re-evaluated here on the same split under one protocol rather than
quoted from the paper:

| model | params | mAP50 | mAP50-95 | median | p99 | FPS |
|---|---|---|---|---|---|---|
| **RT-DETR-L (this model)** | 32.8M | **0.835** | **0.486** | 6.59 ms | **7.46 ms** | 152 |
| YOLOv5l | 46.1M | 0.797 | 0.468 | 7.44 ms | 15.69 ms | 134 |
| YOLOv5s | 7.0M | 0.785 | 0.445 | **4.46 ms** | 7.47 ms | **224** |

Latency: batch-1, end-to-end (letterbox → forward → decode/NMS), 640×640,
fp16 + CUDA graphs, RTX 4080, measured identically for all three.

Because RT-DETR is NMS-free, its latency is nearly independent of how many
objects are in frame: on the busiest test images latency rises **17%**, versus
**59%** for YOLOv5l. Its p99 sits at 1.13× its own median; YOLOv5l's is 2.1×.

> The YOLOv5 rows were scored with upstream `yolov5/val.py`, which rejects 8
> malformed labels in D-Fire's test set where Ultralytics rejects 4 (4,298 vs
> 4,302 images). About 0.1% of the set; it does not change the ordering.

---

## Files

| file | size | notes |
|---|---|---|
| `best.pt` | 66 MB | PyTorch checkpoint, load with Ultralytics |
| `best.onnx` | 110 MB | ONNX (opset 16), static 640×640 — runs without Ultralytics |

## Usage

```python
from ultralytics import RTDETR

model = RTDETR("best.pt")
results = model.predict("image.jpg", imgsz=640)
```

### Fast inference (≈152 FPS, batch-1)

Stock eager inference is CPU-dispatch-bound — fp16 alone buys nothing (0.98×)
and the GPU idles at roughly half its maximum clock. Enabling CUDA graphs
first, then fp16, gives **6.5×** with no accuracy change (0.8359 fp32 → 0.8365
fp16, inside the 0.0008 seed noise):

```python
import torch
from ultralytics import RTDETR
from patches import patch_hgblock_for_compile

patch_hgblock_for_compile()          # dynamo mistraces HGBlock without this
torch.backends.cuda.matmul.allow_tf32 = True

model = RTDETR("best.pt").model.cuda().eval().half()
model = torch.compile(model, mode="reduce-overhead")
```

Requires **static 640×640 input** (CUDA graphs capture fixed shapes — do not
use rectangular inference) and costs 27–42 s of compilation on the first call.

---

## Training

| | stage 1 (FASDD) | stage 2 (D-Fire) |
|---|---|---|
| images | 85,783 train / 9,531 val | 15,499 train / 1,722 val |
| epochs | 80 (converges by ~33) | 20 |
| imgsz / batch | 640 / 16 | 640 / 16 |
| optimizer | AdamW, `lr0=3e-4` | AdamW, `lr0=1e-4` |
| backbone | trained | unfrozen at 0.3× LR |

Three settings were corrected after measurement and matter if you retrain:

- **Do not freeze the backbone.** Unfreezing at 0.3× LR beats `freeze=10` by
  +0.028 mAP50 under both initialisations.
- **Stage 2 wants `lr0=1e-4`.** The obvious value (3e-5, inherited from stage-1
  tuning) costs a COCO-initialised arm 0.078 mAP50 — larger than the entire
  pretraining effect being measured.
- **20 epochs, not 30+.** Learning rate anneals over the *total* run length, so
  a longer run's mid-schedule checkpoint never completes its anneal. 60 epochs
  scores 0.0137 *below* 20.

---

## Limitations

- **Evaluated on one benchmark.** All numbers are D-Fire test. No third-party
  generalisation evaluation was performed — suitable candidate datasets were
  either paywalled, smoke-only, or likely repackaged from D-Fire/FASDD itself.
  Performance on imagery unlike D-Fire's is unmeasured.
- **Latency figures are RTX 4080 only.** No edge or Jetson measurements. The
  ONNX graph contains 18 `GridSample` ops (deformable attention) which are
  commonly unsupported on NPUs such as Rockchip's RKNPU — edge deployment is
  untested and may require trimming decoder layers and input resolution.
- **Smoke outperforms fire** (0.875 vs 0.797 mAP50). Fire boxes in D-Fire skew
  small and distant.
- **mAP50-95 is capped around 0.49** and appears annotation-limited rather than
  model-limited: smoke has no crisp boundary, and an independent published
  result lands on the same mAP50-95/mAP50 ratio (0.58).
- Four D-Fire test labels have out-of-bounds coordinates and are skipped by the
  evaluator.

---

## License and attribution

The model weights are released under **AGPL-3.0**, matching Ultralytics, whose
framework was used for training and inference. Commercial or closed-source use
— including operating it as a hosted service — requires a commercial licence
from Ultralytics. The ONNX export can be run without Ultralytics.

Training data:

- **D-Fire** — [gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset), CC0 1.0
- **FASDD** — Wang et al., *FASDD: An Open-access 100,000-level Flame and Smoke
  Detection Dataset for Deep Learning in Fire Detection*, Earth System Science
  Data, CC BY 4.0
