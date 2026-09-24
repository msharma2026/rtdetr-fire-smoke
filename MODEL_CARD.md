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

| model | params | GFLOPs | mAP50 | mAP50-95 | p50 | p99 | FPS |
|---|---|---|---|---|---|---|---|
| **RT-DETR-L (this model)** | **32.0M** | 105.3 | **0.837** | **0.486** | **5.94 ms** | **7.21 ms** | 168 |
| YOLOv5l | 46.1M | 107.7 | 0.797 | 0.468 | 6.08 ms | 9.95 ms | 165 |
| YOLOv5s | 7.0M | 15.8 | 0.785 | 0.445 | **3.19 ms** | 6.79 ms | **314** |

Latency: batch 1, 640×640, RTX 4080, median of 5 runs, each configuration in
its own process. **Every model at its own fastest setting** — fp16 + TF32 +
`torch.compile(mode="reduce-overhead")` (CUDA graphs) for all three. RT-DETR
additionally runs 100 queries and 4 decoder layers, measured separately as
accuracy-neutral (+0.0011 mAP50). Frames are sampled to match the test set's
52% positive rate; measuring on empty frames leaves NMS nothing to suppress
and understates the NMS-free advantage (YOLOv5s reads 1.70 ms there vs 3.19 ms
here).

Against YOLOv5l, the model of comparable compute (107.7 vs 105.3 GFLOPs),
RT-DETR is ahead on accuracy, parameters, median and tail. The tail is the
clearest: **7.21 ms vs 9.95 ms p99**, winning all 5 runs individually, with a
p99/p50 ratio of 1.21 against 1.64 — the NMS-free property showing up where it
should. It also returns ~45% more detections per frame (1.99 vs 1.37) at that
latency.

YOLOv5s is ~1.9× faster on the median for 5 points less mAP50. Its p99
(6.79 ms) is nonetheless close to RT-DETR's, and its run-to-run spread is
2.22–4.21 ms against RT-DETR's 5.90–6.39.

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
- **mAP50-95 is capped around 0.49, and the cause is small-object localisation
  rather than annotation quality.** The obvious hypothesis was that loose
  labels cap achievable IoU — smoke has no crisp boundary, so annotators
  disagree. That predicts smoke should lose precision faster than fire as the
  IoU threshold rises. It does not: fire degrades faster at *every* threshold
  (at IoU 0.80, fire retains 35.2% of its AP50 against smoke's 52.8%). Box size
  explains it — fire's median box is 0.53% of image area (47 px at 640, with
  65% of boxes under 1% of area) against smoke's 14.24% (242 px), and a fixed
  pixel error costs far more IoU on a small box. Reproduce with
  `scripts/iou_curve.py`.
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
