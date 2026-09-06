# Open items, untested leads, and known mistakes

Written while stage 1 was training. Everything here is either **unverified**,
**deferred**, or **a documented error in how this project was run**. Settled
decisions and their evidence live in `DESIGN_DECISIONS.md`; this file is
deliberately the opposite — what is wrong, missing, or unexamined.

Status at time of writing: stage 1 at epoch 18/30, 23.4 min/epoch,
261 ms/iteration, GPU 60-77%, FASDD val mAP50 0.733 / mAP50-95 0.470.

---

## 1. Bugs and integration gaps in the current code

| # | issue | severity | file |
|---|---|---|---|
| 1.1 ✅ | `use_bf16()` and `use_fused_adamw()` take a `trainer` object, but `train.py` only calls `model.train(...)`, which builds the trainer internally and runs atomically. **There is no point at which these can be called.** They are unreachable by construction. Fix: Ultralytics callbacks (`on_pretrain_routine_end`). | high | `patches.py`, `train.py` |
| 1.2 ✅ | `train_watchdog.py` hardcodes `run_dir = RUNS / STAGE_RUN[args.stage]` but forwards `--name` to `train.py`. Pass `--name` and the watchdog watches the wrong directory, reads 0 epochs, and aborts on first restart as "no progress". | high | `train_watchdog.py` |
| 1.3 ✅ | `enable_activation_checkpointing()` patches `BaseModel._predict_once`, which `RTDETRDetectionModel` overrides. Dead code for this repo. Delete or mark YOLO-only. | low | `patches.py` |
| 1.4 | Segment checkpointing double-updates BatchNorm running statistics (182/182 buffers inside segments drift; 62/62 outside match exactly). Training gradients are unaffected — BN normalises with batch stats — but *inference-time* stats skew. Must be fixed before any real training run uses checkpointing. | blocks 4.3 | `patches.py` |
| 1.5 | `freeze=10` + `compile` has never run together. Frozen layers set `requires_grad=False`, changing the graph dynamo traces. Stage 2 is the first use. Smoke-test with `--stage 2 --epochs 1 --fraction 0.01` first. | medium | — |

---

## 2. Performance: what is still on the table

Current: **261 ms/iteration, GPU 60-77% idle-ish, ~11.6 h for stage 1.**
Theoretical floor if dispatch overhead vanished: **~157 ms (~7 h)**.
Realistic target: **~210 ms (~9 h)**.

Profiled breakdown per iteration:

| cost | calls | time | share |
|---|---|---|---|
| `cudaStreamSynchronize` | 163 | 99.9 ms | 28% |
| `cudaLaunchKernel` | **7,443** | 91.7 ms | 26% |
| compute + other | | ~164 ms | 46% |

### 2.1 Separate forward compilation from the loss — **tested: no benefit**

Ultralytics does `attempt_compile(self.model, ...)` on the whole
`DetectionModel`. In training, `model(batch)` routes through `.loss()`, so the
loss lands *inside* the compiled region, and its `.item()` calls on
data-dependent GT counts graph-break it. This is why
`compile="reduce-overhead"` logged **37× "skipping cudagraphs due to cpu
device"** and ran 13× slower.

Fix: compile `predict()` (static shapes, no syncs) and leave the loss eager,
so the forward becomes CUDA-graph-capturable. Requires `drop_last=True` for
constant batch shape. This is the single change most likely to close the gap,
and would also be a stronger Ultralytics contribution than the HGBlock fix.

**Outcome (measured):** implemented as `compile_forward_only()` in
`patches.py` and benchmarked against whole-model `compile="default"`:
54.82 vs 54.7 img/s — **+0.2%, i.e. nothing.** Default-mode compile was
already fusing the same regions; restructuring what gets compiled changed
nothing. With `mode="reduce-overhead"` the backbone-only region *still*
failed: CUDA graph memory pools pushed VRAM to **17.18 GB on a 16.4 GB
card** and throughput collapsed to 3.87 img/s. Two different failure modes
across two attempts — **CUDA graphs are closed on this hardware**, not
merely blocked by the loss.

### 2.2 Deformable attention runs as a Python loop — **untested, likely large**

`multi_scale_deformable_attn_pytorch` (`nn/modules/utils.py:101`) is a pure
PyTorch fallback that loops over feature levels calling `F.grid_sample`,
rather than the fused CUDA kernel the official RT-DETR ships. Executed in
every decoder layer, every iteration. Given the bottleneck is 7,443 launches,
this is a strong suspect for a large share of them — and it is RT-DETR-specific,
which is why generic optimisation advice keeps missing it.

### 2.3 `need_weights=True` blocks MultiheadAttention's fast path — **adopted**

```python
src2 = self.ma(q, k, value=src, attn_mask=..., key_padding_mask=...)[0]
```

`need_weights` defaults to `True` and the weights are immediately discarded by
`[0]`. That flag disables the fused SDPA path, forcing the unfused math kernel
that materialises the full attention matrix. Passing `need_weights=False` is a
one-word change. Applies to AIFI (`transformer.py:115,142`) and the decoder
(`:698`).

### 2.4 Fused AdamW — **root-caused and adopted**

575 parameter tensors stepped as individual eager kernels every iteration.
The earlier crash was diagnosed: the old optimizer's param_groups carry
per-group `fused: None / foreach: None` keys that silently override the
constructor's `fused=True`, producing a non-fused optimizer that trips
GradScaler's `found_inf` assert. Fixed by rebuilding groups with only
hyperparameter keys. Now applied in production via an
`on_pretrain_routine_end` callback (closing bug 1.1).

**Combined outcome:** TF32 + MHA fast path + single-sync loss + fused AdamW
measured **58.06 vs 54.7 img/s (+6.1%)** over production, stage 1
~11.6 h → **~10.9 h**. All four are wired into `train.py`. Individual
contributions were not isolated; all four are harmless and stack.

### 2.5 Lower-value / measured-marginal

- **TF32** (`torch.set_float32_matmul_precision("high")`) — one line, but under
  AMP most matmuls are already fp16, so expect small. `cudnn.allow_tf32` is
  already on by default.
- **Validation frequency** — ~53 s of a 1,398 s epoch (**3.8%**, ~26 min over
  30 epochs). Ultralytics has no `val_period` arg; needs a callback.
- **bf16** — measured **8% slower** standalone (46.43 vs 50.36 img/s) because
  cuDNN's fp16 conv paths are better tuned on Ada. **But it was never tested
  with `compile`,** and its real value may be removing GradScaler's CPU
  control flow that blocks CUDA graph capture (see 2.1).
- **`bincount` for `gt_groups`** — measured ~0 alone (removes 16 calls at
  ~0.06 ms each), but removes syncs that may block graph capture.

### 2.6 Ruled out by measurement — do not revisit without new evidence

| lever | evidence |
|---|---|
| More dataloader workers | flat at 4/8/16 (44.6/44.7/45.4 img/s); workers at 3.7% CPU |
| Faster storage, `/dev/shm`, pillow-simd | `iowait = 0%`; dataloading is not the constraint |
| Larger batch | 1.41× from 8→16, then flat; GPU saturates |
| Gradient checkpointing | works and frees 45% VRAM, but costs 2-7% throughput and buys nothing when CPU-bound |
| `compile="reduce-overhead"` | failed twice: whole-model 3.92 img/s (loss graph-breaks); backbone-only 3.87 img/s (graph pools exceed 16 GB). Closed on this hardware |

---

## 3. Model and training decisions that were under-examined

### 3.1 RT-DETR-L vs RT-DETR-X — **deferred: future improvement**

Explicitly deferred; **not blocking the 80-epoch stage-1 run**, which
proceeds on RT-DETR-L. Revisit after stage 2, when there is a finished
L-based result to compare against.

Chosen in one sentence on the grounds that X "roughly doubles" epoch time.
That estimate assumed compute scales to wall-clock, i.e. **GPU-bound
execution** — which is exactly what this project is not.

| | L | X |
|---|---|---|
| params | 32.8M | **67.3M** (2.05×) |
| top-level layers | 29 | 33 (**1.14×**) |

X is mostly *wider*, not deeper. In a launch-bound regime kernel count scales
with depth while work-per-kernel scales with width, so X would issue roughly
the same launches doing ~2× the work — filling idle GPU rather than adding
proportional wall-clock. Real cost plausibly **1.3-1.5×, not 2×**.

Unchecked counterweight: VRAM. L uses 11.2 GB with compile; X at batch 16 may
exceed the 16 GB card and force batch 8, which would worsen the dispatch
problem. **Never measured.** A 30-second probe settles it.

This is the clearest misprioritisation in the project: hours spent optimising
throughput to save ~2 h of wall-clock, while the decision that directly bounds
final accuracy got one sentence and no measurement.

### 3.2 No accuracy baseline was ever established

Throughput was benchmarked extensively; **model quality was never compared to
anything.** Nobody asked what published models achieve on FASDD or D-Fire, so
the current FASDD val mAP50 0.733 / mAP50-95 0.470 has no referent. Look up
published D-Fire numbers before interpreting `evaluate.py` output.

### 3.3 Epoch budget is probably generous

Real-run curve: epochs 1-5 gained +0.367 mAP50-95; epochs 13-18 gained +0.016.
Extrapolating, 30 epochs lands near 0.48-0.49 — the final 12 epochs buy ~+0.015
for ~4.7 h. **20-22 epochs would capture ~97% of the result in two-thirds the
time.** `patience=15` cannot trigger here because the model improved on every
epoch 1-18.

### 3.4 Freezing depth is reasoned, not measured

`freeze=10` (the HGNetv2 backbone, ~13.5M params / 41%) is a defensible split
from the module structure, but 6 or 12 were never compared. The probe that
tried to test it was inconclusive — both arms landed at mAP50 ≈ 0.0003 — and
it could not test the real question anyway, since it started from COCO weights
rather than a FASDD checkpoint.

### 3.5 Resolution 1280 is untested, not ruled out

Two probe attempts timed out and their partial runs disagreed at matching
epochs. 640 was chosen on 640-vs-960's clean comparison, not on evidence
against 1280. Segment checkpointing (which works, frees 45% VRAM) makes 1280
testable at batch 8-16 where it previously only fit at batch 4 — **but 1.4
must be fixed first.**

---

## 4. Deferred experiments (cheap once stage 1 exists)

Stage 2 is only ~2 h, so forking a stage-1 checkpoint into variants is cheap.
Crucially these score on **D-Fire's real 4,306-image test set**, not the
probes' 300-image val slice — ~14× more evaluation data.

1. **`evaluate.py --stage 1` vs `--stage 2`** — the before/after that tests
   whether the two-stage premise paid off at all. Without it there is a number
   but no evidence the D-Fire fine-tune helped.
2. **`freeze=10` vs `freeze=0`** — the decisive version of 3.4, and the only
   test that can address catastrophic forgetting of FASDD features.
3. **imgsz 1280 vs 640 for stage 2** — see 3.5.
4. **mosaic on/off** — §4.4 found the gap collapsing (+0.314 → +0.015 over 14
   epochs); 30 epochs at full scale may reverse it entirely.
5. **Per-class metrics** — everything so far is aggregate mAP. Fire and smoke
   likely behave very differently (smoke is amorphous and hard to bound), and
   `evaluate.py` already reports them separately.
6. **Size-stratified eval** — the mosaic and resolution questions are really
   about *small boxes*, but only aggregate mAP was ever measured.
7. **RT-DETR-X VRAM + throughput probe** (see 3.1) — ~2 min: load X, run ~20
   training steps at batch 16, print peak VRAM and img/s. Decides whether X is
   viable at all. If it fits under ~15 GB and costs <1.5x, X is likely the
   better model for a dispatch-bound pipeline. Caveat: all twelve
   hyper-parameter probes were run on L, so `bblr=0.3` would be inherited
   rather than re-measured (same HGNetv2 backbone, so it should transfer).

---

## 5. Upstream contributions

### 5.1 HGBlock / torch.compile — ready to submit

`HGBlock.forward` uses `y.extend(m(y[-1]) for m in self.m)` — a generator
reading `y[-1]` while `list.extend` appends to it. Correct in eager; dynamo
mistraces it and feeds the original `x` to every block. Confirmed still
present on upstream `main`. Fix verified **bit-for-bit identical**
(0.000e+00 max output difference) and unblocks `compile` for +8.2%.

**Untested and must not be claimed without evidence:** the same pattern
appears **12 times across 9 classes**, including `C2f` and `SPPF` — core
YOLOv8/v11 blocks. If they also mistrace, `compile` is quietly broken far more
widely than RT-DETR. Test with `torch.compile(module, backend="eager")` on
CPU (dynamo traces, no inductor, no GPU) before scoping the PR.

### 5.2 Not Ultralytics' bug

`cpp.simdlen = 0` works around a **PyTorch inductor** codegen bug
(`'Vectorized<bool>' has no member named 'cast'`). Keep it out of any
Ultralytics PR — conflating them muddies the fix.

---

## 6. Process mistakes worth not repeating

These are failures in method, not in any single number.

**Measured the wrong axis for hours.** Correctly diagnosed CPU-dispatch
overhead early, then tested batch size, checkpointing, workers, and bf16 —
none of which address dispatch. Each negative result increased confidence that
a ceiling had been reached, when they were really evidence of testing the
wrong variable. Four failed experiments on the wrong axis do not triangulate a
floor.

**Treated a diagnosis as an excuse.** Once "CPU-bound" was established it
started functioning as an explanation for every subsequent failure rather than
as a problem to solve. It is descriptive, not exculpatory.

**Missed a lead that was explicitly labelled.** `reduce-overhead` failing with
37× "skipping cudagraphs due to **cpu device**" names the blocker and its
cause. It was filed as "CUDA graphs don't work here" instead of "CUDA graphs
are blocked by three fixable things." The `bincount` patch was even written
and then dismissed for not helping throughput — never connecting that its
value was unblocking capture.

**Anchored on the baseline instead of the hardware.** 44 → 54 img/s felt like
success, so 60% GPU utilisation was graded against the starting point rather
than against what the card can do. The right question was "what would a
well-optimised version look like?", not "how much better than before is this?"

**Validated mechanisms, not integration.** `use_bf16` and `use_fused_adamw`
were tested in a harness that constructs the trainer manually. They work
there. They are uncallable from `train.py`. The docstrings even say "call
after `trainer._setup_train()`" without anyone noticing there is no such
moment in production.

**Never re-read the production files cold.** Every change to `train.py` was a
targeted edit, each verified in isolation. The file was never read end-to-end
afterwards asking "does this actually use what was built?" A single fresh-eyes
pass found the integration gaps immediately.

**Four measurements produced confident, meaningless numbers** before being
caught — unpinned batch (different data per run), relative error on near-zero
gradients, patching a method the model never calls, and reading bit-identical
output as success rather than as proof the patch was inert. The validation
checklist derived from these is in `DESIGN_DECISIONS.md` §8; it was written for
*measurements* and then not applied to *code*.

**Prioritised the measurable over the important.** Throughput is easy to
benchmark, so it got hours. Model capacity (3.1) and accuracy baselines (3.2)
are harder to benchmark and directly determine whether the final artifact is
any good — they got one sentence and zero measurements respectively.
