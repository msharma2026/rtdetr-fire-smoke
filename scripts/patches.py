"""Runtime patches to Ultralytics, applied from our code rather than by
editing site-packages (so they survive reinstalls and stay reviewable).

Currently one patch: make HGBlock traceable by torch.compile.
"""

import torch
from torch.nn import functional as nn_fn  # noqa: F401  (used inside patched forward)


def patch_hgblock_for_compile() -> bool:
    """Rewrite HGBlock.forward so TorchDynamo can trace it.

    Upstream (ultralytics/nn/modules/block.py):

        y = [x]
        y.extend(m(y[-1]) for m in self.m)

    `list.extend` consumes the generator lazily, appending as it goes, so each
    `y[-1]` read sees the tensor appended by the previous step -- a sequential
    chain where block i consumes block i-1's output. That is correct in eager
    mode, but TorchDynamo mistraces the read-while-mutating pattern and feeds
    the ORIGINAL x to every block, which surfaces as:

        expected input[16, 128, 80, 80] to have 96 channels, but got 128

    The explicit loop below is semantically identical and traces cleanly.
    Returns True if the patch was applied.
    """
    from ultralytics.nn.modules.block import HGBlock

    if getattr(HGBlock, "_compile_patched", False):
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer (compile-traceable)."""
        y = [x]
        for m in self.m:
            y.append(m(y[-1]))
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

    HGBlock.forward = forward
    HGBlock._compile_patched = True
    return True


def enable_activation_checkpointing(model, layers="all"):
    """Wrap per-layer forwards in torch.utils.checkpoint to trade compute for VRAM.

    Ultralytics has no gradient checkpointing for any model (verified against
    upstream main -- the only torch.utils.checkpoint usage is inside the
    vendored SAM encoder), so this patches the layer loop in
    BaseModel._predict_once.

    Why here and not on the transformer blocks: RT-DETR-L is a *hybrid*. AIFI
    is a single 789K-param encoder layer operating on a 20x20 grid, while
    model.0-model.9 (HGNetv2) process 640->320->160->80 feature maps. Almost
    all activation memory is in the CNN stages, so checkpointing only the
    transformer -- the standard advice for ViTs -- would free nearly nothing
    here.

    Args:
        model: an Ultralytics BaseModel (e.g. RTDETR(...).model)
        layers: "all"      -> every layer holding parameters
                "backbone" -> model.0-model.9 only (the memory-heavy CNN)
                iterable   -> explicit layer indices

    Gotchas handled:
      * use_reentrant=False (the reentrant version misbehaves with autograd
        hooks and unused parameters, and requires grad-carrying inputs).
      * Training only -- checkpointing during eval/val is pure overhead with
        no backward pass to save memory for.
      * Skips parameterless layers (Concat, Upsample): checkpoint saves a
        layer's *internal* activations, so a layer with no internals saves
        nothing while still paying recompute.
      * Layers taking a LIST of inputs (Concat-style `m.f` lists) must be
        unpacked -- checkpoint takes varargs tensors, not a list.
    """
    import torch.utils.checkpoint as cp
    from torch.nn import functional as nn_fn
    from ultralytics.nn.tasks import BaseModel

    n_layers = len(model.model)
    if layers == "all":
        idx = {i for i, m in enumerate(model.model) if any(True for _ in m.parameters())}
    elif layers == "backbone":
        idx = {i for i in range(min(10, n_layers))
               if any(True for _ in model.model[i].parameters())}
    else:
        idx = set(layers)
    model._ckpt_layers = idx

    if getattr(BaseModel, "_ckpt_patched", False):
        return idx

    def _predict_once(self, x, profile=False, embed=None):
        """_predict_once with optional per-layer activation checkpointing."""
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        ckpt = getattr(self, "_ckpt_layers", None) or set()
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            if self.training and m.i in ckpt:
                if isinstance(x, list):
                    # unpack: checkpoint takes *tensors, and rebuilding the
                    # list inside keeps the layer's own signature intact
                    x = cp.checkpoint(lambda *a, _m=m: _m(list(a)), *x, use_reentrant=False)
                else:
                    x = cp.checkpoint(m, x, use_reentrant=False)
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if m.i in embed:
                embeddings.append(nn_fn.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    BaseModel._predict_once = _predict_once
    BaseModel._ckpt_patched = True
    return idx


def patch_rtdetr_loss_syncs() -> bool:
    """Replace the per-image .item() loop in RTDETRDetectionModel.loss.

    Upstream (nn/tasks.py):

        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]

    Each .item() is a full GPU->CPU synchronisation -- 16 pipeline stalls per
    iteration at batch 16, measured as part of the 163 cudaStreamSynchronize
    calls costing ~100ms/iter (28% of step time). One bincount + one .tolist()
    produces the identical list of ints with a single sync. The rest of the
    method is reproduced verbatim from upstream 8.4.118.
    """
    from ultralytics.nn.tasks import RTDETRDetectionModel

    if getattr(RTDETRDetectionModel, "_sync_patched", False):
        return False

    def loss(self, batch, preds=None):
        """RT-DETR loss with a single-sync gt_groups computation."""
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        # one sync instead of bs syncs; values identical (indices are whole
        # numbers stored as float)
        gt_groups = torch.bincount(batch_idx.view(-1).long(), minlength=bs).tolist()
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        return sum(loss.values()), {
            "giou_loss": loss["loss_giou"].detach(),
            "cls_loss": loss["loss_class"].detach(),
            "l1_loss": loss["loss_bbox"].detach(),
        }

    RTDETRDetectionModel.loss = loss
    RTDETRDetectionModel._sync_patched = True
    return True


def use_bf16(trainer) -> bool:
    """Switch AMP from fp16+GradScaler to bf16 with NO scaler.

    Measured: `self.scaler.step(self.optimizer)` (engine/trainer.py:846) is the
    single largest source of GPU->CPU traffic in the whole training step --
    **384 of 520 .item() calls per iteration (74%)**, far ahead of the
    Hungarian matcher (7 real syncs) or the gt_groups loop (16).

    fp16 needs GradScaler because its 5-bit exponent cannot represent small
    gradients without underflow, so every step scales, unscales, and then
    inspects per-device `found_inf` flags on the CPU -- that inspection is the
    384 calls. bf16 has fp32's 8-bit exponent, so no scaling is needed and the
    scaler disappears entirely.

    Trade-off: bf16 has 7 mantissa bits vs fp16's 10, i.e. lower precision but
    far greater range. This is the standard choice for modern training on
    Ampere/Ada hardware. NOTE this is a genuine numerical change (unlike
    channels_last, which is bit-exact) -- results will differ slightly.

    Requires a bf16-capable GPU (checked). Call after trainer._setup_train().
    """
    import torch as _t
    import ultralytics.engine.trainer as _tr

    if not _t.cuda.is_bf16_supported():
        return False

    # trainer.py does `from ...torch_utils import autocast`, binding the name
    # in ITS module namespace -- so patch it there, not at the source module.
    def _bf16_autocast(enabled, device="cuda"):
        return _t.autocast(device, dtype=_t.bfloat16, enabled=bool(enabled))

    _tr.autocast = _bf16_autocast

    # enabled=False makes scale()/unscale_()/update() no-ops and step() a
    # direct optimizer.step() -- removing the found_inf CPU inspection.
    trainer.scaler = _t.amp.GradScaler("cuda", enabled=False)
    return True


def use_fused_adamw(trainer) -> bool:
    """Rebuild the trainer's AdamW with fused=True (single multi-tensor CUDA
    kernel per step instead of per-parameter kernel launches).

    With 575 parameter tensors, the stock optimizer step contributes a large
    share of the 7,443 launches/iter. Call after trainer._setup_train().
    Returns True if swapped.
    """
    import torch as _t

    opt = trainer.optimizer
    if not isinstance(opt, _t.optim.AdamW):
        return False
    trainer.optimizer = _t.optim.AdamW(opt.param_groups, fused=True)
    return True


def auto_segments(model):
    """Find maximal runs of layers that can be checkpointed as one unit.

    A run [a..b] is safe when:
      * every layer in it takes its input from the previous layer (`f == -1`),
        so no skip connection crosses the boundary, and
      * no layer in [a..b-1] is in `model.save`, so no intermediate output is
        needed later (the run's final output is kept normally).

    For RT-DETR-L this yields the backbone: (0,3), (4,7), (8,12) -- exactly the
    layers holding the 640->320->160->80 feature maps, i.e. where activation
    memory actually is. The neck (13+) has Concats pulling from earlier layers,
    so it is excluded; segmenting it would require passing those external
    inputs into the checkpoint explicitly.
    """
    save, layers = set(model.save), list(model.model)
    segs, start = [], None
    for i, m in enumerate(layers):
        if m.f != -1:
            if start is not None and i - 1 > start:
                segs.append((start, i - 1))
            start = None
            continue
        if start is None:
            start = i
        if i in save:
            if i > start:
                segs.append((start, i))
            start = i + 1
    return [(a, b) for a, b in segs if b > a]  # single-layer runs save nothing


def enable_segment_checkpointing(model, segments="auto"):
    """Checkpoint multi-layer SEGMENTS of RTDETRDetectionModel.predict.

    IMPORTANT -- which method to patch. RTDETRDetectionModel (nn/tasks.py:890)
    defines its OWN `predict` (line 1043) that overrides BaseModel.predict /
    _predict_once, because RT-DETR's decoder head takes `batch` as a second
    argument. Patching BaseModel._predict_once therefore does NOTHING for
    RT-DETR -- an earlier version of this function did exactly that, and every
    "measurement" from it (identical loss, matching gradients, unchanged
    memory) was simply the unmodified model.

    Per-layer checkpointing is also useless here: `checkpoint(layer, x)`
    discards a layer's *internal* activations, but the layer's OUTPUT must
    still be stored for the next layer, and Ultralytics layers are mostly Conv
    (conv+bn+act) whose internals are tiny beside their outputs. Segments fix
    that -- intermediates *between* layers in a run are discarded and
    recomputed in backward.

    `model.ckpt_segment_calls` counts segment executions, so callers can
    assert the patch actually ran rather than inferring it from plausible
    numbers.

    Returns the list of (start, end) segments installed.
    """
    import torch.utils.checkpoint as cp
    from ultralytics.nn.tasks import RTDETRDetectionModel

    segs = auto_segments(model) if segments == "auto" else list(segments)
    model._ckpt_segments = segs
    model.ckpt_segment_calls = 0

    if getattr(RTDETRDetectionModel, "_seg_ckpt_patched", False):
        return segs

    def predict(self, x, profile=False, batch=None, augment=False, embed=None):
        """RT-DETR forward with optional multi-layer segment checkpointing."""
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        layers = list(self.model[:-1])  # head handled separately, as upstream
        seg_at = {a: (a, b) for a, b in (getattr(self, "_ckpt_segments", None) or [])
                  if b < len(layers)}
        use_ckpt = self.training and not profile and embed == {-1}

        i = 0
        while i < len(layers):
            if use_ckpt and i in seg_at:
                a, b = seg_at[i]
                mods = layers[a:b + 1]

                def run_segment(inp, _mods=mods):
                    for mm in _mods:
                        inp = mm(inp)
                    return inp

                x = cp.checkpoint(run_segment, x, use_reentrant=False)
                self.ckpt_segment_calls = getattr(self, "ckpt_segment_calls", 0) + 1
                for _ in range(a, b):  # intermediates provably not in `save`
                    y.append(None)
                y.append(x if layers[b].i in self.save else None)
                i = b + 1
                continue

            m = layers[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in embed:
                embeddings.append(nn_fn.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
            i += 1

        head = self.model[-1]
        return head([y[j] for j in head.f], batch)

    RTDETRDetectionModel.predict = predict
    RTDETRDetectionModel._seg_ckpt_patched = True
    return segs
