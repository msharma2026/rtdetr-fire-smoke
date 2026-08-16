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
    """Rebuild the trainer's AdamW with fused=True (one multi-tensor CUDA
    kernel per step instead of ~575 per-parameter launches).

    Two things the rebuild must preserve, both learned by breaking them:

    1. `initial_lr` and `param_group`. `_setup_scheduler()` (trainer.py:308)
       stamps `initial_lr` onto every param group and runs BEFORE the
       on_pretrain_routine_end callback (line 415), so a naive rebuild drops
       it and the warmup loop dies with `KeyError: 'initial_lr'`
       (trainer.py:478). `param_group` marks the bias group, which warmup uses
       to ramp bias LR from warmup_bias_lr instead of 0.
    2. Per-group `fused`/`foreach` keys must NOT be copied: they override the
       constructor's fused=True, silently yielding a non-fused optimizer that
       then trips GradScaler's `assert grad_scale is None and found_inf is
       None`.

    The LR scheduler also holds a reference to the old optimizer, so it is
    rebuilt against the new one.

    Call from an `on_pretrain_routine_end` callback. Returns True if swapped.
    """
    import torch as _t

    opt = trainer.optimizer
    if not isinstance(opt, _t.optim.AdamW):
        return False

    # copy hyperparameters and trainer bookkeeping, but not fused/foreach
    keep = ("lr", "initial_lr", "weight_decay", "betas", "eps", "param_group",
            "momentum", "maximize", "amsgrad")
    groups = []
    for g in opt.param_groups:
        ng = {"params": g["params"]}
        ng.update({k: g[k] for k in keep if k in g})
        ng.setdefault("initial_lr", g["lr"])   # belt and braces
        groups.append(ng)

    trainer.optimizer = _t.optim.AdamW(groups, fused=True)

    # the old scheduler still points at the discarded optimizer
    if getattr(trainer, "scheduler", None) is not None and hasattr(trainer, "lf"):
        trainer.scheduler = _t.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lr_lambda=trainer.lf)

    # fail loudly here rather than 200 iterations into warmup
    missing = [i for i, g in enumerate(trainer.optimizer.param_groups)
               if "initial_lr" not in g]
    assert not missing, f"param groups missing initial_lr: {missing}"
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


def patch_mha_fast_path() -> bool:
    """Default nn.MultiheadAttention.forward to need_weights=False.

    Every MHA call site in RT-DETR (AIFI at transformer.py:115/142, the
    decoder self-attention at :698) discards the attention weights with [0],
    yet need_weights defaults to True -- which disables the fused SDPA fast
    path and forces the unfused math kernel that materialises the full
    attention matrix. setdefault preserves any caller that explicitly asks
    for weights; Ultralytics never does.
    """
    import torch.nn as nn

    if getattr(nn.MultiheadAttention, "_fastpath_patched", False):
        return False
    orig = nn.MultiheadAttention.forward

    def forward(self, *args, **kwargs):
        kwargs.setdefault("need_weights", False)
        return orig(self, *args, **kwargs)

    nn.MultiheadAttention.forward = forward
    nn.MultiheadAttention._fastpath_patched = True
    return True


# Compiled backbone+neck functions, keyed by live model object. A WeakKey
# dict (rather than a model attribute) so that Ultralytics' checkpoint
# deepcopy and the EMA model never see -- and never try to pickle -- the
# compiled callable; models absent from this dict fall back to the original
# eager predict.
import weakref

_COMPILED_BN = weakref.WeakKeyDictionary()


def compile_forward_only(trainer, mode="reduce-overhead") -> bool:
    """Compile ONLY the backbone+neck (model.0..27); head and loss stay eager.

    Ultralytics' own attempt_compile wraps the whole DetectionModel, and in
    training model(batch) routes through .loss(), whose .item() calls on
    data-dependent GT counts graph-break the region -- which is why
    compile='reduce-overhead' logged 37x 'skipping cudagraphs due to cpu
    device' and ran 13x slower. The backbone+neck, by contrast, is a pure
    static-shape GPU chain (640x640 in, three feature maps out; Ultralytics
    already sets drop_last=True when compiling), i.e. exactly what CUDA
    graphs want.

    The decoder head runs eagerly because its denoising-group generation
    (.cpu()/.item()) cannot be captured; same for the loss and matcher.

    Use INSTEAD of Ultralytics' compile arg (pass compile=False), from an
    on_pretrain_routine_end callback. EMA is constructed before that callback
    fires, so the EMA copy predates and never carries the compiled function.
    """
    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel

    model = trainer.model
    layers = list(model.model[:-1])
    head = model.model[-1]
    save = set(model.save)
    hf = list(head.f)

    def run_backbone_neck(x):
        y = []
        for m in layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in save else None)
        return tuple(y[j] for j in hf)

    _COMPILED_BN[model] = (torch.compile(run_backbone_neck, mode=mode, dynamic=False), head)

    if getattr(RTDETRDetectionModel, "_fwd_split_patched", False):
        return True
    orig_predict = RTDETRDetectionModel.predict

    def predict(self, x, profile=False, batch=None, augment=False, embed=None):
        entry = _COMPILED_BN.get(self)
        if entry is None or profile or embed:
            return orig_predict(self, x, profile=profile, batch=batch,
                                augment=augment, embed=embed)
        fn, hd = entry
        return hd(list(fn(x)), batch)

    RTDETRDetectionModel.predict = predict
    RTDETRDetectionModel._fwd_split_patched = True
    return True


def use_backbone_lr_multiplier(trainer, mult: float = 0.1, n_backbone: int = 10) -> int:
    """Give the backbone (model.0..model.{n-1}) a lower LR than the rest.

    RT-DETR's paper (Table A) uses base LR 1e-4 with **backbone LR 1e-5** --
    a 10x reduction, standard DETR/DINO practice: the ImageNet-pretrained
    backbone is nudged while the randomly-initialised decoder learns fast.
    Ultralytics applies a single LR to every parameter, so the pretrained
    backbone is trained as hard as the decoder.

    This may also explain why our LR sweep favoured 3e-4: with one LR you are
    forced into a compromise between 'too slow for the decoder' and 'too fast
    for the backbone'.

    Rebuilds param groups, preserving each group's weight_decay (Ultralytics
    splits into decay/no-decay/bias groups). Returns the number of backbone
    parameters moved to the reduced LR. Call from on_pretrain_routine_end.
    """
    import torch as _t

    opt = trainer.optimizer
    core = trainer.model
    core = getattr(core, "_orig_mod", core)   # unwrap torch.compile
    names = {id(p): n for n, p in core.named_parameters()}
    prefixes = tuple(f"model.{i}." for i in range(n_backbone))

    def is_backbone(p):
        n = names.get(id(p), "")
        n = n.replace("_orig_mod.", "")
        return n.startswith(prefixes)

    groups, moved = [], 0
    keep = ("weight_decay", "betas", "eps")
    for g in opt.param_groups:
        base_lr = g["lr"]
        hp = {k: g[k] for k in keep if k in g}
        bb = [p for p in g["params"] if is_backbone(p)]
        rest = [p for p in g["params"] if not is_backbone(p)]
        moved += len(bb)
        if bb:
            groups.append({"params": bb, "lr": base_lr * mult,
                           "initial_lr": base_lr * mult, **hp})
        if rest:
            groups.append({"params": rest, "lr": base_lr,
                           "initial_lr": base_lr, **hp})

    fused = dict(fused=True) if _t.cuda.is_available() else {}
    trainer.optimizer = _t.optim.AdamW(groups, **fused)
    # the LR scheduler multiplies each group's lr by a lambda; rebuild it so
    # it tracks the new group list rather than the discarded optimizer
    if getattr(trainer, "scheduler", None) is not None and hasattr(trainer, "lf"):
        trainer.scheduler = _t.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lr_lambda=trainer.lf)
    return moved


def set_grad_clip(max_norm: float = 0.1) -> bool:
    """Set the gradient-clipping threshold used in BaseTrainer.optimizer_step.

    Ultralytics hardcodes max_norm=10.0 (engine/trainer.py:843) -- a
    YOLO-inherited default so loose it essentially never engages. RT-DETR's
    paper specifies **clip gradient norm 0.1**, 100x tighter; DETR-family
    training is known to need tight clipping for stability.
    """
    import torch as _t
    from ultralytics.engine.trainer import BaseTrainer

    BaseTrainer._clip_max_norm = max_norm
    if getattr(BaseTrainer, "_clip_patched", False):
        return True

    def optimizer_step(self):
        """optimizer_step with a configurable clip threshold."""
        self.scaler.unscale_(self.optimizer)
        _t.nn.utils.clip_grad_norm_(self.model.parameters(),
                                    max_norm=getattr(self, "_clip_max_norm", 10.0))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    BaseTrainer.optimizer_step = optimizer_step
    BaseTrainer._clip_patched = True
    return True
