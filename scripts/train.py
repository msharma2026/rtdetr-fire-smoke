"""Two-stage RT-DETR training for fire/smoke detection.

Stage 1 pretrains on FASDD (85,783 images) starting from COCO weights.
Stage 2 fine-tunes that checkpoint on D-Fire at a 10x lower LR, with the
backbone frozen.

Three Ultralytics behaviours this deliberately works around:

  1. optimizer="auto" SILENTLY IGNORES lr0. It logs "optimizer='auto' found,
     ignoring 'lr0=...'" and picks its own LR from dataset size. Stage 2's
     entire premise is a lower LR than stage 1, so leaving this on auto would
     restart D-Fire at stage-1 LR and wash out the pretrained features. The
     optimizer is therefore pinned explicitly so lr0 actually takes effect.

  2. deterministic=True is the Ultralytics default. RT-DETR hits ops with no
     deterministic CUDA kernel (grid_sampler_2d_backward, cumsum), so it just
     warns and falls back -- costing speed for determinism it cannot deliver.
     Disabled here in favour of a fixed seed.

  3. Ultralytics derives gradient accumulation as round(nbs/batch) with
     nbs=64. batch=16 therefore accumulates 4 mini-batches per optimizer
     step, holding the effective batch at 64 regardless of the memory-driven
     batch size. Nothing to configure -- but it means changing --batch
     silently changes accumulation, not the effective batch.

Measured on an RTX 4080 (16GB), rtdetr-l, peak reserved VRAM:
    imgsz 640  batch 16 -> 12.4GB      <- stage default
    imgsz 960  batch  8 -> 12.1GB
    imgsz 960  batch 10 -> 15.9GB      <- too tight, <0.5GB headroom
    imgsz 1280 batch  4 -> 10.5GB

Resolution decision (see DESIGN_DECISIONS.md sec 3.4/4 for full detail):
960 was the original choice, reasoned from D-Fire's larger native images
(median 1200x720 vs rtdetr-l.pt's 640 pretraining resolution). Probing found
the opposite: 640 beat 960 and 1280 on FASDD (4 epochs) and on D-Fire at both
4 and 14 epochs, on train loss, mAP50, and generalisation gap, with no sign
of 960/1280 closing the gap given more epochs. 1280's longer probe was
inconclusive on its own terms -- it timed out twice, and the two partial runs
disagreed with each other at matching epochs, evidence of high run-to-run
noise in this probe rather than support for either resolution. 640 is also
what rtdetr-l.pt was actually pretrained at (confirmed via its checkpoint's
train_args), so it requires no resolution-adaptation from the backbone,
AIFI's position embeddings, or token count -- unlike 960/1280, where multiple
resolution-sensitive subsystems are simultaneously out of distribution.

Calibrate before committing to a long run:
    python scripts/train.py --stage 1 --epochs 1 --fraction 0.02
"""

import argparse
import sys
from pathlib import Path

import torch
import torch._inductor.config as _inductor_config

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `patches`

# --- throughput settings, all measured (see DESIGN_DECISIONS.md sec 6) ---
# Training is CPU-dispatch-bound, not GPU-bound: ~7,443 kernel launches and
# ~163 stream syncs per iteration account for over half of each step. Only
# changes that shrink CPU work help; batch size, checkpointing and more
# dataloader workers were all measured to do nothing.

# cuDNN autotunes the fastest conv algorithm per input shape. Ultralytics
# leaves this off (utils/torch_utils.py has it commented out as an AutoBatch
# workaround), but our shapes are fixed (imgsz 640, rect=False) -- exactly the
# case benchmark mode exists for. Measured +1.6% alone.
torch.backends.cudnn.benchmark = True

# TorchInductor's VECTORISED CPU codegen emits invalid C++ for this model
# ("Vectorized<bool> has no member named cast"), which is why compile failed
# outright before. That path is only reached because RT-DETR's loss calls
# .item() on data-dependent GT counts and graph-breaks a fragment to CPU.
# Scalar CPU codegen sidesteps it; GPU kernels come from Triton regardless.
_inductor_config.cpp.simdlen = 0

# TF32 matmuls on tensor cores for the FP32 ops AMP leaves alone. Measured as
# part of a +6.1% stack (58.06 vs 54.7 img/s) with the three patches below.
torch.set_float32_matmul_precision("high")

from ultralytics import RTDETR  # noqa: E402  (after inductor config)
from patches import (  # noqa: E402
    patch_hgblock_for_compile,
    patch_mha_fast_path,
    patch_rtdetr_loss_syncs,
    patch_build_optimizer,
    use_backbone_lr_multiplier,
    use_fused_adamw,
)

# Every MHA call site discards attention weights yet need_weights defaults to
# True, blocking the fused SDPA path. And RT-DETR's loss does one .item() GPU
# sync per image for gt_groups; bincount does it in one.
patch_mha_fast_path()
patch_rtdetr_loss_syncs()

ROOT = Path.home() / "repos/fire_detection"
DATA = ROOT / "data"
RUNS = ROOT / "runs"

# Graceful pause. `touch PAUSE` in the repo root and training exits cleanly at
# the next epoch boundary, then `--resume` picks up exactly where it stopped.
#
# Why not SIGSTOP: it freezes the process but the CUDA context keeps its ~12 GB
# of VRAM reserved, so the GPU is still unusable for anything else. Freeing the
# card requires the process to actually exit.
#
# The callback runs on on_fit_epoch_end, which fires AFTER save_model()
# (trainer.py:610) and BEFORE the `if self.stop: break` check (:633), so the
# checkpoint on disk is always complete.
PAUSE_FILE = ROOT / "PAUSE"

# lr0=3e-4 (stage 1), 3e-5 (stage 2) -- empirically probed, not the DETR-
# convention 1e-4 this started as. A 5-point sweep (3e-5/1e-4/3e-4/1e-3/3e-3)
# on FASDD found 3e-4 winning on train loss, mAP50, AND generalisation gap
# simultaneously, with 1e-3 already worse and 3e-3 clearly unstable -- a real
# local optimum, not a point on a still-climbing curve. Stage 2 keeps the same
# 10x-lower ratio to stage 1 (now 3e-5, was 1e-5) for the same reason as
# before: a smaller safety margin against disturbing FASDD's features while
# fine-tuning on the much smaller D-Fire set. Stage 2's specific value was NOT
# independently probed -- this is a proportional carry-over of stage 1's
# tested result, not a separately-measured optimum for D-Fire fine-tuning.
#
# freeze=10 pins model.0-model.9 (the HGNetv2 CNN backbone, ~13.5M params /
# 41% of the net). The neck (AIFI encoder + RepC3 fusion) and RTDETRDecoder
# stay trainable. Rationale: FASDD and D-Fire are the same task, so low-level
# fire/smoke features should transfer. Freezing the backbone was the original
# guess -- MEASURED WRONG: freeze=10 loses 0.028 mAP50 to unfreezing at a 0.3x
# backbone LR, under both COCO and FASDD initialisation. A low LR does not
# merely "soften" the guarantee, it outperforms it.
STAGES = {
    1: {
        "data": "fasdd.yaml", "epochs": 80, "lr0": 3e-4,
        "imgsz": 640, "batch": 16, "freeze": None, "name": "stage1_fasdd",
        "bblr": 0.3,
    },
    2: {
        # Measured recipe (3 seeds, D-Fire test mAP50 0.8352 +/- 0.0008):
        #   lr0=1e-4  -- 3e-5 was inherited from stage-1 tuning and costs a
        #                COCO-initialised arm 0.078 mAP50. Both arms were
        #                re-tuned with a ladder extended until the winner was
        #                interior, so neither is a grid edge.
        #   epochs=20 -- LR anneals over the TOTAL run length, so a longer
        #                run's mid-schedule checkpoint never completes its
        #                anneal. 60 epochs scores 0.0137 BELOW 20 (~17 sigma).
        #   freeze=0  -- see the note above STAGES.
        "data": "dfire.yaml", "epochs": 20, "lr0": 1e-4,
        "imgsz": 640, "batch": 16, "freeze": 0, "name": "stage2_dfire",
        "bblr": 0.3,
    },
}


def resolve_weights(stage: int, override: str | None) -> str:
    if override:
        return override
    if stage == 1:
        return "rtdetr-l.pt"  # COCO-pretrained, auto-downloaded
    best = RUNS / STAGES[1]["name"] / "weights/best.pt"
    if not best.exists():
        raise SystemExit(
            f"stage 2 needs stage 1's weights but {best} does not exist.\n"
            f"Run stage 1 first, or pass --weights explicitly."
        )
    return str(best)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--weights", default=None, help="override starting checkpoint")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="see VRAM table in module docstring")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--freeze", type=int, default=None, help="freeze model.0..model.N-1; 0 disables")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--bblr", type=float, default=None,
                   help="backbone LR multiplier; 0 disables. Probed optimum 0.3")
    p.add_argument("--device", default="0")
    p.add_argument("--fraction", type=float, default=1.0, help="<1 to calibrate on a slice")
    p.add_argument("--patience", type=int, default=15, help="early-stop; cheap insurance on long runs")
    p.add_argument("--save-period", type=int, default=-1,
                   help="checkpoint every N epochs; -1 = last/best only")
    # Needed for the annealed, mosaic-free polish phase: stage 1 plateaued at
    # ~1.1e-4 for 16 epochs, and the previous 30-epoch run showed its whole
    # tail gain (+0.031) came from annealing that LR down, not from more steps
    # at it. These let a short run replicate that tail directly.
    p.add_argument("--lrf", type=float, default=None,
                   help="final LR as a fraction of lr0 (Ultralytics default 0.01)")
    p.add_argument("--mosaic", type=float, default=None,
                   help="mosaic probability; 0 disables (probed as ~neutral)")
    p.add_argument("--warmup", type=float, default=None,
                   help="warmup epochs; 0 when continuing from trained weights")
    # [added by Claude 2026-08-19] seed was hardcoded to 0, so repeat runs
    # shared head init and data order and could differ only by GPU
    # nondeterminism. Needed to measure run-to-run variance.
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed; vary it to measure run-to-run variance")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--no-compile", action="store_true",
                   help="disable torch.compile (escape hatch; costs ~7%% throughput)")
    args = p.parse_args()

    # torch.compile fuses many small kernels into fewer larger ones, directly
    # attacking the measured launch-overhead bottleneck: +8.2% on top of
    # channels_last+cudnn (54.47 vs 50.36 img/s), and it lowers peak VRAM
    # 13.5 -> 11.2 GB. Costs ~400s of one-time compilation, plus one extra
    # graph for the smaller final batch of each epoch (compiled once, cached).
    # Requires the HGBlock patch: dynamo mistraces its self-referential
    # generator (verified bit-identical in eager mode).
    # NOTE mode="reduce-overhead" is deliberately NOT used -- CUDA graph
    # capture is skipped on every attempt ("cpu device") because of the loss's
    # CPU fragments, leaving all of the overhead and none of the benefit:
    # measured 3.92 img/s, a 13x slowdown.
    use_compile = not args.no_compile
    if use_compile:
        patch_hgblock_for_compile()

    cfg = STAGES[args.stage]
    run_name = args.name or cfg["name"]

    # --resume must pass an explicit path. Ultralytics treats resume=True as
    # "find the newest **/last*.pt anywhere under cwd" (utils.files.get_latest_run),
    # which with probe runs sitting in runs/ would happily resume the wrong job.
    resume = False
    if args.resume:
        resume = RUNS / run_name / "weights/last.pt"
        if not resume.exists():
            raise SystemExit(f"--resume given but {resume} does not exist")
        resume = str(resume)

    weights = resume or resolve_weights(args.stage, args.weights)

    # argparse default None means "inherit the stage default"; an explicit
    # --freeze 0 must still disable freezing, hence the `is not None` checks.
    freeze = args.freeze if args.freeze is not None else cfg["freeze"]
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]
    batch = args.batch if args.batch is not None else cfg["batch"]

    model = RTDETR(weights)

    # model.train() builds its trainer internally and runs atomically, so
    # trainer-level patches can only be applied via callbacks. This one fires
    # at the end of _setup_train, after the optimizer exists. (An earlier
    # version shipped use_fused_adamw() with no way to ever call it.)
    # The optimizer must have its final param-group layout BEFORE
    # resume_training() loads saved state (trainer.py:413), which is two lines
    # earlier than on_pretrain_routine_end (:415). Patching build_optimizer
    # (:300) is the only hook early enough. Doing this from the callback is
    # what broke --resume 33 epochs into the 80-epoch run.
    bblr = args.bblr if args.bblr is not None else cfg.get("bblr")
    patch_build_optimizer(mult=bblr)

    def _apply_trainer_patches(trainer):
        from ultralytics.engine.trainer import BaseTrainer
        st = getattr(BaseTrainer, "_bopt_stats", None)
        assert st, "build_optimizer patch never ran"
        assert st["fused"], "optimizer is not fused"
        if bblr:
            assert st["backbone_params"] > 0, "no backbone params were split"
            lrs = sorted({g["lr"] for g in trainer.optimizer.param_groups})
            assert len(lrs) > 1, f"expected >1 distinct LR, got {lrs}"
        print(f"trainer patches: {st}")

    def _check_pause(trainer):
        if PAUSE_FILE.exists():
            trainer.stop = True
            ep = trainer.epoch + 1
            print("", flush=True)
            print(f"PAUSE file found -- stopping cleanly after epoch {ep}."
                  f" Checkpoint saved.", flush=True)
            print(f"  resume with: rm {PAUSE_FILE} && python scripts/train.py"
                  f" --stage {args.stage} --resume", flush=True)

    model.add_callback("on_pretrain_routine_end", _apply_trainer_patches)
    model.add_callback("on_fit_epoch_end", _check_pause)

    print(f"stage {args.stage}: {weights} -> {cfg['data']} "
          f"(imgsz={imgsz} batch={batch} freeze={freeze} resume={bool(resume)} "
          f"compile={use_compile})")

    model.train(
        data=str(DATA / cfg["data"]),
        epochs=args.epochs or cfg["epochs"],
        batch=batch,
        imgsz=imgsz,
        freeze=freeze or 0,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        patience=args.patience,
        save_period=args.save_period,
        resume=resume,
        project=str(RUNS),
        name=run_name,
        exist_ok=True,
        # see module docstring -- both of these are load-bearing
        optimizer="AdamW",
        lr0=args.lr0 or cfg["lr0"],
        **({"lrf": args.lrf} if args.lrf is not None else {}),
        **({"mosaic": args.mosaic} if args.mosaic is not None else {}),
        **({"warmup_epochs": args.warmup} if args.warmup is not None else {}),
        deterministic=False,
        seed=args.seed,
        # NHWC memory layout: what cuDNN's Tensor-Core conv kernels want
        # natively. NCHW forces a transpose around every conv, inflating both
        # kernel count and launch overhead -- and launch overhead is the
        # measured bottleneck (7,443 launches/iter = 26% of step time).
        # Lossless on CUDA per Ultralytics (the "numerically wrong" caveat in
        # their source refers to MPS). Measured +9.0% alone.
        channels_last=True,
        compile="default" if use_compile else False,
        # 15GB system RAM total, so caching 86k images is not an option
        cache=False,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
