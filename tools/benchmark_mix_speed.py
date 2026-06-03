#!/usr/bin/env python3
"""Benchmark mixup augmentation speed.

Loads a training config, builds the dataset / dataloader and model
backbone, then times the configured mixing operations – including any
pre-mix backbone / gradient computations – without running the main
training forward pass (no loss, no weight update).

What is timed for each mode category
-------------------------------------
static      : mix function only (cutmix, mixup, fmix, …)
attentivemix: backbone_k forward (feature extraction) + mix function
puzzlemix   : backbone+head eval forward → loss.backward() (saliency
              gradients) + mix function
snapmix     : backbone forward (get CAM features) + mix function
guidedmix   : backbone+head eval forward → loss.backward() + mix
tokenmix /
  mixpro /
  smmix     : backbone forward (attention maps) + mix function
alignmix    : backbone forward (feature maps) + mix function in
              feature space
manifoldmix : index shuffle + mask generation only (no backbone;
              the actual mixing happens inside a specialised backbone
              forward which is out of scope here)
samix       : backbone_k forward (feature maps) + two mix_block
              forwards inside pixel_mixup (AutoMixup model)
transmix    : SKIPPED – its label-correction step runs after the main
              backbone forward and adds negligible overhead there.

Usage
-----
    python tools/benchmark_mix_speed.py CONFIG [CONFIG ...] [options]

Examples
--------
    python tools/benchmark_mix_speed.py \\
        configs/classification/cub200/r18/r18_mixups.py \\
        --num-iters 50 --warmup 5

    python tools/benchmark_mix_speed.py \\
        configs/classification/cub200/deit_s/cutmix_fga/\\
        deit_s_cutmix_fga_abs_l1p0.py --batch-size 16 --device cuda

    python tools/benchmark_mix_speed.py \\
        configs/classification/cub200/r18/r18_mixups.py \\
        configs/classification/cub200/deit_s/cutmix_fga/deit_s_cutmix_fga_abs_l1p0.py \\
        --num-iters 50
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from torch.autograd import Variable
from torch.utils.data import DataLoader

from openmixup.datasets import build_dataset
from openmixup.models import build_model
from openmixup.models.augments import (
    alignmix,
    attentivemix,
    attentivemix_fga,
    augmix,
    cutmix,
    cutmix_foreground_area,
    cutmix_stupid,
    fmix,
    gridmix,
    guidedmix,
    mask_attentivemix,
    mask_mixup,
    mixpro,
    mixup,
    puzzlemix,
    resizemix,
    saliencymix,
    smmix,
    smoothmix,
    snapmix,
    tokenmix,
)

# Default mix_args matching MixUpClassification defaults.
_DEFAULT_MIX_ARGS = dict(
    augmix=dict(mixture_depth=-1, mixture_width=3, severity=1),
    alignmix=dict(eps=0.1, max_iter=100),
    attentivemix=dict(grid_size=32, top_k=6, beta=8),
    attentivemix_fga=dict(
        grid_size=32, top_k=6, beta=8, fg_weight=1, mode="absolute"
    ),
    cutmix=dict(),
    cutmix_foreground_area=dict(fg_weight=1.0),
    cutmix_stupid=dict(stupidity=1.0),
    fmix=dict(
        decay_power=3, size=(224, 224), max_soft=0.0, reformulate=False
    ),
    gridmix=dict(
        n_holes=(2, 6),
        hole_aspect_ratio=1.0,
        cut_area_ratio=(0.5, 1),
        cut_aspect_ratio=(0.5, 2),
    ),
    guidedmix=dict(
        guided_type="ap",
        condition="greedy",
        distance_metric="l2",
        size=(7, 7),
        sigma=(3, 3),
    ),
    manifoldmix=dict(layer=(0, 3)),
    mask_attentivemix=dict(
        grid_size=32, top_k=6, fg_weight=1.0, mode="absolute"
    ),
    mask_mixup=dict(),
    mixpro=dict(
        mask_mix=True,
        mask_patch_size=64,
        model_patch_size=16,
        num_class=None,
        smoothing=0.1,
    ),
    mixup=dict(),
    puzzlemix=dict(
        transport=True,
        t_batch_size=32,
        block_num=4,
        beta=1.2,
        gamma=0.5,
        eta=0.2,
        neigh_size=4,
        n_labels=3,
        t_eps=0.8,
        t_size=-1,
    ),
    resizemix=dict(scope=(0.1, 0.8), use_alpha=False),
    saliencymix=dict(),
    smmix=dict(side=14, min_side_ratio=0.25, max_side_ratio=0.75),
    smoothmix=dict(),
    snapmix=dict(),
    tokenmix=dict(
        mask_type="block",
        minimum_tokens=14,
        num_classes=None,
        smoothing=0.1,
    ),
)

SKIP_MODES = {"transmix", "automix", "tla", "vanilla"}


# -------------------------------------------------------------------- #
# Helpers                                                               #
# -------------------------------------------------------------------- #

def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def merge_mix_args(mode, alpha, cfg_mix_args):
    """Return merged mix_args dict with alpha and dist_mode pre-filled."""
    args = dict(_DEFAULT_MIX_ARGS.get(mode, {}))
    args.update(cfg_mix_args.get(mode, {}))
    args["alpha"] = alpha
    args["dist_mode"] = False
    return args


def simple_collate(batch):
    imgs = torch.stack([b["img"] for b in batch])
    labels = torch.tensor(
        [b["gt_label"] for b in batch], dtype=torch.long
    )
    masks = None
    if "mask" in batch[0] and batch[0]["mask"] is not None:
        masks = torch.stack([b["mask"] for b in batch])
    return {"img": imgs, "gt_label": labels, "mask": masks}


def prepare_dataset_cfg(cfg):
    """Return dataset config with prefetch disabled and tensors ensured."""
    dataset_cfg = cfg.data.train.copy()
    dataset_cfg["prefetch"] = False
    pipeline = list(dataset_cfg.get("pipeline", []))
    if not any(p.get("type") == "ToTensor" for p in pipeline):
        img_norm_cfg = getattr(
            cfg,
            "img_norm_cfg",
            dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        pipeline = pipeline + [
            dict(type="ToTensor"),
            dict(type="Normalize", **img_norm_cfg),
        ]
        dataset_cfg["pipeline"] = pipeline
    return dataset_cfg


# -------------------------------------------------------------------- #
# Per-category benchmark functions                                      #
# -------------------------------------------------------------------- #

def _timed(fn, warmup, n_iters, device):
    """Run fn() warmup+n_iters times; return array of timed ms values."""
    times = []
    for i in range(warmup + n_iters):
        cuda_sync(device)
        t0 = time.perf_counter()
        fn()
        cuda_sync(device)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)
    return np.array(times)


def bench_static(fn, img, gt_label, mix_args, mask, warmup, n_iters, device):
    """Static input-space modes (no backbone required)."""
    def _iter():
        kwargs = dict(mix_args)
        if mask is not None:
            kwargs["mask"] = mask
        with torch.no_grad():
            fn(img, gt_label, **kwargs)

    return _timed(_iter, warmup, n_iters, device)


def bench_attentivemix(
    mode, model, img, gt_label, mask, mix_args, warmup, n_iters, device
):
    """backbone_k forward (feature extraction) + attentivemix."""
    fn = attentivemix if mode == "attentivemix" else attentivemix_fga
    feat_size = mix_args.get("feat_size", 224)

    def _iter():
        img_sc = F.interpolate(
            img,
            scale_factor=feat_size / img.size(2),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            features = model.backbone_k(img_sc)[0]
        kwargs = dict(mix_args, features=features)
        if mode == "attentivemix_fga":
            kwargs["obj_mask"] = mask
        with torch.no_grad():
            fn(img, gt_label, **kwargs)

    return _timed(_iter, warmup, n_iters, device)


def bench_gradient_based(
    mode, model, img, gt_label, mix_args, warmup, n_iters, device
):
    """backbone+head eval forward → loss.backward() (saliency) + mix."""
    fn = puzzlemix if mode == "puzzlemix" else guidedmix

    def _iter():
        input_var = Variable(img.clone().detach(), requires_grad=True)
        model.backbone.eval()
        model.head.eval()
        pred = model.head(model.backbone(input_var))
        loss = model.head.loss(pred, gt_label)["loss"]
        loss.backward()
        features = torch.sqrt(
            torch.mean(input_var.grad ** 2, dim=1)
        )
        model.backbone.zero_grad()
        model.head.zero_grad()
        model.backbone.train()
        model.head.train()
        with torch.no_grad():
            fn(img, gt_label, **dict(mix_args, features=features))

    return _timed(_iter, warmup, n_iters, device)


def bench_snapmix(
    model, img, gt_label, mix_args, warmup, n_iters, device
):
    """backbone forward (CAM features) + snapmix."""
    def _iter():
        with torch.no_grad():
            last_feat = model.backbone(img)[-1]
            if isinstance(last_feat, (list, tuple)):
                last_feat = last_feat[0]
        weight = model.head.fc.weight.data  # (num_classes, C)
        if gt_label.dim() == 1:
            class_weights = weight[gt_label]
        else:
            class_weights = gt_label.float() @ weight
        cam = F.relu(torch.einsum('nc,nchw->nhw', class_weights, last_feat))
        cam = F.interpolate(
            cam.unsqueeze(1), size=img.shape[2:],
            mode='bilinear', align_corners=False).squeeze(1)
        features = (cam / (cam.sum(dim=(1, 2), keepdim=True) + 1e-8)).detach()
        with torch.no_grad():
            snapmix(img, gt_label, **dict(mix_args, features=features))

    return _timed(_iter, warmup, n_iters, device)


def bench_vit_attn(
    mode, model, img, gt_label, mix_args, warmup, n_iters, device
):
    """backbone forward (attention maps) + tokenmix / mixpro / smmix."""
    fn_map = {"tokenmix": tokenmix, "mixpro": mixpro, "smmix": smmix}
    fn = fn_map[mode]
    backbone = (
        model.backbone_k
        if model.backbone_k is not None
        else model.backbone
    )

    def _iter():
        with torch.no_grad():
            output = backbone(img)
        attn = output[-1][-1]
        kwargs = dict(mix_args, return_mask=False)
        with torch.no_grad():
            fn(img, gt_label, attn, **kwargs)

    return _timed(_iter, warmup, n_iters, device)


def bench_alignmix(
    model, img, gt_label, mix_args, warmup, n_iters, device
):
    """backbone forward (feature maps) + alignmix in feature space."""
    def _iter():
        with torch.no_grad():
            x = model.backbone(img)[-1]
        # alignmix operates on features; it is not @no_grad
        alignmix(x, gt_label, **mix_args)

    return _timed(_iter, warmup, n_iters, device)


def bench_manifoldmix(
    model, img, gt_label, mix_args, warmup, n_iters, device
):
    """Index shuffle + mask generation only (no backbone needed)."""
    alpha = mix_args["alpha"]

    def _iter():
        with torch.no_grad():
            model._manifoldmix(img, gt_label, alpha)

    return _timed(_iter, warmup, n_iters, device)


def bench_samix(model, img, gt_label, warmup, n_iters, device):
    """SAMix (AutoMixup): mixing overhead only (no backbone_q training forward).

    Times the operations unique to SAMix per iteration:
      1. backbone_k forward on original images → feature extraction
      2. pixel_mixup → two mix_block forwards (produce img_mix_mb / img_mix_bb)
      3. forward_k → backbone_k forward on img_mix_mb + head_mix_k
         (trains the mix_block; has no equivalent in other methods)

    forward_q (backbone_q training on the mixed image) is intentionally
    excluded: it is equivalent to the training step every mixing method
    performs after mixing and is therefore baseline cost, not SAMix overhead.
    """
    alpha = model.alpha
    bs = img.size(0)

    def _iter():
        lam = np.random.beta(alpha, alpha, 2)
        with torch.no_grad():
            index_mb = torch.randperm(bs, device=device)
            index_bb = torch.randperm(bs, device=device)
            feature = model.backbone_k(img)[0]
            results = model.pixel_mixup(
                img, gt_label, lam, [index_mb, index_bb], feature
            )
            model.forward_k(
                results["img_mix_mb"], gt_label, index_mb, lam[0]
            )

    return _timed(_iter, warmup, n_iters, device)


# -------------------------------------------------------------------- #
# Dispatch                                                              #
# -------------------------------------------------------------------- #

STATIC_MODES = {
    "mixup", "cutmix", "fmix", "gridmix", "saliencymix", "smoothmix",
    "resizemix", "augmix", "cutmix_stupid", "mask_mixup",
    "cutmix_foreground_area", "mask_attentivemix",
}
_STATIC_FNS = {
    "mixup": mixup,
    "cutmix": cutmix,
    "fmix": fmix,
    "gridmix": gridmix,
    "saliencymix": saliencymix,
    "smoothmix": smoothmix,
    "resizemix": resizemix,
    "augmix": augmix,
    "cutmix_stupid": cutmix_stupid,
    "mask_mixup": mask_mixup,
    "cutmix_foreground_area": cutmix_foreground_area,
    "mask_attentivemix": mask_attentivemix,
}
_MASK_REQUIRED = {"mask_mixup", "cutmix_foreground_area", "mask_attentivemix"}


def dispatch(
    mode, model, img, gt_label, mask, mix_args, warmup, n_iters, device
):
    """Route a mode to its benchmark function; return ms array or None."""
    if mode in STATIC_MODES:
        needs_mask = mode in _MASK_REQUIRED
        if needs_mask and mask is None:
            print(f"    SKIPPED: {mode} requires mask but dataset has none")
            return None
        return bench_static(
            _STATIC_FNS[mode], img, gt_label, mix_args,
            mask=(mask if needs_mask else None),
            warmup=warmup, n_iters=n_iters, device=device,
        )

    if mode in ("attentivemix", "attentivemix_fga"):
        if model.backbone_k is None:
            print(f"    SKIPPED: {mode} requires backbone_k, not in config")
            return None
        return bench_attentivemix(
            mode, model, img, gt_label, mask, mix_args,
            warmup, n_iters, device,
        )

    if mode in ("puzzlemix", "guidedmix"):
        return bench_gradient_based(
            mode, model, img, gt_label, mix_args,
            warmup, n_iters, device,
        )

    if mode == "snapmix":
        if not hasattr(model.head, "fc"):
            print(
                "    SKIPPED: snapmix requires head.fc (linear head), "
                "not found"
            )
            return None
        return bench_snapmix(
            model, img, gt_label, mix_args, warmup, n_iters, device
        )

    if mode in ("tokenmix", "mixpro", "smmix"):
        return bench_vit_attn(
            mode, model, img, gt_label, mix_args, warmup, n_iters, device
        )

    if mode == "alignmix":
        return bench_alignmix(
            model, img, gt_label, mix_args, warmup, n_iters, device
        )

    if mode == "manifoldmix":
        return bench_manifoldmix(
            model, img, gt_label, mix_args, warmup, n_iters, device
        )

    print(f"    SKIPPED: {mode} – not supported by this benchmark")
    return None


# -------------------------------------------------------------------- #
# Main                                                                  #
# -------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark mixup augmentation speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config", nargs="+", help="train config file path(s)"
    )
    parser.add_argument(
        "--num-iters", type=int, default=100,
        help="number of timed iterations per mode",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="warmup iterations (not timed)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="override imgs_per_gpu from config",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="override workers_per_gpu from config",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def benchmark_config(args, config_path, device):
    """Run the full benchmark for a single config file.

    Returns a dict mapping mode -> result dict, and the batch size used.
    """
    cfg = Config.fromfile(config_path)

    # ---------------------------------------------------------------- #
    # Extract mixing config                                             #
    # ---------------------------------------------------------------- #
    model_cfg = cfg.model
    model_type = model_cfg.get("type", "")
    is_samix = model_type == "AutoMixup"

    if not is_samix:
        mix_modes = model_cfg.get("mix_mode", "mixup")
        mix_modes = (
            mix_modes if isinstance(mix_modes, list) else [str(mix_modes)]
        )
        alphas = model_cfg.get("alpha", 1.0)
        alphas = alphas if isinstance(alphas, list) else [float(alphas)]
        if len(alphas) < len(mix_modes):
            alphas = alphas + [alphas[-1]] * (len(mix_modes) - len(alphas))
        cfg_mix_args = model_cfg.get("mix_args", {})

    # ---------------------------------------------------------------- #
    # Build model (no main pretrained, pretrained_k as in training)    #
    # ---------------------------------------------------------------- #
    print(f"\nBuilding model (random weights)...")
    _mcfg = model_cfg.copy()
    _mcfg["pretrained"] = None
    model = build_model(_mcfg)
    model.to(device)
    model.train()  # default; benchmark fns switch as needed per mode

    # ---------------------------------------------------------------- #
    # Build dataset + dataloader                                        #
    # ---------------------------------------------------------------- #
    print(f"Building dataset from: {config_path}")
    dataset_cfg = prepare_dataset_cfg(cfg)
    dataset = build_dataset(dataset_cfg)
    print(f"  Dataset size : {len(dataset)}")

    bs = args.batch_size or cfg.data.imgs_per_gpu
    nw = (
        args.num_workers
        if args.num_workers is not None
        else cfg.data.workers_per_gpu
    )

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=simple_collate,
    )
    print(f"  Batch size   : {bs}  |  Workers: {nw}  |  Device: {device}")

    # ---------------------------------------------------------------- #
    # Load one representative batch                                     #
    # ---------------------------------------------------------------- #
    print("\nLoading benchmark batch...")
    t0 = time.perf_counter()
    batch = next(iter(loader))
    data_load_ms = (time.perf_counter() - t0) * 1000

    img = batch["img"].to(device, non_blocking=True)
    gt_label = batch["gt_label"].to(device, non_blocking=True)
    mask = batch["mask"]
    if mask is not None:
        mask = mask.to(device, non_blocking=True)
    cuda_sync(device)

    print(f"  First batch  : {data_load_ms:.1f} ms  |  "
          f"img {tuple(img.shape)}  |  "
          f"mask {tuple(mask.shape) if mask is not None else 'None'}")

    # ---------------------------------------------------------------- #
    # Run benchmarks                                                    #
    # ---------------------------------------------------------------- #
    results = {}

    if is_samix:
        print("\n--- samix (AutoMixup) ---")
        times = bench_samix(
            model, img, gt_label,
            warmup=args.warmup, n_iters=args.num_iters, device=device,
        )
        tp = bs / (times.mean() / 1000)
        print(
            f"  {times.mean():.2f} ± {times.std():.2f} ms  "
            f"(p50={np.percentile(times,50):.2f}  "
            f"p95={np.percentile(times,95):.2f})  "
            f"→ {tp:.0f} img/s"
        )
        results["samix"] = dict(
            mean_ms=times.mean(), std_ms=times.std(),
            p50_ms=np.percentile(times, 50),
            p95_ms=np.percentile(times, 95),
            throughput=tp,
        )
    else:
        skipped = [m for m in mix_modes if m in SKIP_MODES]
        if skipped:
            print(f"\n  Skipping (not benchmarkable): {skipped}")

        for mode, alpha in zip(mix_modes, alphas):
            if mode in SKIP_MODES:
                continue
            mix_args = merge_mix_args(mode, alpha, cfg_mix_args)
            print(f"\n--- {mode}  (alpha={alpha}) ---")
            times = dispatch(
                mode, model, img, gt_label, mask, mix_args,
                warmup=args.warmup, n_iters=args.num_iters, device=device,
            )
            if times is None:
                continue
            tp = bs / (times.mean() / 1000)
            print(
                f"  {times.mean():.2f} ± {times.std():.2f} ms  "
                f"(p50={np.percentile(times,50):.2f}  "
                f"p95={np.percentile(times,95):.2f})  "
                f"→ {tp:.0f} img/s"
            )
            results[mode] = dict(
                mean_ms=times.mean(),
                std_ms=times.std(),
                p50_ms=np.percentile(times, 50),
                p95_ms=np.percentile(times, 95),
                throughput=tp,
            )

    # ---------------------------------------------------------------- #
    # Per-config summary table                                          #
    # ---------------------------------------------------------------- #
    if results:
        W = 65
        print(f"\n{'=' * W}")
        print(f"SUMMARY   config={config_path}")
        print(f"          batch={bs}  device={device}")
        print(f"{'=' * W}")
        hdr = (
            f"{'Mode':<26} {'Mean ms':>9} {'Std':>6} "
            f"{'P50':>6} {'P95':>6} {'img/s':>8}"
        )
        print(hdr)
        print(f"{'-' * W}")
        for mode, r in results.items():
            print(
                f"{mode:<26} {r['mean_ms']:>9.2f} {r['std_ms']:>6.2f} "
                f"{r['p50_ms']:>6.2f} {r['p95_ms']:>6.2f} "
                f"{r['throughput']:>8.0f}"
            )
        print(f"{'=' * W}")

    return results, bs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device
        if (args.device == "cpu" or torch.cuda.is_available())
        else "cpu"
    )

    configs = args.config

    if len(configs) == 1:
        # Single config: run in-process (no subprocess overhead).
        benchmark_config(args, configs[0], device)
        return

    # Multiple configs: run each in its own subprocess so that CUDA
    # allocator cache and loaded model/dataset state don't carry over.
    # Strip the config paths from sys.argv and keep all flags.
    config_set = set(configs)
    extra_argv = [a for a in sys.argv[1:] if a not in config_set]

    W = 65
    failed = []
    for i, config_path in enumerate(configs):
        print(f"\n{'#' * W}")
        print(f"# Config {i + 1}/{len(configs)}: {config_path}")
        print(f"{'#' * W}")
        cmd = [sys.executable, __file__, config_path] + extra_argv
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            failed.append(config_path)

    if failed:
        print(f"\n[WARNING] The following configs exited with errors:")
        for p in failed:
            print(f"  {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
