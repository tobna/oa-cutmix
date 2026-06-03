"""
For every validation image, estimate the mean and std of the CutMix label
error for both absolute (abs) and relative (rel) FGA label modes.

Definition
----------
Label error is the absolute difference between the standard CutMix label
and the FGA CutMix label for the *same* image A, the *same* cut bounding
box, and a random partner image B:

    lam_area    = 1 - cut_area / (H * W)          (standard CutMix)
    lam_fg_abs  = fg_vis_a / (fg_vis_a + fg_vis_b) (absolute FGA)
    lam_fg_rel  = rel_vis_a / (rel_vis_a + rel_vis_b) (relative FGA)

    err_abs = |lam_area - lam_fg_abs|
    err_rel = |lam_area - lam_fg_rel|

For each image A, --n-samples random (B, bbox) pairs are drawn to estimate
the per-image mean and std of both error types.

Output .npz (shape N_val for all 1-D arrays)
--------------------------------------------
    indices             int64   dataset index (same scheme as analyze_predictions)
    gt_labels           int64   class label
    fg_frac             float32 fg pixels / total pixels  (object size proxy)
    label_err_abs_mean  float32
    label_err_abs_std   float32
    label_err_rel_mean  float32
    label_err_rel_std   float32

Usage
-----
    python tools/analyze_label_error.py \\
        configs/classification/tiny_imagenet/r18/cutmix_fga/r18_cutmix_fga_abs.py \\
        --output work_dirs/r18_cutmix_fga_abs/label_error.npz \\
        [--n-samples 100] [--alpha 1.0] [--split val]
"""

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mmcv
import numpy as np
import torch
from torch.utils.data import DataLoader

from openmixup.datasets import build_dataset

EPS = 1e-7

# ---------------------------------------------------------------------------
# Geometry — must match cutmix_foreground_area._rand_bbox exactly
# ---------------------------------------------------------------------------


def _rand_bbox(H, W, lam):
    """Return (bbx1, bbx2, bby1, bby2) matching cutmix_foreground_area."""
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cx = np.random.randint(H)
    cy = np.random.randint(W)
    bbx1 = int(np.clip(cx - cut_h // 2, 0, H))
    bbx2 = int(np.clip(cx + cut_h // 2, 0, H))
    bby1 = int(np.clip(cy - cut_w // 2, 0, W))
    bby2 = int(np.clip(cy + cut_w // 2, 0, W))
    return bbx1, bbx2, bby1, bby2


# ---------------------------------------------------------------------------
# Per-image error estimation
# ---------------------------------------------------------------------------


def compute_errors_for_image(mask_a, all_masks, b_indices, alpha):
    """Estimate label error distribution for a single image A.

    Args:
        mask_a (np.ndarray): float32 (H, W), foreground mask for image A.
        all_masks (np.ndarray): float32 (N, H, W), all val masks.
        b_indices (np.ndarray): int (n_samples,), pre-drawn B partner indices.
        alpha (float): Beta distribution parameter for lam sampling.

    Returns:
        errors (np.ndarray): float32 (n_samples, 4): for each sample, store: abs errors as img A, rel error as img A, abs error as img B, rel error as img A
        errors_rel (np.ndarray): float32 (n_samples,)
    """
    n_samples = len(b_indices)
    H, W = mask_a.shape
    total_area = float(H * W)
    fg_a = float(mask_a.sum())

    lam_values = np.random.beta(alpha, alpha, size=n_samples).astype(np.float32)

    errors = np.empty((n_samples, 4), dtype=np.float32)

    for i in range(n_samples):
        bbx1, bbx2, bby1, bby2 = _rand_bbox(H, W, float(lam_values[i]))
        cut_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam_area = 1.0 - cut_area / total_area

        mask_b = all_masks[b_indices[i]]
        fg_b = float(mask_b.sum())

        fg_a_inside = float(mask_a[bbx1:bbx2, bby1:bby2].sum())
        fg_a_outside = fg_a - fg_a_inside
        fg_b_inside = float(mask_b[bbx1:bbx2, bby1:bby2].sum())
        fg_b_outside = fg_b - fg_b_inside

        # --- absolute mode ---
        fg_vis_a = fg_a_outside
        fg_vis_b = fg_b_inside
        total_vis = fg_vis_a + fg_vis_b
        lam_fg_abs = 0.5 if total_vis <= EPS else fg_vis_a / total_vis
        errors[i, 0] = abs(lam_area - lam_fg_abs)

        fg_invis_a = fg_a_inside
        fg_invis_b = fg_b_outside
        total_invis = fg_invis_a + fg_invis_b
        lam_fg_abs = 0.5 if total_invis <= EPS else fg_invis_a / total_invis
        errors[i, 2] = abs(lam_area - lam_fg_abs)

        # --- relative mode ---
        rel_a_inside = fg_a_inside / (fg_a + EPS)
        rel_a_outside = 1.0 - rel_a_inside
        rel_b_inside = fg_b_inside / (fg_b + EPS)
        rel_b_outside = 1.0 - rel_b_inside

        rel_vis_a = rel_a_outside
        rel_vis_b = rel_b_inside
        rel_invis_b = rel_b_inside
        rel_total = rel_vis_a + rel_vis_b
        lam_fg_rel = 0.5 if rel_total <= EPS else rel_vis_a / rel_total
        errors[i, 1] = abs(lam_area - lam_fg_rel)

        rel_invis_a = rel_a_inside
        rel_invis_b = rel_b_outside
        rel_invis_total = rel_invis_a + rel_invis_b
        lam_fg_rel = 0.5 if rel_invis_total <= EPS else rel_invis_a / rel_invis_total
        errors[i, 3] = abs(lam_area - lam_fg_rel)

    return errors


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _has_transform(pipeline_cfg, t_type):
    return any(p.get("type") == t_type for p in pipeline_cfg)


def build_val_dataset(cfg, args):
    split = args.split
    if split == "val":
        ds_cfg = copy.deepcopy(cfg.data.val)
    elif split == "train":
        ds_cfg = copy.deepcopy(cfg.data.train)
    else:
        raise ValueError(f"Unknown split: {split}")

    ds_cfg.prefetch = False

    norm_cfg = cfg.get("img_norm_cfg", dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    pipeline = list(ds_cfg.pipeline)
    if not _has_transform(pipeline, "ToTensor"):
        pipeline.append(dict(type="ToTensor"))
    if not _has_transform(pipeline, "Normalize"):
        pipeline.append(dict(type="Normalize", **norm_cfg))
    ds_cfg.pipeline = pipeline

    src = dict(ds_cfg.data_source)
    if args.data_root:
        src["root"] = args.data_root
    if args.mask_zip:
        src["mask_zip"] = args.mask_zip
    ds_cfg.data_source = src

    return build_dataset(ds_cfg)


def preload_masks(dataset, num_workers):
    """Load every sample once and store masks + metadata as numpy arrays.

    Returns:
        indices   (N,) int64
        gt_labels (N,) int64
        masks     (N, H, W) float32
    """

    print(f"dataset returns: {dataset[0]}")
    print(dataset[0].keys())
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: (
            torch.stack([b["mask"] for b in batch]),  # (B, 1, H, W) or (B, H, W)
            torch.tensor([b["gt_label"] for b in batch]),
            torch.tensor([b["idx"] for b in batch]),
        ),
    )

    all_masks = []
    all_labels = []
    all_indices = []
    total = len(dataset)
    done = 0
    next_log = 1000

    print(f"Pre-loading masks for {total} images …")
    for masks, labels, idxs in loader:
        # masks may be (B, 1, H, W) after ToTensor on a single-channel mask
        if masks.ndim == 4:
            masks = masks[:, 0]  # (B, H, W)
        # Convert binary 0/255 → 0/1
        if masks.max() > 1.0:
            masks = masks / 255.0
        all_masks.append(masks.numpy().astype(np.float32))
        all_labels.append(labels.numpy().astype(np.int64))
        all_indices.append(idxs.numpy().astype(np.int64))
        done += len(idxs)
        if done >= next_log:
            print(f"  {done}/{total} masks loaded")
            next_log += 1000

    print(f"  {done}/{total} masks loaded — done.")
    return (
        np.concatenate(all_indices),
        np.concatenate(all_labels),
        np.concatenate(all_masks),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate per-image CutMix label error for abs and rel modes")
    parser.add_argument("config", help="Classification config file path")
    parser.add_argument("--output", default=None, help="Output .npz path (default: label_error.npz next to config)")
    parser.add_argument("--n-samples", type=int, default=100, help="Random (B, bbox) pairs per image (default: 100)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Beta distribution alpha for lam sampling")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--mask-zip", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = mmcv.Config.fromfile(args.config)

    if args.output is None:
        out_dir = os.path.dirname(os.path.abspath(args.config))
        args.output = os.path.join(out_dir, "label_error.npz")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Config     : {args.config}")
    print(f"Split      : {args.split}")
    print(f"n_samples  : {args.n_samples}")
    print(f"alpha      : {args.alpha}")
    print(f"Output     : {args.output}")

    dataset = build_val_dataset(cfg, args)
    N = len(dataset)
    print(f"Dataset    : {N} images · {len(dataset.CLASSES)} classes")

    # Pre-load all masks — avoids random-access thrashing during inner loop
    indices, gt_labels, all_masks = preload_masks(dataset, args.num_workers)
    H, W = all_masks.shape[1], all_masks.shape[2]
    total_area = float(H * W)
    print(f"Mask shape : {H} x {W}  (total_area={int(total_area)})")

    # Relative foreground fraction per image (object size proxy)
    fg_frac = all_masks.sum(axis=(1, 2)) / total_area  # (N,)

    # Sort by dataset index so output order matches analyze_predictions.npz
    order = np.argsort(indices)
    indices = indices[order]
    gt_labels = gt_labels[order]
    all_masks = all_masks[order]
    fg_frac = fg_frac[order]

    # Per-image error estimation
    err_mean = np.empty((N, 4), dtype=np.float32)
    err_std = np.empty((N, 4), dtype=np.float32)

    print(f"\nEstimating label errors for {N} images …")
    next_log = 10

    for i in range(N):
        # Sample random B partners (allow self-pairing; rare and inconsequential)
        b_indices = np.random.randint(0, N, size=args.n_samples)

        errs = compute_errors_for_image(all_masks[i], all_masks, b_indices, args.alpha)

        err_mean[i] = errs.mean(axis=0)
        err_std[i] = errs.std(axis=0)

        if (i + 1) >= next_log:
            print(f"[CPU]  {i + 1}/{N} images processed")
            next_log = ((i + 1) // 100 + 1) * 100

    print(f"Done. {N}/{N} images processed.")

    results = dict(
        indices=indices,
        gt_labels=gt_labels,
        fg_frac=fg_frac.astype(np.float32),
        label_err_abs_a_mean=err_mean[:, 0],
        label_err_rel_a_mean=err_mean[:, 1],
        label_err_abs_b_mean=err_mean[:, 2],
        label_err_rel_b_mean=err_mean[:, 3],
        label_err_abs_a_std=err_std[:, 0],
        label_err_rel_a_std=err_std[:, 1],
        label_err_abs_b_std=err_std[:, 2],
        label_err_rel_b_std=err_std[:, 3],
    )
    np.savez(args.output, **results)
    print(f"\nSaved to {args.output}")

    # Summary
    print("\nSummary")
    print("=======")
    print(
        f"  mean |err_abs| as A: {err_mean[:, 0].mean():.4f}+-{err_std[:, 0].mean()}  (std across images:"
        f" {err_mean[:, 0].std():.4f})"
    )
    print(
        f"  mean |err_abs| as B: {err_mean[:, 2].mean():.4f}+-{err_std[:, 2].mean()}    (std across images:"
        f" {err_mean[:, 2].std():.4f})"
    )
    print(
        f"  mean |err_rel| as A: {err_mean[:, 1].mean():.4f}+-{err_std[:, 1].mean()}    (std across images:"
        f" {err_mean[:, 1].std():.4f})"
    )
    print(
        f"  mean |err_rel| as B: {err_mean[:, 3].mean():.4f}+-{err_std[:, 3].mean()}    (std across images:"
        f" {err_mean[:, 3].std():.4f})"
    )
    print(f"  mean fg_frac   : {fg_frac.mean():.4f}  (std: {fg_frac.std():.4f})")
    print(f"  Arrays in .npz : {list(results.keys())}")
    print(f"  Shapes         : {[(k, v.shape) for k, v in results.items()]}")


if __name__ == "__main__":
    main()
