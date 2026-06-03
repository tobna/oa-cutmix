#!/usr/bin/env python3
"""
CutMix visualization and label tracking script for TinyImageNet with mask images.

Loads dataset and pipeline directly from a config file (e.g. a cutmix_fga config),
then runs CutMix and tracks:

  - Area-based CutMix label  (λ_area = 1 - cut_bbox_area / total_image_area)
  - Foreground-area label    (λ_fg   = fg_visible_A / (fg_visible_A + fg_visible_B))
  - Object pixels in each original image (from the segmentation mask)
  - Object pixels visible from each source image in the mixed output

Usage:
    python tools/cutmix_mask_demo.py \
        configs/classification/tiny_imagenet/r18/cutmix_fga/r18_cutmix_fga_abs_l0p8.py \
        [--mask-zip /path/to/train_masks.zip] \
        [--data-root data/TinyImageNetHD] \
        [--batch-size 16] [--num-batches 5] [--alpha 1.0] \
        [--visualize] [--save-viz cutmix_viz.png]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import mmcv
import numpy as np
import torch

matplotlib.use("Agg")  # headless-safe; use --show for interactive
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from openmixup.datasets import build_dataset

# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def _has_transform(pipeline_cfg, t_type):
    return any(p.get("type") == t_type for p in pipeline_cfg)


def prepare_dataset_cfg(cfg, args):
    """
    Derive a ClassificationDataset config from cfg.data.train that:
      - uses MaskImageFolder (overriding root / mask_zip if given on CLI)
      - has prefetch=False so that ToTensor + Normalize are in the pipeline
        and __getitem__ returns proper float tensors
    """
    import copy

    ds_cfg = copy.deepcopy(cfg.data.train)

    # ---- force prefetch=False so the pipeline includes ToTensor+Normalize ----
    ds_cfg.prefetch = False

    # Append ToTensor / Normalize if the config omitted them (prefetch=True path)
    norm_cfg = cfg.get("img_norm_cfg", dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    pipeline = list(ds_cfg.pipeline)
    if not _has_transform(pipeline, "ToTensor"):
        pipeline.append(dict(type="ToTensor"))
    if not _has_transform(pipeline, "Normalize"):
        pipeline.append(dict(type="Normalize", **norm_cfg))
    ds_cfg.pipeline = pipeline

    # ---- CLI overrides for data_source ----
    src = dict(ds_cfg.data_source)  # mutable copy
    if args.data_root:
        src["root"] = args.data_root
    if args.mask_zip:
        src["mask_zip"] = args.mask_zip
    ds_cfg.data_source = src

    return ds_cfg, norm_cfg


# ---------------------------------------------------------------------------
# CutMix with full stat tracking
# ---------------------------------------------------------------------------


def _rand_bbox(H: int, W: int, lam: float):
    """Random cut bounding box — replicates cutmix.py logic."""
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


def apply_cutmix(imgs, masks, labels, alpha=1.0):
    """
    Apply CutMix to a batch and track area / foreground label statistics.

    Args:
        imgs:   (N, C, H, W)  normalized image tensors
        masks:  (N, 1, H, W)  binary segmentation masks, values in [0, 1]
        labels: (N,)          integer class labels
        alpha:  Beta distribution parameter

    Returns:
        mixed_imgs  (N, C, H, W)
        labels_a    (N,)   labels for the background image (A)
        labels_b    (N,)   labels for the pasted image (B)
        imgs_b      (N, C, H, W)  the B images (for visualization)
        masks_b     (N, 1, H, W)  the B masks
        stats       list[dict] — per-sample metrics:
            label_a / label_b         class indices
            lam_area                  standard CutMix λ (fraction of A remaining)
            lam_fg                    foreground-weighted λ
            fg_pixels_a_total         object pixels in original image A
            fg_pixels_b_total         object pixels in original image B
            fg_visible_a_in_mix       A's fg pixels outside the cut region
            fg_visible_b_in_mix       B's fg pixels inside the cut region
            bbox                      (bbx1, bbx2, bby1, bby2)  row/col coords
            cut_area / total_area     raw pixel counts
    """
    N, C, H, W = imgs.shape
    lam_sampled = float(np.random.beta(alpha, alpha))

    rand_index = torch.randperm(N)
    imgs_b = imgs[rand_index]
    masks_b = masks[rand_index]
    labels_b = labels[rand_index]

    bbx1, bbx2, bby1, bby2 = _rand_bbox(H, W, lam_sampled)

    mixed_imgs = imgs.clone()
    mixed_imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs_b[:, :, bbx1:bbx2, bby1:bby2]

    cut_area = (bbx2 - bbx1) * (bby2 - bby1)
    total_area = H * W
    lam_area = 1.0 - cut_area / total_area

    region_mask = torch.zeros(H, W)
    region_mask[bbx1:bbx2, bby1:bby2] = 1.0

    stats = []
    for i in range(N):
        ma = masks[i, 0]
        mb = masks_b[i, 0]

        fg_a_total = ma.sum().item()
        fg_b_total = mb.sum().item()
        fg_vis_a = (ma * (1.0 - region_mask)).sum().item()
        fg_vis_b = (mb * region_mask).sum().item()
        lam_fg = fg_vis_a / (fg_vis_a + fg_vis_b + 1e-8)

        stats.append(
            dict(
                label_a=labels[i].item(),
                label_b=labels_b[i].item(),
                lam_area=lam_area,
                lam_fg=lam_fg,
                fg_pixels_a_total=fg_a_total,
                fg_pixels_b_total=fg_b_total,
                fg_visible_a_in_mix=fg_vis_a,
                fg_visible_b_in_mix=fg_vis_b,
                bbox=(bbx1, bbx2, bby1, bby2),
                cut_area=cut_area,
                total_area=total_area,
            )
        )

    return mixed_imgs, labels, labels_b, imgs_b, masks_b, stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _denorm(t, mean, std):
    """Reverse normalization → HxWx3 float32 in [0, 1]."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_batch(imgs, masks, imgs_b, masks_b, mixed_imgs, stats, norm_cfg, n_show=4, save_path=None):
    """Grid: Image A | Mask A | Image B (cut region highlighted) | Mask B | Mixed | Stats"""
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    n_show = min(n_show, len(stats))
    fig, axes = plt.subplots(n_show, 6, figsize=(22, 4 * n_show))
    if n_show == 1:
        axes = axes[None]

    for col, title in enumerate(["Image A", "Mask A", "Image B (source)", "Mask B", "Mixed", "Statistics"]):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold")

    for row in range(n_show):
        s = stats[row]
        bbx1, bbx2, bby1, bby2 = s["bbox"]

        img_a_np = _denorm(imgs[row], mean, std)
        img_b_np = _denorm(imgs_b[row], mean, std)
        img_mix = _denorm(mixed_imgs[row], mean, std)
        mask_a_np = masks[row, 0].numpy()
        mask_b_np = masks_b[row, 0].numpy()

        axes[row, 0].imshow(img_a_np)
        axes[row, 0].set_ylabel(f"cls {s['label_a']}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_a_np, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(img_b_np)
        axes[row, 2].add_patch(
            mpatches.Rectangle(
                (bby1, bbx1),
                bby2 - bby1,
                bbx2 - bbx1,
                linewidth=2,
                edgecolor="lime",
                facecolor="lime",
                alpha=0.25,
            )
        )
        axes[row, 2].set_ylabel(f"cls {s['label_b']}", fontsize=8)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(mask_b_np, cmap="hot", vmin=0, vmax=1)
        axes[row, 3].axis("off")

        axes[row, 4].imshow(img_mix)
        axes[row, 4].add_patch(
            mpatches.Rectangle(
                (bby1, bbx1),
                bby2 - bby1,
                bbx2 - bbx1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )
        axes[row, 4].axis("off")

        axes[row, 5].axis("off")
        text = (
            f"λ_area  = {s['lam_area']:.3f}\n"
            f"λ_fg    = {s['lam_fg']:.3f}\n"
            f"Δλ      = {abs(s['lam_area'] - s['lam_fg']):.3f}\n\n"
            f"FG orig A : {s['fg_pixels_a_total']:>6.0f} px\n"
            f"FG orig B : {s['fg_pixels_b_total']:>6.0f} px\n\n"
            f"FG A→mix  : {s['fg_visible_a_in_mix']:>6.0f} px\n"
            f"FG B→mix  : {s['fg_visible_b_in_mix']:>6.0f} px\n\n"
            f"Cut: rows [{bbx1}:{bbx2}]\n"
            f"     cols [{bby1}:{bby2}]\n"
            f"Area: {s['cut_area']}/{s['total_area']} px²"
        )
        axes[row, 5].text(
            0.05,
            0.95,
            text,
            transform=axes[row, 5].transAxes,
            fontsize=8,
            va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(all_stats):
    if not all_stats:
        return
    lam_area = np.array([s["lam_area"] for s in all_stats])
    lam_fg = np.array([s["lam_fg"] for s in all_stats])
    fg_a = np.array([s["fg_pixels_a_total"] for s in all_stats])
    fg_b = np.array([s["fg_pixels_b_total"] for s in all_stats])
    fg_va = np.array([s["fg_visible_a_in_mix"] for s in all_stats])
    fg_vb = np.array([s["fg_visible_b_in_mix"] for s in all_stats])

    def row(name, arr):
        print(f"  {name:<28}  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")

    print("\n" + "=" * 65)
    print(f"AGGREGATE STATISTICS  (n={len(all_stats)} samples)")
    print("=" * 65)
    row("λ_area (standard CutMix)", lam_area)
    row("λ_fg   (foreground-based)", lam_fg)
    row("|λ_area − λ_fg|", np.abs(lam_area - lam_fg))
    print()
    row("FG pixels in orig A (px)", fg_a)
    row("FG pixels in orig B (px)", fg_b)
    print()
    row("FG from A visible in mix", fg_va)
    row("FG from B visible in mix", fg_vb)
    print()
    print(f"  Correlation(λ_area, λ_fg): {np.corrcoef(lam_area, lam_fg)[0, 1]:.4f}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CutMix label tracking for TinyImageNet — loads pipeline from config")
    parser.add_argument(
        "config", help="Path to a classification config file (e.g. configs/.../cutmix_fga/r18_cutmix_fga_abs_l0p8.py)"
    )
    parser.add_argument("--data-root", default=None, help="Override data_source.root from the config")
    parser.add_argument("--mask-zip", default=None, help="Override data_source.mask_zip from the config")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0, help="Beta(alpha, alpha) for CutMix lambda sampling")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--visualize", action="store_true", help="Save a visualization grid for the first batch")
    parser.add_argument("--save-viz", default="cutmix_viz.png")
    parser.add_argument("--show", action="store_true", help="Show plot interactively instead of saving")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.show:
        matplotlib.use("TkAgg")

    # ---- load config ----
    cfg = mmcv.Config.fromfile(args.config)
    print(f"Config: {args.config}")

    ds_cfg, norm_cfg = prepare_dataset_cfg(cfg, args)

    # Override the split
    ds_cfg.data_source["split"] = args.split

    print(f"Data source: {ds_cfg.data_source.get('type')}  root={ds_cfg.data_source.get('root')}  split={args.split}")
    print(f"Pipeline ({len(ds_cfg.pipeline)} transforms): {[p['type'] for p in ds_cfg.pipeline]}")

    dataset = build_dataset(ds_cfg)
    print(f"Dataset: {len(dataset)} images  ·  {len(dataset.CLASSES)} classes")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda batch: (
            torch.stack([b["img"] for b in batch]),
            torch.stack([b["mask"] for b in batch]),
            torch.tensor([b["gt_label"] for b in batch]),
        ),
    )

    all_stats = []
    first_batch_data = None

    header = (
        f"{'#':>4}  {'cls_a':>5}  {'cls_b':>5}  "
        f"{'λ_area':>7}  {'λ_fg':>7}  "
        f"{'fg_a_orig':>10}  {'fg_b_orig':>10}  "
        f"{'fg_a_mix':>9}  {'fg_b_mix':>9}"
    )

    for batch_idx, (imgs, masks, labels) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        mixed_imgs, labels_a, labels_b, imgs_b, masks_b, stats = apply_cutmix(imgs, masks, labels, alpha=args.alpha)
        all_stats.extend(stats)

        print(f"\n── Batch {batch_idx + 1}/{args.num_batches}  (imgs {tuple(imgs.shape)}) ──")
        print(header)
        for i, s in enumerate(stats):
            print(
                f"{i:>4}  {s['label_a']:>5}  {s['label_b']:>5}  "
                f"{s['lam_area']:>7.3f}  {s['lam_fg']:>7.3f}  "
                f"{s['fg_pixels_a_total']:>10.0f}  {s['fg_pixels_b_total']:>10.0f}  "
                f"{s['fg_visible_a_in_mix']:>9.0f}  {s['fg_visible_b_in_mix']:>9.0f}"
            )

        if batch_idx == 0 and (args.visualize or args.show):
            first_batch_data = (imgs, masks, imgs_b, masks_b, mixed_imgs, stats)

    print_summary(all_stats)

    if first_batch_data is not None:
        imgs, masks, imgs_b, masks_b, mixed_imgs, stats = first_batch_data
        visualize_batch(
            imgs,
            masks,
            imgs_b,
            masks_b,
            mixed_imgs,
            stats,
            norm_cfg=norm_cfg,
            n_show=min(4, args.batch_size),
            save_path=None if args.show else args.save_viz,
        )


if __name__ == "__main__":
    main()
