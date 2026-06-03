#!/usr/bin/env python3
"""
Visualize images and masks from a data source defined in a config file.

For each sampled image the script shows:
  - the original image
  - the foreground mask (if the data source provides one)
  - the image with the mask overlaid

Layout (one row per sample):
    [ image ]  [ mask ]  [ overlay ]     ← mask sources
    [ image ]                            ← plain sources

Usage:
    python tools/visualize_data_source.py <config> [options]

    python tools/visualize_data_source.py \\
        configs/classification/cub200/200ep/r18/cutmix_fga/r18_cutmix_fga_rel_l1p0.py \\
        --split train --num-samples 12 --cols 4 --seed 0

    python tools/visualize_data_source.py \\
        configs/classification/imagenet/100ep/r18/cutmix_fga/r18_cutmix_fga_debug.py \\
        --data-root /data/ImageNet \\
        --mask-zip /fscratch/nauen/datasets/imagenet-masks/train_masks.zip \\
        --split train --num-samples 8 --out vis_imagenet.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mmcv
import numpy as np
from PIL import Image

from openmixup.datasets.registry import DATASOURCES

# ---------------------------------------------------------------------------
# Classname lookup
# ---------------------------------------------------------------------------

_CLASSNAMES_FILE = os.path.join(
    os.path.dirname(__file__), "prepare_data", "imagenet_classnames.txt"
)


def load_classnames(path=_CLASSNAMES_FILE):
    mapping = {}
    if not os.path.isfile(path):
        return mapping
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            synset, name = line.split(",", 1)
            mapping[synset.strip()] = name.strip()
    return mapping


_SYNSET_MAP = load_classnames()


def resolve_classname(source, label):
    """Return a human-readable class name for a given integer label."""
    classes = getattr(source, "CLASSES", None)
    if classes is None or label >= len(classes):
        return str(label)
    raw = classes[label]
    # If it looks like a WordNet synset (n########) resolve it
    if isinstance(raw, str) and raw.startswith("n") and len(raw) == 9:
        return _SYNSET_MAP.get(raw, raw)
    return str(raw)


# ---------------------------------------------------------------------------
# Data source helpers
# ---------------------------------------------------------------------------

def build_source(cfg, args):
    """Build data source from config, applying CLI overrides."""
    split = args.split

    # Try train split config first, fall back to val
    data_cfg = cfg.data.get(split) or cfg.data.get("train")
    src_cfg = dict(data_cfg.data_source)
    src_cfg["split"] = split

    if args.data_root:
        src_cfg["root"] = args.data_root
    if args.mask_zip:
        src_cfg["mask_zip"] = args.mask_zip

    src_cfg.setdefault("return_label", True)
    return DATASOURCES.build(src_cfg)


def has_masks(source, probe_idx=0):
    """Check whether the data source returns masks."""
    sample = source.get_sample(probe_idx)
    return isinstance(sample, (tuple, list)) and len(sample) == 3


def get_sample(source, idx, with_mask):
    """Return (img_pil, mask_pil_or_None, label_int)."""
    sample = source.get_sample(idx)
    if with_mask:
        img, mask, label = sample
        return img, mask, int(label)
    else:
        img, label = sample
        return img, None, int(label)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

OVERLAY_COLOR = (72, 120, 207)   # blue
OVERLAY_ALPHA = 0.4
BORDER_THICKNESS = 2


def _make_circular_kernel(radius):
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= radius ** 2


def _dilate(mask, kernel):
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = kernel.shape
    padded = np.pad(mask, ((kh // 2, kh // 2), (kw // 2, kw // 2)), constant_values=0)
    windows = sliding_window_view(padded, (kh, kw))
    return (windows & kernel).any(axis=(-2, -1))


def _erode(mask, kernel):
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = kernel.shape
    padded = np.pad(mask, ((kh // 2, kh // 2), (kw // 2, kw // 2)), constant_values=0)
    windows = sliding_window_view(padded, (kh, kw))
    return (windows | ~kernel).all(axis=(-2, -1))


def make_overlay(img_pil, mask_pil, color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA,
                 border_thickness=BORDER_THICKNESS):
    """Blend a colored mask over the image, with a solid border."""
    img = np.array(img_pil, dtype=np.float32)
    fg = np.array(mask_pil, dtype=np.float32) > 127  # (H, W) bool

    # Interior blend
    overlay = np.array(color, dtype=np.float32)
    img[fg] = (1 - alpha) * img[fg] + alpha * overlay

    # Border via dilation XOR erosion
    if border_thickness > 0:
        kernel = _make_circular_kernel(border_thickness)
        border = _dilate(fg, kernel) & ~_erode(fg, kernel)
        img[border] = overlay

    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def mask_colorize(mask_pil):
    """Convert a grayscale mask to a blue-tinted RGBA image for display."""
    arr = np.array(mask_pil, dtype=np.float32) / 255.0  # [0,1]
    H, W = arr.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, 0] = (OVERLAY_COLOR[0] * arr).astype(np.uint8)
    rgba[:, :, 1] = (OVERLAY_COLOR[1] * arr).astype(np.uint8)
    rgba[:, :, 2] = (OVERLAY_COLOR[2] * arr).astype(np.uint8)
    rgba[:, :, 3] = arr > 0.5  # alpha channel: 1 for fg, 0 for bg
    # Use white background for display
    bg = np.full((H, W, 3), 240, dtype=np.uint8)
    fg_mask = rgba[:, :, 3:4].astype(np.float32) / 255.0
    blended = (rgba[:, :, :3] * fg_mask + bg * (1 - fg_mask)).astype(np.uint8)
    return Image.fromarray(blended)


def fg_fraction(mask_pil):
    arr = np.array(mask_pil)
    return arr.mean() / 255.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize images and masks from a config data source"
    )
    parser.add_argument("config", help="mmcv config file (.py)")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"],
                        help="Dataset split to use (default: train)")
    parser.add_argument("--num-samples", type=int, default=12,
                        help="Number of images to display (default: 12)")
    parser.add_argument("--cols", type=int, default=4,
                        help="Images per row in the grid (default: 4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None,
                        help="Override the data root in the config")
    parser.add_argument("--mask-zip", default=None,
                        help="Override the mask ZIP path in the config")
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="Specific dataset indices to show (overrides --num-samples)")
    parser.add_argument("--out", default=None,
                        help="Save figure to this path instead of showing it")
    parser.add_argument("--title-fontsize", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ---- load config & build data source ----
    cfg = mmcv.Config.fromfile(args.config)
    print(f"Config : {args.config}")
    print(f"Split  : {args.split}")

    source = build_source(cfg, args)
    n = len(source)
    print(f"Source : {source.__class__.__name__}  ({n} samples)")

    with_mask = has_masks(source)
    print(f"Masks  : {'yes' if with_mask else 'no'}")

    # ---- pick indices ----
    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(n, size=min(args.num_samples, n), replace=False).tolist()
    indices = sorted(indices)

    # ---- layout ----
    # With masks: each sample occupies (cols_per_sample=3) sub-columns
    # Without    : each sample is 1 sub-column
    import matplotlib
    matplotlib.use("Agg" if args.out else "TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    cols_per_sample = 3 if with_mask else 1
    grid_cols = args.cols          # images per visual row
    n_rows = (len(indices) + grid_cols - 1) // grid_cols

    fig_w = grid_cols * cols_per_sample * 2.2
    fig_h = n_rows * 2.5
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)

    dataset_name = os.path.splitext(os.path.basename(args.config))[0]
    fig.suptitle(
        f"{source.__class__.__name__} · {dataset_name} · split={args.split}",
        fontsize=9, y=1.01,
    )

    outer = gridspec.GridSpec(
        n_rows, grid_cols,
        figure=fig,
        hspace=0.5, wspace=0.15,
    )

    col_labels = ["image", "mask", "overlay"] if with_mask else ["image"]

    for plot_i, idx in enumerate(indices):
        row = plot_i // grid_cols
        col = plot_i % grid_cols

        img_pil, mask_pil, label = get_sample(source, idx, with_mask)
        classname = resolve_classname(source, label)

        inner = gridspec.GridSpecFromSubplotSpec(
            2, cols_per_sample,
            subplot_spec=outer[row, col],
            hspace=0.05, wspace=0.05,
            height_ratios=[1, 0.06],
        )

        panels = [img_pil]
        if with_mask:
            panels.append(mask_colorize(mask_pil))
            panels.append(make_overlay(img_pil, mask_pil))

        for sub_col, (panel, col_label) in enumerate(zip(panels, col_labels)):
            ax = fig.add_subplot(inner[0, sub_col])
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#aaaaaa")
            if sub_col == 0:
                ax.set_ylabel(
                    f"#{idx}",
                    fontsize=5, labelpad=2, rotation=0, ha="right", va="center",
                )
            # column label on first row only
            if plot_i < grid_cols:
                ax.set_title(col_label, fontsize=6, pad=2, color="#555555")

        # Sample title below (spans all sub-columns)
        title_ax = fig.add_subplot(inner[1, :])
        title_ax.axis("off")
        extra = ""
        if with_mask and mask_pil is not None:
            ff = fg_fraction(mask_pil)
            extra = f"  fg={ff:.0%}"
        title_ax.text(
            0.5, 0.5,
            f"{classname} (cls {label}){extra}",
            ha="center", va="center",
            fontsize=args.title_fontsize,
            transform=title_ax.transAxes,
            wrap=True,
        )

    plt.tight_layout()

    if args.out:
        fig.savefig(args.out, bbox_inches="tight", dpi=150)
        print(f"\nSaved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
