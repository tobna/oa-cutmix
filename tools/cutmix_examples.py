#!/usr/bin/env python3
"""
Generate CutMix example images with mask overlays and metadata.

For each requested image pair, applies N different CutMix cuts and saves:

  out_dir/
    pair_00/
      img_a.png           original image A (denormalized)
      img_b.png           original image B
      mask_a.png          segmentation mask A  (grayscale)
      mask_b.png          segmentation mask B
      img_a_overlay.png   image A with blue  fg overlay
      img_b_overlay.png   image B with red   fg overlay
      metadata.json       pair-level info (labels, fg pixel counts)
      sample_00/
        mixed.png         CutMix result
        mixed_mask.png    colored mask overlay:
                            blue  = A object pixels still visible
                            red   = B object pixels pasted in
                            bbox  = white dashed outline
        metadata.json     lam_area, lam_fg, fg counts, bbox
      sample_01/ ...
    pair_01/ ...

Usage:
    python tools/cutmix_examples.py \
        configs/classification/tiny_imagenet/r18/cutmix_fga/r18_cutmix_fga_abs_l0p8.py \
        --out-dir cutmix_examples \
        --mask-zip /ds-sds/images/tiny-imagenet/train_masks.zip \
        [--data-root data/TinyImageNetHD] \
        [--num-pairs 2] [--num-samples 10] [--seed 0]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mmcv
import numpy as np
import torch
from PIL import Image, ImageDraw

from openmixup.datasets import build_dataset

# ---------------------------------------------------------------------------
# Config helpers (same pattern as cutmix_collect.py)
# ---------------------------------------------------------------------------


def _has_transform(pipeline_cfg, t_type):
    return any(p.get("type") == t_type for p in pipeline_cfg)


def prepare_dataset_cfg(cfg, args):
    import copy

    ds_cfg = copy.deepcopy(cfg.data.train)
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
    return ds_cfg, norm_cfg


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------


def _rand_bbox(H, W, lam):
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


def cutmix_single(img_a, mask_a, img_b, mask_b, alpha=1.0):
    """Apply one CutMix draw to a single pair. Returns mixed tensors + stats dict."""
    C, H, W = img_a.shape
    lam = float(np.random.beta(alpha, alpha))
    bbx1, bbx2, bby1, bby2 = _rand_bbox(H, W, lam)

    mixed = img_a.clone()
    mixed[:, bbx1:bbx2, bby1:bby2] = img_b[:, bbx1:bbx2, bby1:bby2]

    cut_area = (bbx2 - bbx1) * (bby2 - bby1)
    total_area = H * W
    lam_area = 1.0 - cut_area / total_area

    region = torch.zeros(H, W)
    region[bbx1:bbx2, bby1:bby2] = 1.0

    ma = mask_a[0]
    mb = mask_b[0]
    fg_vis_a = (ma * (1.0 - region)).sum().item()
    fg_vis_b = (mb * region).sum().item()
    lam_fg = fg_vis_a / (fg_vis_a + fg_vis_b + 1e-8)

    stats = dict(
        lam_area=round(lam_area, 6),
        lam_fg=round(lam_fg, 6),
        fg_visible_a_in_mix=round(fg_vis_a, 1),
        fg_visible_b_in_mix=round(fg_vis_b, 1),
        cut_area=cut_area,
        total_area=total_area,
        bbox=dict(bbx1=bbx1, bbx2=bbx2, bby1=bby1, bby2=bby2),
    )
    return mixed, region, stats


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def denorm(t, mean, std):
    """Tensor (C,H,W) → uint8 PIL Image."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    arr = (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def mask_to_pil(mask_tensor):
    """(1,H,W) float [0,1] → grayscale PIL."""
    arr = (mask_tensor[0].numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_circular_kernel(radius: int) -> np.ndarray:
    """Boolean circular kernel of given radius."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x**2 + y**2) <= radius**2


def _binary_erosion_np(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    windows = sliding_window_view(padded, (kh, kw))  # (H, W, kh, kw)
    # Only check pixels inside the circular kernel
    # Pixels outside the kernel are ignored (treated as True so they don't block erosion)
    kernel_inv = ~kernel  # positions outside kernel -> set to True so .all() ignores them
    return (windows | kernel_inv).all(axis=(-2, -1))


def _binary_dilation_np(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask with the given boolean kernel."""
    from numpy.lib.stride_tricks import sliding_window_view

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    windows = sliding_window_view(padded, (kh, kw))  # (H, W, kh, kw)
    # A pixel is set if ANY kernel-selected neighbour is foreground
    return (windows & kernel).any(axis=(-2, -1))


def _binary_erosion_np(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    windows = sliding_window_view(padded, (kh, kw))  # (H, W, kh, kw)
    # Only check pixels inside the circular kernel
    # Pixels outside the kernel are ignored (treated as True so they don't block erosion)
    kernel_inv = ~kernel  # positions outside kernel -> set to True so .all() ignores them
    return (windows | kernel_inv).all(axis=(-2, -1))


def make_masked_image(img_pil, mask_tensor, color, alpha=0.3, border_thickness=1, border_color=None):
    """
    Overlay a solid color on the foreground pixels of img_pil,
    and draw a solid border along the actual mask boundary.

    Args:
        img_pil:            PIL RGB image.
        mask_tensor:        (H, W) float tensor with values in [0, 1].
        color:              (R, G, B) tuple to paint over foreground pixels.
        alpha:              Blend factor for the interior overlay (0 = invisible, 1 = solid).
        border_thickness:   Width of the border drawn along the mask edge (in pixels).
        border_color:       (R, G, B) tuple for the border. Defaults to `color` if None.
    Returns:
        PIL RGB image with the colored overlay and border blended in.
    """
    if border_color is None:
        border_color = color

    img_arr = np.array(img_pil, dtype=np.float32)
    fg = mask_tensor.numpy() > 0.5  # (H, W) bool

    # ------------------------------------------------------------------
    # 1. Interior overlay
    # ------------------------------------------------------------------
    overlay = np.array(color, dtype=np.float32)
    img_arr[fg] = (1 - alpha) * img_arr[fg] + alpha * overlay

    # ------------------------------------------------------------------
    # 2. Compute the mask border via dilation XOR erosion
    #    using a circular kernel for an isotropic (even) border
    # ------------------------------------------------------------------
    kernel = _make_circular_kernel(border_thickness)

    eroded = _binary_erosion_np(fg, kernel)
    dilated = _binary_dilation_np(fg, kernel)

    border_mask = dilated & ~eroded  # even ring around the mask boundary

    # ------------------------------------------------------------------
    # 3. Paint the border as fully solid (alpha = 1.0)
    # ------------------------------------------------------------------
    img_arr[border_mask] = np.array(border_color, dtype=np.float32)

    return Image.fromarray(img_arr.clip(0, 255).astype(np.uint8))


def make_mixed_mask(mask_a, mask_b, region, bbx1, bbx2, bby1, bby2):
    """
    Colored overlay showing object provenance in the mixed image:
      Blue  (#4878cf) = A's fg pixels outside the cut box (kept from A)
      Red   (#d65f5f) = B's fg pixels inside the cut box  (pasted from B)
      White dashed rectangle = cut box boundary
    Background is black.
    """
    H, W = mask_a.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    fg_a_vis = (mask_a * (1.0 - region)).numpy()  # A pixels still in image
    fg_b_vis = (mask_b * region).numpy()  # B pixels pasted in

    # Blue for A
    rgb[fg_a_vis > 0.5, 0] = 72
    rgb[fg_a_vis > 0.5, 1] = 120
    rgb[fg_a_vis > 0.5, 2] = 207

    # Red for B  (overwrites overlapping pixels)
    rgb[fg_b_vis > 0.5, 0] = 214
    rgb[fg_b_vis > 0.5, 1] = 95
    rgb[fg_b_vis > 0.5, 2] = 95

    img = Image.fromarray(rgb)

    # Draw dashed bbox outline
    draw = ImageDraw.Draw(img)
    box = [bby1, bbx1, bby2, bbx2]  # PIL: (left, top, right, bottom)
    dash = 8
    # top / bottom
    for x in range(box[0], box[2], dash * 2):
        draw.line([(x, box[1]), (min(x + dash, box[2]), box[1])], fill=(255, 255, 255), width=1)
        draw.line([(x, box[3]), (min(x + dash, box[2]), box[3])], fill=(255, 255, 255), width=1)
    # left / right
    for y in range(box[1], box[3], dash * 2):
        draw.line([(box[0], y), (box[0], min(y + dash, box[3]))], fill=(255, 255, 255), width=1)
        draw.line([(box[2], y), (box[2], min(y + dash, box[3]))], fill=(255, 255, 255), width=1)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate CutMix example images with mask overlays")
    parser.add_argument("config", help="Classification config (should use MaskImageFolder)")
    parser.add_argument("--out-dir", default="cutmix_examples")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--mask-zip", default=None)
    parser.add_argument("--num-pairs", type=int, default=2, help="Number of image pairs to sample")
    parser.add_argument("--num-samples", type=int, default=10, help="CutMix draws per pair")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- dataset ----
    cfg = mmcv.Config.fromfile(args.config)
    ds_cfg, norm_cfg = prepare_dataset_cfg(cfg, args)
    ds_cfg.data_source["split"] = args.split
    dataset = build_dataset(ds_cfg)

    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    n = len(dataset)
    pair_indices = [(int(np.random.randint(n)), int(np.random.randint(n))) for _ in range(args.num_pairs)]

    print(f"Config : {args.config}")
    print(f"Output : {args.out_dir}/")
    print(f"Pairs  : {args.num_pairs}  ×  {args.num_samples} samples each\n")

    for pair_idx, (ia, ib) in enumerate(pair_indices):
        sample_a = dataset[ia]
        sample_b = dataset[ib]

        img_a, mask_a = sample_a["img"], sample_a["mask"]
        img_b, mask_b = sample_b["img"], sample_b["mask"]
        label_a = int(sample_a["gt_label"])
        label_b = int(sample_b["gt_label"])

        pair_dir = os.path.join(args.out_dir, f"pair_{pair_idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)

        # Save originals
        COLOR_A = (72, 120, 207)  # blue  – consistent with make_mixed_mask
        COLOR_B = (214, 95, 95)  # red

        pil_a = denorm(img_a, mean, std)
        pil_b = denorm(img_b, mean, std)
        pil_a.save(os.path.join(pair_dir, "img_a.png"))
        pil_b.save(os.path.join(pair_dir, "img_b.png"))
        mask_to_pil(mask_a).save(os.path.join(pair_dir, "mask_a.png"))
        mask_to_pil(mask_b).save(os.path.join(pair_dir, "mask_b.png"))
        make_masked_image(pil_a, mask_a[0], COLOR_A).save(os.path.join(pair_dir, "img_a_overlay.png"))
        make_masked_image(pil_b, mask_b[0], COLOR_B).save(os.path.join(pair_dir, "img_b_overlay.png"))

        pair_meta = dict(
            idx_a=ia,
            idx_b=ib,
            label_a=label_a,
            label_b=label_b,
            class_a=dataset.CLASSES[label_a] if dataset.CLASSES else label_a,
            class_b=dataset.CLASSES[label_b] if dataset.CLASSES else label_b,
            fg_pixels_a_total=round(mask_a[0].sum().item(), 1),
            fg_pixels_b_total=round(mask_b[0].sum().item(), 1),
            img_size=list(img_a.shape[-2:]),
        )
        with open(os.path.join(pair_dir, "metadata.json"), "w") as f:
            json.dump(pair_meta, f, indent=2)

        print(
            f"  pair {pair_idx:02d}  "
            f"A=idx{ia} cls{label_a}({pair_meta['class_a']})  "
            f"B=idx{ib} cls{label_b}({pair_meta['class_b']})"
        )

        for s_idx in range(args.num_samples):
            mixed, region, stats = cutmix_single(img_a, mask_a, img_b, mask_b, alpha=args.alpha)

            bbox = stats["bbox"]
            bbx1, bbx2, bby1, bby2 = bbox["bbx1"], bbox["bbx2"], bbox["bby1"], bbox["bby2"]

            mixed_mask = make_mixed_mask(mask_a[0], mask_b[0], region, bbx1, bbx2, bby1, bby2)

            sample_dir = os.path.join(pair_dir, f"sample_{s_idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            pil_mixed = denorm(mixed, mean, std)
            pil_mixed.save(os.path.join(sample_dir, "mixed.png"))
            mixed_mask.save(os.path.join(sample_dir, "mixed_mask.png"))

            pil_mixed_overlay = make_masked_image(pil_mixed, mask_a[0] * (1.0 - region), COLOR_A)
            pil_mixed_overlay = make_masked_image(pil_mixed_overlay, mask_b[0] * region, COLOR_B)
            pil_mixed_overlay.save(os.path.join(sample_dir, "mixed_overlay.png"))

            stats["label_a"] = label_a
            stats["label_b"] = label_b
            with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
                json.dump(stats, f, indent=2)

            print(
                f"    sample {s_idx:02d}  "
                f"lam_area={stats['lam_area']:.3f}  lam_fg={stats['lam_fg']:.3f}  "
                f"bbox=[{bbx1}:{bbx2},{bby1}:{bby2}]"
            )

    print(f"\nDone. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()
