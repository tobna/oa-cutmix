#!/usr/bin/env python3
"""
Collect CutMix statistics to a CSV file.

Loads dataset and pipeline from a config file, applies CutMix across many
batches, and writes one row per mixed sample to a CSV.

CSV columns:
    sample_idx          global row counter
    idx_a, idx_b        dataset indices of the two images
    label_a, label_b    class indices
    lam_sampled         raw Beta sample before bbox clipping
    lam_area            1 - cut_area / total_area  (standard CutMix λ)
    lam_fg              fg_visible_a / (fg_visible_a + fg_visible_b)
    fg_pixels_a_total   foreground pixels in original image A
    fg_pixels_b_total   foreground pixels in original image B
    fg_visible_a_in_mix A's fg pixels outside the cut region
    fg_visible_b_in_mix B's fg pixels inside the cut region
    cut_area            pixels in the cut bounding box
    total_area          total image pixels (H * W)
    bbx1, bbx2          cut box row range  [bbx1, bbx2)
    bby1, bby2          cut box col range  [bby1, bby2)

Usage:
    python tools/cutmix_collect.py \
        configs/classification/tiny_imagenet/r18/cutmix_fga/r18_cutmix_fga_abs_l0p8.py \
        --out cutmix_stats.csv \
        [--mask-zip /path/to/train_masks.zip] \
        [--data-root data/TinyImageNetHD] \
        [--batch-size 64] [--num-samples 10000] \
        [--alpha 1.0] [--split train] [--seed 42]
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mmcv
import numpy as np
import torch
from torch.utils.data import DataLoader

from openmixup.datasets import build_dataset


# ---------------------------------------------------------------------------
# Config helpers  (shared logic with cutmix_mask_demo.py)
# ---------------------------------------------------------------------------

def _has_transform(pipeline_cfg, t_type):
    return any(p.get("type") == t_type for p in pipeline_cfg)


def prepare_dataset_cfg(cfg, args):
    import copy
    ds_cfg = copy.deepcopy(cfg.data.train)
    ds_cfg.prefetch = False

    norm_cfg = cfg.get("img_norm_cfg", dict(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]))
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
    return ds_cfg


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


def apply_cutmix(imgs, masks, labels, idxs, alpha):
    N, C, H, W = imgs.shape
    lam_sampled = float(np.random.beta(alpha, alpha))

    rand_index = torch.randperm(N)
    imgs_b  = imgs[rand_index]
    masks_b = masks[rand_index]
    labels_b = labels[rand_index]
    idxs_b   = idxs[rand_index]

    bbx1, bbx2, bby1, bby2 = _rand_bbox(H, W, lam_sampled)

    cut_area   = (bbx2 - bbx1) * (bby2 - bby1)
    total_area = H * W
    lam_area   = 1.0 - cut_area / total_area

    region_mask = torch.zeros(H, W)
    region_mask[bbx1:bbx2, bby1:bby2] = 1.0

    rows = []
    for i in range(N):
        ma = masks[i, 0]
        mb = masks_b[i, 0]

        fg_a_total = ma.sum().item()
        fg_b_total = mb.sum().item()
        fg_vis_a   = (ma * (1.0 - region_mask)).sum().item()
        fg_vis_b   = (mb * region_mask).sum().item()
        lam_fg     = fg_vis_a / (fg_vis_a + fg_vis_b + 1e-8)

        rows.append(dict(
            idx_a=idxs[i].item(),
            idx_b=idxs_b[i].item(),
            label_a=labels[i].item(),
            label_b=labels_b[i].item(),
            lam_sampled=lam_sampled,
            lam_area=lam_area,
            lam_fg=lam_fg,
            fg_pixels_a_total=fg_a_total,
            fg_pixels_b_total=fg_b_total,
            fg_visible_a_in_mix=fg_vis_a,
            fg_visible_b_in_mix=fg_vis_b,
            cut_area=cut_area,
            total_area=total_area,
            bbx1=bbx1, bbx2=bbx2,
            bby1=bby1, bby2=bby2,
        ))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "sample_idx",
    "idx_a", "idx_b",
    "label_a", "label_b",
    "lam_sampled", "lam_area", "lam_fg",
    "fg_pixels_a_total", "fg_pixels_b_total",
    "fg_visible_a_in_mix", "fg_visible_b_in_mix",
    "cut_area", "total_area",
    "bbx1", "bbx2", "bby1", "bby2",
]


def main():
    parser = argparse.ArgumentParser(
        description="Collect CutMix statistics to CSV"
    )
    parser.add_argument("config",
                        help="Classification config file path")
    parser.add_argument("--out",         default="cutmix_stats.csv",
                        help="Output CSV path")
    parser.add_argument("--data-root",   default=None)
    parser.add_argument("--mask-zip",    default=None)
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=10_000,
                        help="Approximate number of mixed samples to collect "
                             "(0 = full dataset pass)")
    parser.add_argument("--alpha",       type=float, default=1.0)
    parser.add_argument("--split",       default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = mmcv.Config.fromfile(args.config)
    ds_cfg = prepare_dataset_cfg(cfg, args)
    ds_cfg.data_source["split"] = args.split

    print(f"Config  : {args.config}")
    print(f"Source  : {ds_cfg.data_source.get('type')}  "
          f"root={ds_cfg.data_source.get('root')}  split={args.split}")
    print(f"Pipeline: {[p['type'] for p in ds_cfg.pipeline]}")

    dataset = build_dataset(ds_cfg)
    print(f"Dataset : {len(dataset)} images  ·  {len(dataset.CLASSES)} classes")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=lambda batch: (
            torch.stack([b["img"]       for b in batch]),
            torch.stack([b["mask"]      for b in batch]),
            torch.tensor([b["gt_label"] for b in batch]),
            torch.tensor([b["idx"]      for b in batch]),
        ),
    )

    target = args.num_samples if args.num_samples > 0 else float("inf")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        sample_idx = 0
        for batch_idx, (imgs, masks, labels, idxs) in enumerate(loader):
            rows = apply_cutmix(imgs, masks, labels, idxs, alpha=args.alpha)
            for row in rows:
                row["sample_idx"] = sample_idx
                writer.writerow(row)
                sample_idx += 1

            if (batch_idx + 1) % 20 == 0 or sample_idx >= target:
                print(f"  {sample_idx:>7} samples written ...", flush=True)

            if sample_idx >= target:
                break

    print(f"\nDone. {sample_idx} rows → {args.out}")


if __name__ == "__main__":
    main()
