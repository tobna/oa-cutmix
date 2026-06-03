#!/usr/bin/env python3
"""Generate data/meta/ImageNet200/classes.txt from the TinyImageNetHD folder.

Usage:
    python tools/prepare_data/create_imagenet200_classes.py \
        --tiny-imagenet-root data/TinyImageNetHD \
        --output data/meta/ImageNet200/classes.txt
"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiny-imagenet-root",
        default="data/TinyImageNetHD",
        help="Root of TinyImageNetHD (must contain a 'train' subdir)",
    )
    parser.add_argument(
        "--output",
        default="data/meta/ImageNet200/classes.txt",
        help="Output path for the classes file",
    )
    args = parser.parse_args()

    train_dir = Path(args.tiny_imagenet_root) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")

    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if len(classes) == 0:
        raise ValueError(f"No class subdirectories found in {train_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(classes) + "\n")

    print(f"Wrote {len(classes)} classes to {args.output}")


if __name__ == "__main__":
    main()
