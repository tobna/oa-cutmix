"""Degrade mask ZIP files for ablation studies.

Creates three degraded variants of a SAM3 mask ZIP:
  1. bbox   — each mask replaced by its bounding box rectangle
  2. intra  — masks shuffled within each class (intra-class only)
  3. inter  — masks shuffled across all classes (inter-class)

Usage:
    python degrade_mask_zip.py masks.zip
    python degrade_mask_zip.py masks.zip --output-dir /path/to/dir

Output files (written next to the input by default):
    masks_bbox.zip
    masks_intra_shuffle.zip
    masks_inter_shuffle.zip
"""
import argparse
import io
import random
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Mask transforms
# ---------------------------------------------------------------------------

def mask_to_bbox(mask_arr: np.ndarray) -> np.ndarray:
    """Return a filled bounding-box mask for the non-zero region."""
    rows = np.any(mask_arr > 0, axis=1)
    cols = np.any(mask_arr > 0, axis=0)
    if not rows.any():
        # blank mask — keep blank
        return mask_arr.copy()
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    out = np.zeros_like(mask_arr)
    out[rmin : rmax + 1, cmin : cmax + 1] = 255
    return out


def encode_mask(mask_arr: np.ndarray) -> bytes:
    img = Image.fromarray(mask_arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def read_mask(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.array(img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_zip", type=str, help="Path to the source mask ZIP")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output ZIPs (default: same as input)",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffles"
    )
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    src_path = Path(args.input_zip)
    assert src_path.exists(), f"Input ZIP not found: {src_path}"

    out_dir = Path(args.output_dir) if args.output_dir else src_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = src_path.stem
    bbox_path = out_dir / f"{stem}_bbox.zip"
    intra_path = out_dir / f"{stem}_intra_shuffle.zip"
    inter_path = out_dir / f"{stem}_inter_shuffle.zip"

    # ------------------------------------------------------------------
    # 1. Read all entries and group by class (first path component)
    # ------------------------------------------------------------------
    print("Reading source ZIP …")
    with ZipFile(src_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        raw: dict[str, bytes] = {}
        for name in tqdm(names, desc="Loading masks"):
            raw[name] = zf.read(name)

    # Group names by class (first directory component).
    # Flat zips (aircraft/cars) have no subfolder → intra-class shuffle
    # is undefined and will be skipped.
    by_class: dict[str, list[str]] = defaultdict(list)
    for name in names:
        parts = name.split("/")
        cls = parts[0] if len(parts) > 1 else None
        by_class[cls].append(name)

    has_classes = None not in by_class
    print(
        f"Loaded {len(names)} masks"
        + (f" across {len(by_class)} classes." if has_classes else " (flat ZIP, no class subfolders).")
    )

    # ------------------------------------------------------------------
    # 2. BBox degradation
    # ------------------------------------------------------------------
    print(f"Writing bbox ZIP → {bbox_path}")
    with ZipFile(bbox_path, "w", compression=ZIP_DEFLATED) as zf:
        for name in tqdm(names, desc="BBox"):
            arr = read_mask(raw[name])
            bbox_arr = mask_to_bbox(arr)
            zf.writestr(name, encode_mask(bbox_arr))

    # ------------------------------------------------------------------
    # 3. Intra-class shuffle (skipped for flat ZIPs without class folders)
    # ------------------------------------------------------------------
    if not has_classes:
        print("Skipping intra-class shuffle: ZIP has no class subfolders.")
    else:
        print(f"Writing intra-class shuffle ZIP → {intra_path}")
        intra_map: dict[str, str] = {}
        for cls, cls_names in by_class.items():
            shuffled = cls_names.copy()
            rng.shuffle(shuffled)
            for orig, src in zip(cls_names, shuffled):
                intra_map[orig] = src

        with ZipFile(intra_path, "w", compression=ZIP_DEFLATED) as zf:
            for name in tqdm(names, desc="Intra-shuffle"):
                zf.writestr(name, raw[intra_map[name]])

    # ------------------------------------------------------------------
    # 4. Inter-class shuffle (all masks)
    # ------------------------------------------------------------------
    print(f"Writing inter-class shuffle ZIP → {inter_path}")
    all_names = names.copy()
    shuffled_all = names.copy()
    rng.shuffle(shuffled_all)
    inter_map: dict[str, str] = {
        orig: src for orig, src in zip(all_names, shuffled_all)
    }

    with ZipFile(inter_path, "w", compression=ZIP_DEFLATED) as zf:
        for name in tqdm(names, desc="Inter-shuffle"):
            zf.writestr(name, raw[inter_map[name]])

    print("Done.")
    print(f"  {bbox_path}")
    if has_classes:
        print(f"  {intra_path}")
    print(f"  {inter_path}")


if __name__ == "__main__":
    main()
