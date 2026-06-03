#!/usr/bin/env python3
"""
Batch script to fix fully black/white masks in the ZIP file.

For each problematic mask (black or white):
- If folder masks exist: replace with union of folder masks
- If no folder masks: replace with fully white mask

Optimized for network mounts: loads all masks into memory, writes ZIP once.

Usage:
    python scripts/fix_masks_batch.py [--dry-run] [--backup]
"""

import argparse
import io
import os
import shutil
import zipfile
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================================
# CONSTANTS
# ============================================================================

IMAGE_ROOT = "/ds-sds/images/tiny-imagenet"
MASK_ZIP = "/fscratch/nauen/datasets/tiny-imagenet-masks/train_masks.zip"
MASK_DIR = "/fscratch/nauen/datasets/tiny-imagenet-masks/train/masks"
SPLIT = "train"

BACKUP_DIR = MASK_ZIP.replace(".zip", "_backups")
LOG_FILE = MASK_ZIP.replace(".zip", "_batch_fix_log.jsonl")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def is_completely_black(arr: np.ndarray) -> bool:
    unique = np.unique(arr)
    return len(unique) == 1 and unique[0] == 0


def is_completely_white(arr: np.ndarray) -> bool:
    unique = np.unique(arr)
    return len(unique) == 1 and unique[0] == 255


def get_mask_status(arr: np.ndarray) -> str:
    if is_completely_black(arr):
        return "black"
    elif is_completely_white(arr):
        return "white"
    else:
        return "other"


def get_folder_masks(class_name: str, img_id: str) -> list:
    """Get available folder masks for an image."""
    mask_class_dir = os.path.join(MASK_DIR, class_name)
    if not os.path.exists(mask_class_dir):
        return []

    masks = []

    base_mask_path = os.path.join(mask_class_dir, f"{img_id}.JPEG")
    if os.path.exists(base_mask_path):
        masks.append({"version": "base", "path": base_mask_path})

    for f in os.listdir(mask_class_dir):
        if f.startswith(f"{img_id}_v") and f.endswith(".JPEG"):
            try:
                version_str = f.split("_v")[1].split(".")[0]
                version_num = int(version_str)
                masks.append({"version": f"v{version_num}", "path": os.path.join(mask_class_dir, f)})
            except (ValueError, IndexError):
                continue

    return masks


def load_folder_mask_as_array(path: str):
    try:
        mask = Image.open(path).convert("L")
        return np.array(mask)
    except Exception:
        return None


def compute_union_mask(folder_masks: list) -> np.ndarray:
    """Compute union (max) of all folder masks."""
    arrays = []
    for m in folder_masks:
        arr = load_folder_mask_as_array(m["path"])
        if arr is not None:
            arrays.append(arr.astype(np.float32) / 255.0)

    if not arrays:
        return None

    union = np.maximum.reduce(arrays)
    union = (union > 0.5).astype(np.uint8) * 255
    return union


def log_action(action: str, image_path: str, old_status: str, new_status: str, details: str = ""):
    import json
    entry = {
        "timestamp": datetime.now().isoformat(),
        "image_path": image_path,
        "action": action,
        "old_status": old_status,
        "new_status": new_status,
        "details": details,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Batch fix black/white masks in ZIP")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backup before modifying (default: True)")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup")
    args = parser.parse_args()

    print("=" * 60)
    print("Batch Mask Fixer - TinyImageNet (Optimized)")
    print("=" * 60)

    if not os.path.exists(MASK_ZIP):
        print(f"ERROR: ZIP file not found: {MASK_ZIP}")
        return

    if not os.path.exists(MASK_DIR):
        print(f"ERROR: MASK directory not found: {MASK_DIR}")
        return

    # Create backup
    if args.backup and not args.no_backup:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"train_masks_{timestamp}.zip")
        shutil.copy2(MASK_ZIP, backup_path)
        print(f"Backup created: {backup_path}")

    print("\n[Step 1/4] Loading all masks from ZIP into memory...")

    # Load ALL masks into memory at once (much faster for network mounts)
    all_masks = {}
    mask_files = []

    with zipfile.ZipFile(MASK_ZIP, "r") as zf:
        mask_files = [f for f in zf.namelist() if f.endswith(".png")]
        
        for mask_path in tqdm(mask_files, desc="Loading masks", unit="mask"):
            try:
                mask_data = zf.read(mask_path)
                mask = Image.open(io.BytesIO(mask_data)).convert("L")
                mask_arr = np.array(mask)
                all_masks[mask_path] = {
                    "arr": mask_arr,
                    "status": get_mask_status(mask_arr),
                    "size": mask.size,
                }
            except Exception as e:
                print(f"Error reading {mask_path}: {e}")
                all_masks[mask_path] = {"arr": None, "status": "error", "size": None}

    print(f"  Loaded {len(all_masks)} masks into memory")

    print("\n[Step 2/4] Identifying problematic masks...")

    problematic_masks = []
    for mask_path, info in all_masks.items():
        if info["status"] in ("black", "white"):
            problematic_masks.append(mask_path)

    black_count = sum(1 for p in problematic_masks if all_masks[p]["status"] == "black")
    white_count = sum(1 for p in problematic_masks if all_masks[p]["status"] == "white")

    print(f"  Found {len(problematic_masks)} problematic masks:")
    print(f"    Black: {black_count}")
    print(f"    White: {white_count}")

    if args.dry_run:
        print("\n[DRY RUN] No changes will be made.")
        print("\n[Step 3/4] Processing (dry run)...")
        for mask_path in tqdm(problematic_masks, desc="Checking", unit="mask"):
            old_status = all_masks[mask_path]["status"]
            width, height = all_masks[mask_path]["size"]
            
            parts = mask_path.split("/")
            if len(parts) != 2:
                continue
            class_name = parts[0]
            filename = parts[1]
            img_id = os.path.splitext(filename)[0]
            
            folder_masks = get_folder_masks(class_name, img_id)
            
            if folder_masks:
                action = "would use union"
            else:
                action = "would use white"
            
            log_action(action, mask_path, old_status, "other", f"dry-run: {action}")
        
        print("\n[Step 4/4] Summary")
        print(f"  Total problematic: {len(problematic_masks)}")
        print(f"  Would fix: {len(problematic_masks)}")
        print(f"\n[DRY RUN COMPLETE] Run without --dry-run to apply changes.")
        return

    print("\n[Step 3/4] Processing masks in memory...")

    # Process all masks in memory (no network writes yet!)
    fixed_count = 0
    for mask_path in tqdm(problematic_masks, desc="Processing", unit="mask"):
        old_status = all_masks[mask_path]["status"]
        width, height = all_masks[mask_path]["size"]

        parts = mask_path.split("/")
        if len(parts) != 2:
            continue

        class_name = parts[0]
        filename = parts[1]
        img_id = os.path.splitext(filename)[0]

        folder_masks = get_folder_masks(class_name, img_id)

        if folder_masks:
            new_mask_arr = compute_union_mask(folder_masks)
            if new_mask_arr is None:
                continue
            action = "replace_with_union"
            details = f"union of {len(folder_masks)} folder masks"
        else:
            new_mask_arr = np.full((height, width), 255, dtype=np.uint8)
            action = "replace_with_white"
            details = "no folder masks found"

        all_masks[mask_path]["arr"] = new_mask_arr
        all_masks[mask_path]["status"] = log_action(action, mask_path, old_status, "other", details)
        "other"
        fixed_count += 1

    print(f"  Processed {fixed_count} masks in memory")

    print("\n[Step 4/4] Writing new ZIP file (single write to network)...")

    # Write the entire new ZIP at once (one network write!)
    temp_path = MASK_ZIP + ".tmp"
    
    with zipfile.ZipFile(MASK_ZIP, "r") as zf_in:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
            # Copy all entries, replacing problematic ones
            for item in tqdm(zf_in.infolist(), desc="Writing ZIP", unit="entry"):
                mask_path = item.filename
                
                if mask_path in all_masks and all_masks[mask_path]["arr"] is not None:
                    # Replace with processed mask
                    new_mask_img = Image.fromarray(all_masks[mask_path]["arr"], mode="L")
                    buf = io.BytesIO()
                    new_mask_img.save(buf, format="PNG")
                    buf.seek(0)
                    zf_out.writestr(item, buf.read())
                else:
                    # Copy original
                    zf_out.writestr(item, zf_in.read(item.filename))

    # Atomic replace
    os.replace(temp_path, MASK_ZIP)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total masks: {len(all_masks)}")
    print(f"  Problematic: {len(problematic_masks)}")
    print(f"  Fixed: {fixed_count}")
    print(f"  Log file: {LOG_FILE}")
    print(f"  New ZIP written (single network write)")


if __name__ == "__main__":
    main()
