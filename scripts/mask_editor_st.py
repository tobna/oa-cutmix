#!/usr/bin/env python3
"""
Interactive Mask Editor for TinyImageNet - Streamlit Web UI Version.

Usage:
    streamlit run scripts/mask_editor_st.py

Keyboard shortcuts:
    j/k or ←/→ : Previous/Next image
    1-9         : Select folder mask variant
    z           : Keep ZIP mask (no change)
    f           : Update ZIP with selected folder mask
    m           : Update ZIP with mean of folder masks
    u           : Update ZIP with union of folder masks
    b           : Replace ZIP with all-black mask
    w           : Replace ZIP with all-white mask
    s           : Save manual backup
"""

import argparse
import io
import json
import os
import shutil
import zipfile
from datetime import datetime
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

# ============================================================================
# CONSTANTS
# ============================================================================

IMAGE_ROOT = "/ds-sds/images/tiny-imagenet"
MASK_ZIP = "/fscratch/nauen/datasets/tiny-imagenet-masks/train_masks.zip"
MASK_DIR = "/fscratch/nauen/datasets/tiny-imagenet-masks/train/masks"
SPLIT = "train"

LOG_FILE = MASK_ZIP.replace(".zip", "_edit_log.jsonl")
BACKUP_DIR = MASK_ZIP.replace(".zip", "_backups")


# ============================================================================
# MASK ANALYSIS (keep from original)
# ============================================================================


def is_completely_black(arr: np.ndarray) -> bool:
    return np.max(arr) == 0


def is_completely_white(arr: np.ndarray) -> bool:
    return np.min(arr) == 255


def get_mask_status(arr: np.ndarray) -> str:
    if is_completely_black(arr):
        return "black"
    elif is_completely_white(arr):
        return "white"
    else:
        return "other"


def analyze_zip_masks(zip_path: str):
    """Analyze all masks in the ZIP file."""
    results = {}
    mask_paths_set = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        mask_files = [f for f in zf.namelist() if f.endswith(".png")]

        for mask_path in mask_files:
            mask_paths_set.add(mask_path)

            try:
                mask_data = zf.read(mask_path)
                mask = Image.open(io.BytesIO(mask_data)).convert("L")
                mask_arr = np.array(mask)

                results[mask_path] = {
                    "status": get_mask_status(mask_arr),
                    "unique_values": list(np.unique(mask_arr)),
                    "size": mask.size,
                }
            except Exception as e:
                results[mask_path] = {"status": "error", "error": str(e), "size": None}

    return results, mask_paths_set


def count_mask_statuses(results: dict) -> tuple[int, int, int]:
    black = sum(1 for r in results.values() if r.get("status") == "black")
    white = sum(1 for r in results.values() if r.get("status") == "white")
    other = sum(1 for r in results.values() if r.get("status") == "other")
    return black, white, other


# ============================================================================
# IMAGE INDEX (keep from original)
# ============================================================================


def build_image_index(image_root: str, split: str, zip_mask_paths: set = None) -> list[dict]:
    """Build index of all images with their mask info."""
    split_root = os.path.join(image_root, split)
    images = []

    class_dirs = sorted([d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))])

    for class_name in class_dirs:
        class_dir = os.path.join(split_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        img_files = sorted(
            [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        )

        for img_name in img_files:
            image_path = os.path.join(class_dir, img_name)
            img_id = os.path.splitext(img_name)[0]

            zip_mask_path = f"{class_name}/{img_id}.png"
            zip_mask_exists = zip_mask_paths is not None and zip_mask_path in zip_mask_paths

            folder_masks = get_folder_masks(class_name, img_id)

            images.append(
                {
                    "class_name": class_name,
                    "image_name": img_name,
                    "image_path": image_path,
                    "zip_mask_path": zip_mask_path,
                    "zip_mask_exists": zip_mask_exists,
                    "folder_masks": folder_masks,
                }
            )

    return images


INDEX_CACHE_FILE = MASK_ZIP.replace(".zip", "_index_cache.pkl")


def save_index_cache(images: list, zip_mask_paths: set):
    import pickle

    cache_data = {"images": images, "zip_mask_paths": zip_mask_paths, "timestamp": datetime.now().isoformat()}
    with open(INDEX_CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f)


def load_index_cache() -> tuple[list, set] | None:
    import pickle

    if not os.path.exists(INDEX_CACHE_FILE):
        return None
    try:
        with open(INDEX_CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)
        return cache_data["images"], cache_data["zip_mask_paths"]
    except Exception:
        return None


def filter_problematic_images(images: list, zip_masks: dict) -> list:
    """Filter images to only those with problematic masks."""
    problematic = []

    for img in images:
        zip_exists = img.get("zip_mask_exists", False)

        if not zip_exists:
            problematic.append(img)
        elif zip_exists:
            mask_path = img["zip_mask_path"]
            status = zip_masks.get(mask_path, {}).get("status", "unknown")
            if status in ["black", "white"]:
                problematic.append(img)

    return problematic


def get_folder_masks(class_name: str, img_id: str) -> list[dict]:
    """Get available folder masks for an image."""
    mask_class_dir = os.path.join(MASK_DIR, class_name)
    if not os.path.exists(mask_class_dir):
        return []

    masks = []

    base_mask_path = os.path.join(mask_class_dir, f"{img_id}.JPEG")
    if os.path.exists(base_mask_path):
        masks.append({"version": "base", "path": base_mask_path, "filename": f"{img_id}.JPEG"})

    for f in os.listdir(mask_class_dir):
        if f.startswith(f"{img_id}_v") and f.endswith(".JPEG"):
            try:
                version_str = f.split("_v")[1].split(".")[0]
                version_num = int(version_str)
                masks.append({"version": f"v{version_num}", "path": os.path.join(mask_class_dir, f), "filename": f})
            except (ValueError, IndexError):
                continue

    def sort_key(m):
        if m["version"] == "base":
            return -1
        try:
            return int(m["version"].replace("v", ""))
        except:
            return 0

    masks.sort(key=sort_key)
    return masks


def load_folder_mask_as_array(path: str) -> Optional[np.ndarray]:
    try:
        mask = Image.open(path).convert("L")
        return np.array(mask)
    except Exception:
        return None


def compute_mean_mask(folder_masks: list[dict]) -> Optional[np.ndarray]:
    arrays = []
    for m in folder_masks:
        arr = load_folder_mask_as_array(m["path"])
        if arr is not None:
            arrays.append(arr)

    if not arrays:
        return None

    return np.mean(arrays, axis=0).astype(np.uint8)


def compute_union_mask(folder_masks: list[dict]) -> Optional[np.ndarray]:
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


# ============================================================================
# ZIP MODIFICATION (keep from original)
# ============================================================================


def create_mask_backup(zip_path: str, backup_dir: str) -> str:
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"train_masks_{timestamp}.zip")
    shutil.copy2(zip_path, backup_path)
    return backup_path


def create_manual_backup(zip_path: str, backup_dir: str, label: str = None, action_count: int = 0) -> str:
    os.makedirs(backup_dir, exist_ok=True)

    if label:
        safe_label = label.replace("/", "_").replace(" ", "_")
        backup_path = os.path.join(backup_dir, f"train_masks_{safe_label}.zip")
    else:
        backup_path = os.path.join(backup_dir, f"train_masks_state_{action_count:04d}.zip")

    shutil.copy2(zip_path, backup_path)
    return backup_path


def update_mask_in_zip(zip_path: str, mask_path: str, new_mask_arr: np.ndarray) -> bool:
    temp_path = None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf_in:
            existing_files = set(zf_in.namelist())
            mask_exists = mask_path in existing_files

        temp_path = zip_path + ".tmp"

        with zipfile.ZipFile(zip_path, "r") as zf_in:
            with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
                for item in zf_in.infolist():
                    if item.filename == mask_path:
                        new_mask_img = Image.fromarray(new_mask_arr, mode="L")
                        buf = io.BytesIO()
                        new_mask_img.save(buf, format="PNG")
                        buf.seek(0)
                        zf_out.writestr(item, buf.read())
                    else:
                        zf_out.writestr(item, zf_in.read(item.filename))

                if not mask_exists:
                    new_mask_img = Image.fromarray(new_mask_arr, mode="L")
                    buf = io.BytesIO()
                    new_mask_img.save(buf, format="PNG")
                    buf.seek(0)
                    new_item = zipfile.ZipInfo(mask_path)
                    new_item.compress_type = zipfile.ZIP_DEFLATED
                    zf_out.writestr(new_item, buf.read())

        os.replace(temp_path, zip_path)
        return True
    except Exception as e:
        st.error(f"Error updating ZIP: {e}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return False


# ============================================================================
# LOGGING (keep from original)
# ============================================================================


def log_action(action: str, image_info: dict, old_status: str, new_status: str, folder_versions: list = None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "image_path": f"{image_info['class_name']}/{image_info['image_name']}",
        "action": action,
        "old_mask_status": old_status,
        "new_mask_status": new_status,
        "folder_mask_versions": folder_versions or [],
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# STREAMLIT UI
# ============================================================================


def main(args):
    st.set_page_config(
        page_title="Mask Editor - TinyImageNet",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎭 Mask Editor - TinyImageNet")

    # Check paths exist
    if not os.path.exists(args.mask_zip):
        st.error(f"ZIP file not found: {args.mask_zip}")
        return

    if not os.path.exists(args.image_root):
        st.error(f"Image root not found: {args.image_root}")
        return

    # Initialize session state
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "selected_folder_idx" not in st.session_state:
        st.session_state.selected_folder_idx = 0
    if "action_count" not in st.session_state:
        st.session_state.action_count = 0
    if "images" not in st.session_state:
        st.session_state.images = None
    if "zip_masks" not in st.session_state:
        st.session_state.zip_masks = None
    if "initialized" not in st.session_state:
        # Create backup
        st.info("Creating backup...")
        backup_path = create_mask_backup(args.mask_zip, args.backup_dir)
        st.success(f"Backup created: {backup_path}")

        # Analyze ZIP masks
        st.info("Analyzing ZIP masks...")
        zip_masks, zip_mask_paths = analyze_zip_masks(args.mask_zip)
        black, white, other = count_mask_statuses(zip_masks)
        st.session_state.zip_masks = zip_masks

        col1, col2, col3 = st.columns(3)
        col1.metric("Black Masks", black)
        col2.metric("White Masks", white)
        col3.metric("Other Masks", other)

        # Build or load index
        st.info("Building image index...")
        cache_result = load_index_cache()
        if cache_result is not None:
            images, cached_zip_paths = cache_result
            st.success(f"Loaded {len(images)} images from cache")
        else:
            images = build_image_index(args.image_root, args.split, zip_mask_paths)
            save_index_cache(images, zip_mask_paths)
            st.success(f"Built {len(images)} images")

        # Filter to problematic
        images = filter_problematic_images(images, zip_masks)
        st.success(f"Filtered to {len(images)} problematic images")

        st.session_state.images = images
        st.session_state.initialized = True
        st.rerun()

    if st.session_state.images is None or len(st.session_state.images) == 0:
        st.warning("No images to display!")
        return

    images = st.session_state.images
    zip_masks = st.session_state.zip_masks
    idx = st.session_state.current_idx
    selected_folder_idx = st.session_state.selected_folder_idx

    # Bounds check
    if idx < 0:
        st.session_state.current_idx = 0
        idx = 0
    elif idx >= len(images):
        st.session_state.current_idx = len(images) - 1
        idx = len(images) - 1

    img_info = images[idx]

    # Load images
    original_img = Image.open(img_info["image_path"]).convert("RGB")

    zip_mask = None
    zip_status = "not_in_zip"
    if img_info.get("zip_mask_exists", False):
        try:
            with zipfile.ZipFile(args.mask_zip, "r") as zf:
                mask_data = zf.read(img_info["zip_mask_path"])
                zip_mask = Image.open(io.BytesIO(mask_data)).convert("L")
                zip_status = get_mask_status(np.array(zip_mask))
        except:
            zip_status = "error"

    folder_masks = img_info["folder_masks"]
    selected_folder_mask = None
    if folder_masks and selected_folder_idx < len(folder_masks):
        selected_folder_mask_arr = load_folder_mask_as_array(folder_masks[selected_folder_idx]["path"])
        if selected_folder_mask_arr is not None:
            selected_folder_mask = Image.fromarray(selected_folder_mask_arr, mode="L")

    mean_mask = None
    mean_mask_img = None
    if folder_masks:
        mean_mask = compute_mean_mask(folder_masks)
        if mean_mask is not None:
            mean_mask_img = Image.fromarray(mean_mask, mode="L")

    # Display
    st.header(f"[{idx + 1}/{len(images)}] {img_info['image_name']}")
    st.caption(f"Class: {img_info['class_name']} | ZIP: {zip_status} | Folder masks: {len(folder_masks)}")

    # Image columns
    col_orig, col_zip, col_folder, col_mean = st.columns(4)

    with col_orig:
        st.image(original_img, caption="Original", use_container_width=True)

    with col_zip:
        if zip_mask:
            st.image(zip_mask, caption=f"ZIP ({zip_status})", use_container_width=True)
        else:
            st.warning("NOT IN ZIP")

    with col_folder:
        if selected_folder_mask:
            version = folder_masks[selected_folder_idx]["version"]
            st.image(selected_folder_mask, caption=f"Folder ({version})", use_container_width=True)
        else:
            st.warning("NO FOLDER MASK")

    with col_mean:
        if mean_mask_img:
            st.image(mean_mask_img, caption="Mean", use_container_width=True)
        else:
            st.warning("N/A")

    # Folder mask selector
    if folder_masks:
        st.subheader("Select Folder Mask")
        options = [f"{i+1}. {m['version']}" for i, m in enumerate(folder_masks)]
        selected = st.radio(
            "Folder mask variant:",
            options,
            index=selected_folder_idx,
            horizontal=True,
            key="folder_selector",
        )
        if selected:
            new_idx = int(selected.split(".")[0]) - 1
            if new_idx != selected_folder_idx:
                st.session_state.selected_folder_idx = new_idx
                st.rerun()

    # Action buttons
    st.subheader("Actions")

    col_nav1, col_nav2 = st.columns([1, 1])
    with col_nav1:
        if st.button("⬅️ Previous (j)", use_container_width=True):
            st.session_state.current_idx = max(0, idx - 1)
            st.session_state.selected_folder_idx = 0
            st.rerun()
    with col_nav2:
        if st.button("Next (k) ➡️", use_container_width=True):
            st.session_state.current_idx = min(len(images) - 1, idx + 1)
            st.session_state.selected_folder_idx = 0
            st.rerun()

    col_keep, col_folder, col_mean, col_union = st.columns(4)
    with col_keep:
        if st.button("Keep ZIP (z)", use_container_width=True, type="secondary"):
            log_action("keep_zip", img_info, zip_status, zip_status)
            st.toast(f"Kept ZIP mask: {img_info['image_name']}")

    with col_folder:
        if st.button("Use Folder (f)", use_container_width=True, type="primary"):
            if not folder_masks:
                st.error("No folder masks available")
            elif selected_folder_mask is None:
                st.error("Failed to load folder mask")
            else:
                old_status = zip_status
                new_status = get_mask_status(np.array(selected_folder_mask))
                action_type = "add_with_folder_mask" if old_status == "not_in_zip" else "update_with_folder_mask"

                if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], np.array(selected_folder_mask)):
                    log_action(
                        action_type, img_info, old_status, new_status, [folder_masks[selected_folder_idx]["version"]]
                    )
                    st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": new_status}
                    st.session_state.action_count += 1
                    st.toast(f"Updated ZIP with folder mask: {img_info['image_name']}")
                    st.rerun()

    with col_mean:
        if st.button("Use Mean (m)", use_container_width=True, type="primary"):
            if not folder_masks:
                st.error("No folder masks available")
            elif mean_mask is None:
                st.error("Failed to compute mean mask")
            else:
                old_status = zip_status
                new_status = get_mask_status(mean_mask)
                action_type = (
                    "add_with_mean_folder_mask" if old_status == "not_in_zip" else "update_with_mean_folder_mask"
                )

                if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], mean_mask):
                    versions = [m["version"] for m in folder_masks]
                    log_action(action_type, img_info, old_status, new_status, versions)
                    st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": new_status}
                    st.session_state.action_count += 1
                    st.toast(f"Updated ZIP with mean mask: {img_info['image_name']}")
                    st.rerun()

    with col_union:
        if st.button("Use Union (u)", use_container_width=True, type="primary"):
            if not folder_masks:
                st.error("No folder masks available")
            else:
                union_mask = compute_union_mask(folder_masks)
                if union_mask is None:
                    st.error("Failed to compute union mask")
                else:
                    old_status = zip_status
                    new_status = get_mask_status(union_mask)
                    action_type = (
                        "add_with_union_folder_mask" if old_status == "not_in_zip" else "update_with_union_folder_mask"
                    )

                    if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], union_mask):
                        versions = [m["version"] for m in folder_masks]
                        log_action(action_type, img_info, old_status, new_status, versions)
                        st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": new_status}
                        st.session_state.action_count += 1
                        st.toast(f"Updated ZIP with union mask: {img_info['image_name']}")
                        st.rerun()

    col_black, col_white, col_save = st.columns(3)
    with col_black:
        if st.button("Replace with Black (b)", use_container_width=True):
            old_status = zip_status
            black_mask = np.zeros((original_img.height, original_img.width), dtype=np.uint8)
            action_type = "add_with_black" if old_status == "not_in_zip" else "replace_with_black"

            if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], black_mask):
                log_action(action_type, img_info, old_status, "black")
                st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": "black"}
                st.session_state.action_count += 1
                st.toast(f"Replaced with black mask: {img_info['image_name']}")
                st.rerun()

    with col_white:
        if st.button("Replace with White (w)", use_container_width=True):
            old_status = zip_status
            white_mask = np.full((original_img.height, original_img.width), 255, dtype=np.uint8)
            action_type = "add_with_white" if old_status == "not_in_zip" else "replace_with_white"

            if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], white_mask):
                log_action(action_type, img_info, old_status, "white")
                st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": "white"}
                st.session_state.action_count += 1
                st.toast(f"Replaced with white mask: {img_info['image_name']}")
                st.rerun()

    with col_save:
        if st.button("Save Backup (s)", use_container_width=True):
            label = f"after_{img_info['class_name']}"
            backup_path = create_manual_backup(
                args.mask_zip, args.backup_dir, label=label, action_count=st.session_state.action_count
            )
            st.toast(f"Backup saved: {backup_path}")

    # Keyboard shortcuts using Streamlit's hotkey feature
    # This captures keyboard events and triggers actions
    def handle_key(key: str):
        images = st.session_state.images
        idx = st.session_state.current_idx

        if key in ["j", "arrowleft"]:
            st.session_state.current_idx = max(0, idx - 1)
            st.session_state.selected_folder_idx = 0
            st.rerun()
        elif key in ["k", "arrowright"]:
            st.session_state.current_idx = min(len(images) - 1, idx + 1)
            st.session_state.selected_folder_idx = 0
            st.rerun()
        elif key == "z":
            log_action("keep_zip", img_info, zip_status, zip_status)
            st.toast(f"Kept ZIP mask: {img_info['image_name']}")
        elif key == "f" and folder_masks and selected_folder_mask is not None:
            old_status = zip_status
            new_status = get_mask_status(np.array(selected_folder_mask))
            action_type = "add_with_folder_mask" if old_status == "not_in_zip" else "update_with_folder_mask"
            if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], np.array(selected_folder_mask)):
                log_action(
                    action_type, img_info, old_status, new_status, [folder_masks[selected_folder_idx]["version"]]
                )
                st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": new_status}
                st.session_state.action_count += 1
                st.toast(f"Updated ZIP with folder mask: {img_info['image_name']}")
                st.rerun()
        elif key == "m" and folder_masks and mean_mask is not None:
            old_status = zip_status
            new_status = get_mask_status(mean_mask)
            action_type = "add_with_mean_folder_mask" if old_status == "not_in_zip" else "update_with_mean_folder_mask"
            if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], mean_mask):
                versions = [m["version"] for m in folder_masks]
                log_action(action_type, img_info, old_status, new_status, versions)
                st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": new_status}
                st.session_state.action_count += 1
                st.toast(f"Updated ZIP with mean mask: {img_info['image_name']}")
                st.rerun()
        elif key == "b":
            old_status = zip_status
            black_mask = np.zeros((original_img.height, original_img.width), dtype=np.uint8)
            action_type = "add_with_black" if old_status == "not_in_zip" else "replace_with_black"
            if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], black_mask):
                log_action(action_type, img_info, old_status, "black")
                st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": "black"}
                st.session_state.action_count += 1
                st.toast(f"Replaced with black mask: {img_info['image_name']}")
                st.rerun()
        elif key == "w":
            old_status = zip_status
            white_mask = np.full((original_img.height, original_img.width), 255, dtype=np.uint8)
            action_type = "add_with_white" if old_status == "not_in_zip" else "replace_with_white"
            if update_mask_in_zip(args.mask_zip, img_info["zip_mask_path"], white_mask):
                log_action(action_type, img_info, old_status, "white")
                st.session_state.zip_masks[img_info["zip_mask_path"]] = {"status": "white"}
                st.session_state.action_count += 1
                st.toast(f"Replaced with white mask: {img_info['image_name']}")
                st.rerun()
        elif key == "s":
            label = f"after_{img_info['class_name']}"
            backup_path = create_manual_backup(
                args.mask_zip, args.backup_dir, label=label, action_count=st.session_state.action_count
            )
            st.toast(f"Backup saved: {backup_path}")
        elif key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            num = int(key) - 1
            if folder_masks and num < len(folder_masks):
                st.session_state.selected_folder_idx = num
                st.rerun()

    # Check for keyboard input using st.text_input (works in newer Streamlit)
    # This creates an invisible input that captures focus
    key_input = st.text_input(
        "Keyboard shortcuts (click here then press key)",
        key="key_capture",
        placeholder="j/k: nav, z: keep, f: folder, m: mean, b: black, w: white, s: save, 1-9: select",
        label_visibility="collapsed",
    )

    if key_input:
        handle_key(key_input)
        # Clear the input after handling
        st.session_state.key_capture = ""

    # Status bar
    st.divider()
    st.caption(f"Log file: {args.log_file} | Backups: {args.backup_dir} | Actions: {st.session_state.action_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default=IMAGE_ROOT)
    parser.add_argument("--mask-zip", default=MASK_ZIP)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--split", default=SPLIT)
    args = parser.parse_args()
    args.log_file = args.mask_zip.replace(".zip", "_edit_log.jsonl")
    args.backup_dir = args.mask_zip.replace(".zip", "_backups")

    main(args)
