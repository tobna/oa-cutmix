#!/usr/bin/env python3
"""
Interactive Mask Editor for TinyImageNet masks.

This script:
1. Counts completely black and white masks in the ZIP file
3. Logs all actions to a JSON file

Usage:
    python scripts/mask_editor.py

Keyboard shortcuts:
    j/k or ←/→ : Previous/Next image
    1-9         : Select folder mask variant
    z           : Keep ZIP mask (no change)
    f           : Update ZIP with selected folder mask
    m           : Update ZIP with mean of folder masks
    u           : Update ZIP with union of folder masks
    b           : Replace ZIP with all-black mask
    w           : Replace ZIP with all-white mask
    q           : Quit and save
"""

import argparse
import io
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from typing import Any, Optional

import numpy as np
import curses
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich_pixels import Pixels
from tqdm import tqdm


def get_terminal_width() -> int:
    """Get terminal width for image resizing."""
    try:
        return curses.tigetnum('cols') or 80
    except:
        return 80


def resize_for_terminal(img: Image.Image, max_width: int = 80) -> Image.Image:
    """Resize image to fit terminal width while preserving aspect ratio."""
    if img.width <= max_width:
        return img
    
    ratio = max_width / img.width
    new_height = int(img.height * ratio)
    return img.resize((max_width, new_height), Image.Resampling.LANCZOS)


def check_terminal_image_support():
    """Check if terminal supports inline images. Exit with error if not."""
    import os
    import sys

    console = Console()

    if not console.is_terminal:
        print("Error: Not running in a terminal. Inline images require an interactive terminal.", file=sys.stderr)
        print("       Please run this script directly in a terminal, not piped or redirected.", file=sys.stderr)
        sys.exit(1)

    term_program = os.environ.get("TERM_PROGRAM", "")
    term = os.environ.get("TERM", "")

    supported_terminals = [
        "iTerm.app",
        "Apple_Terminal",
        "vscode",
        "JetBrains",
        "WindowsTerminal",
        "conhost",
        "Terminus-Sublime",
        "Terminus-Babel",
        "Hyper",
        "Alacritty",
        "Kitty",
        "WezTerm",
        "Foot",
        "Sway",
    ]

    supports_sixel = term in ("tmux-256color", "screen-256color", "xterm-256color")

    supported = term_program in supported_terminals or supports_sixel or "256color" in term

    if not supported:
        print("Error: Your terminal does not support inline images.", file=sys.stderr)
        print(f"       TERM_PROGRAM: {term_program}", file=sys.stderr)
        print(f"       TERM: {term}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Supported terminals for inline images:", file=sys.stderr)
        for t in supported_terminals:
            print(f"  - {t}", file=sys.stderr)
        print("  - Any terminal with 256-color support (TERM=*-256color)", file=sys.stderr)
        print("  - Terminals with Sixel support", file=sys.stderr)
        print("", file=sys.stderr)
        print("Please use a compatible terminal and try again.", file=sys.stderr)
        sys.exit(1)

    return console


# ============================================================================
# CONSTANTS
# ============================================================================

# Paths
IMAGE_ROOT = "/ds-sds/images/tiny-imagenet"
MASK_ZIP = "/fscratch/nauen/datasets/tiny-imagenet-masks/train_masks.zip"
MASK_DIR = "/fscratch/nauen/datasets/tiny-imagenet-masks/train/masks"
SPLIT = "train"

# Log file (next to ZIP)
LOG_FILE = MASK_ZIP.replace(".zip", "_edit_log.jsonl")

# Backup directory
BACKUP_DIR = MASK_ZIP.replace(".zip", "_backups")


# ============================================================================
# MASK ANALYSIS
# ============================================================================


def is_completely_black(arr: np.ndarray) -> bool:
    """Check if mask is completely black (all zeros)."""
    unique = np.unique(arr)
    return len(unique) == 1 and unique[0] == 0


def is_completely_white(arr: np.ndarray) -> bool:
    """Check if mask is completely white (all 255)."""
    unique = np.unique(arr)
    return len(unique) == 1 and unique[0] == 255


def get_mask_status(arr: np.ndarray) -> str:
    """Get mask status: 'black', 'white', or 'other'."""
    if is_completely_black(arr):
        return "black"
    elif is_completely_white(arr):
        return "white"
    else:
        return "other"


def analyze_zip_masks(zip_path: str, progress_callback=None) -> tuple[dict[str, dict[str, Any]], set]:
    """
    Analyze all masks in the ZIP file.

    Returns tuple: (results_dict, mask_paths_set)

    results_dict: {
        'class/image_id.png': {
            'status': 'black' | 'white' | 'other',
            'unique_values': [...],
            'size': (w, h)
        }
    }
    mask_paths_set: set of all mask paths in the ZIP
    """
    results = {}
    mask_paths_set = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        mask_files = [f for f in zf.namelist() if f.endswith(".png")]
        iterator = tqdm(mask_files, desc="  Analyzing masks", unit="mask")

        for i, mask_path in enumerate(iterator):
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
    """Count black, white, and other masks."""
    black = sum(1 for r in results.values() if r.get("status") == "black")
    white = sum(1 for r in results.values() if r.get("status") == "white")
    other = sum(1 for r in results.values() if r.get("status") == "other")
    return black, white, other


# ============================================================================
# IMAGE INDEX
# ============================================================================


def build_image_index(image_root: str, split: str, zip_mask_paths: set = None) -> list[dict]:
    """
    Build index of all images with their mask info.

    Args:
        image_root: Root directory containing class subdirectories
        split: Dataset split (train, val, test)
        zip_mask_paths: Set of mask paths that exist in the ZIP (for quick lookup)

    Returns sorted list: [
        {
            'class_name': 'n01443537',
            'image_name': 'n01443537_1029.JPEG',
            'image_path': '/ds-sds/...',
            'zip_mask_path': 'n01443537/n01443537_1029.png',
            'zip_mask_exists': True/False,
            'folder_masks': ['v0', 'v1', ...]
        }
    ]
    """
    split_root = os.path.join(image_root, split)
    images = []

    # First, get all class directories
    class_dirs = sorted([d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))])

    # Count total images for progress bar
    total_images = 0
    for class_name in class_dirs:
        class_dir = os.path.join(split_root, class_name)
        total_images += len(
            [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        )

    print(f"  Found {total_images} images in {len(class_dirs)} classes")

    iterator = tqdm(class_dirs, desc="  Indexing classes", unit="class")

    for class_name in iterator:
        class_dir = os.path.join(split_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Sort images for consistent ordering
        img_files = sorted(
            [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        )

        img_iterator = tqdm(img_files, desc=f"  {class_name}", unit="img", leave=False)

        for img_name in img_iterator:
            image_path = os.path.join(class_dir, img_name)
            img_id = os.path.splitext(img_name)[0]

            # ZIP mask path
            zip_mask_path = f"{class_name}/{img_id}.png"
            zip_mask_exists = zip_mask_paths is not None and zip_mask_path in zip_mask_paths

            # Folder masks
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


# Cache file path
INDEX_CACHE_FILE = MASK_ZIP.replace(".zip", "_index_cache.pkl")


def save_index_cache(images: list, zip_mask_paths: set):
    """Save the image index and ZIP mask paths to a cache file."""
    import pickle

    cache_data = {"images": images, "zip_mask_paths": zip_mask_paths, "timestamp": datetime.now().isoformat()}
    with open(INDEX_CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"  Index cache saved: {INDEX_CACHE_FILE}")


def load_index_cache() -> tuple[list, set] | None:
    """Load the image index and ZIP mask paths from cache file."""
    import pickle

    if not os.path.exists(INDEX_CACHE_FILE):
        return None
    try:
        with open(INDEX_CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)
        print(f"  Index cache loaded: {INDEX_CACHE_FILE}")
        print(f"    Images: {len(cache_data['images'])}")
        print(f"    Cached: {cache_data.get('timestamp', 'unknown')}")
        return cache_data["images"], cache_data["zip_mask_paths"]
    except Exception as e:
        print(f"  Warning: Failed to load cache: {e}")
        return None


def filter_problematic_images(images: list, zip_masks: dict) -> list:
    """
    Filter images to only those with problematic masks:
    - Black masks in ZIP
    - White masks in ZIP
    - No mask in ZIP
    """
    problematic = []

    for img in images:
        zip_exists = img.get("zip_mask_exists", False)

        if not zip_exists:
            # No mask in ZIP - problematic
            problematic.append(img)
        elif zip_exists:
            # Check if black or white
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

    # Check for non-versioned mask
    base_mask_path = os.path.join(mask_class_dir, f"{img_id}.JPEG")
    if os.path.exists(base_mask_path):
        masks.append({"version": "base", "path": base_mask_path, "filename": f"{img_id}.JPEG"})

    # Check for versioned masks
    for f in os.listdir(mask_class_dir):
        if f.startswith(f"{img_id}_v") and f.endswith(".JPEG"):
            try:
                version_str = f.split("_v")[1].split(".")[0]
                version_num = int(version_str)
                masks.append({"version": f"v{version_num}", "path": os.path.join(mask_class_dir, f), "filename": f})
            except (ValueError, IndexError):
                continue

    # Sort by version
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
    """Load a folder mask as numpy array."""
    try:
        mask = Image.open(path).convert("L")
        return np.array(mask)
    except Exception:
        return None


def compute_mean_mask(folder_masks: list[dict]) -> Optional[np.ndarray]:
    """Compute mean of all folder masks."""
    arrays = []
    for m in folder_masks:
        arr = load_folder_mask_as_array(m["path"])
        if arr is not None:
            arrays.append(arr)

    if not arrays:
        return None

    return np.mean(arrays, axis=0).astype(np.uint8)


def compute_union_mask(folder_masks: list[dict]) -> Optional[np.ndarray]:
    """Compute union (max) of all folder masks."""
    arrays = []
    for m in folder_masks:
        arr = load_folder_mask_as_array(m["path"])
        if arr is not None:
            arrays.append(arr.astype(np.float32) / 255.0)  # Normalize to 0-1

    if not arrays:
        return None

    union = np.maximum.reduce(arrays)
    # Threshold at 0.5
    union = (union > 0.5).astype(np.uint8) * 255
    return union


# ============================================================================
# ZIP MODIFICATION
# ============================================================================


def create_mask_backup(zip_path: str, backup_dir: str) -> str:
    """Create a backup of the ZIP file."""
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"train_masks_{timestamp}.zip")

    shutil.copy2(zip_path, backup_path)
    return backup_path


def create_manual_backup(zip_path: str, backup_dir: str, label: str = None, action_count: int = 0) -> str:
    """
    Create a manual named backup of the current ZIP state.

    Args:
        zip_path: Path to the ZIP file
        backup_dir: Directory to store backups
        label: Optional label (e.g., 'after_n01443537')
        action_count: Current number of actions performed

    Returns:
        Path to the created backup file
    """
    os.makedirs(backup_dir, exist_ok=True)

    if label:
        # Sanitize label for filename
        safe_label = label.replace("/", "_").replace(" ", "_")
        backup_path = os.path.join(backup_dir, f"train_masks_{safe_label}.zip")
    else:
        # Create numbered backup
        backup_path = os.path.join(backup_dir, f"train_masks_state_{action_count:04d}.zip")

    shutil.copy2(zip_path, backup_path)
    return backup_path


def update_mask_in_zip(zip_path: str, mask_path: str, new_mask_arr: np.ndarray) -> bool:
    """
    Update or add a single mask in the ZIP file.
    Uses atomic replacement: copy, modify, swap.

    Args:
        zip_path: Path to the ZIP file
        mask_path: Path inside the ZIP (e.g., 'class/image.png')
        new_mask_arr: Numpy array with the new mask

    Returns:
        True if successful, False otherwise
    """
    temp_path = None
    try:
        # Check if mask already exists in ZIP
        with zipfile.ZipFile(zip_path, "r") as zf_in:
            existing_files = set(zf_in.namelist())
            mask_exists = mask_path in existing_files

        # Create temporary file
        temp_path = zip_path + ".tmp"

        # Read original ZIP and write new one
        with zipfile.ZipFile(zip_path, "r") as zf_in:
            with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
                # Copy all existing entries
                for item in zf_in.infolist():
                    if item.filename == mask_path:
                        # Replace existing mask
                        new_mask_img = Image.fromarray(new_mask_arr, mode="L")
                        buf = io.BytesIO()
                        new_mask_img.save(buf, format="PNG")
                        buf.seek(0)
                        zf_out.writestr(item, buf.read())
                    else:
                        # Copy original
                        zf_out.writestr(item, zf_in.read(item.filename))

                # If mask doesn't exist, add it
                if not mask_exists:
                    new_mask_img = Image.fromarray(new_mask_arr, mode="L")
                    buf = io.BytesIO()
                    new_mask_img.save(buf, format="PNG")
                    buf.seek(0)

                    # Create a new ZipInfo entry
                    new_item = zipfile.ZipInfo(mask_path)
                    new_item.compress_type = zipfile.ZIP_DEFLATED
                    zf_out.writestr(new_item, buf.read())

        # Direct replacement (atomic on most filesystems)
        os.replace(temp_path, zip_path)

        return True
    except Exception as e:
        print(f"Error updating ZIP: {e}")
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return False


# ============================================================================
# LOGGING
# ============================================================================


def log_action(action: str, image_info: dict, old_status: str, new_status: str, folder_versions: list = None):
    """Log an action to the JSON file."""
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
# VISUALIZATION
# ============================================================================


class MaskVisualizer:
    """Interactive Rich terminal visualization for masks."""

    def __init__(self, images: list, zip_masks: dict, on_keypress=None):
        self.images = images
        self.zip_masks = zip_masks
        self.current_idx = 0
        self.selected_folder_idx = 0
        self.on_keypress = on_keypress
        self.action_count = 0
        self.running = True

        self.console = Console()
        self.console.print("[bold green]Initializing Mask Editor...[/bold green]")

        self.draw_current()
        self.run_event_loop()

    def get_contextual_shortcuts(self) -> str:
        """Generate contextual shortcuts based on current image state."""
        img_info = self.images[self.current_idx]
        has_zip = img_info.get("zip_mask_exists", False)
        folder_masks = img_info["folder_masks"]
        has_folder = len(folder_masks) > 0

        parts = []

        parts.append("j/k:Nav")
        parts.append("left/right:Nav")

        if has_zip:
            parts.append("z:Keep")
        else:
            parts.append("-:Keep")

        if has_folder:
            parts.append("f:Folder")
            parts.append("m:Mean")
            parts.append("u:Union")
        else:
            parts.append("-:Folder")
            parts.append("-:Mean")
            parts.append("-:Union")

        parts.append("b:Black")
        parts.append("w:White")
        parts.append("s:Save")
        parts.append("q:Quit")

        return " | ".join(parts)

    def run_event_loop(self):
        """Run the keyboard event loop using curses."""
        def inner_loop(stdscr):
            curses.curs_set(0)
            stdscr.nodelay(True)
            stdscr.timeout(100)
            
            while self.running:
                try:
                    key = stdscr.getch()
                    if key == -1:
                        continue
                    
                    if key == ord('j') or key == curses.KEY_DOWN:
                        self.handle_key('j')
                    elif key == ord('k') or key == curses.KEY_UP:
                        self.handle_key('k')
                    elif key == ord('q'):
                        self.handle_key('q')
                    elif key == ord('z'):
                        self.handle_key('z')
                    elif key == ord('f'):
                        self.handle_key('f')
                    elif key == ord('m'):
                        self.handle_key('m')
                    elif key == ord('u'):
                        self.handle_key('u')
                    elif key == ord('b'):
                        self.handle_key('b')
                    elif key == ord('w'):
                        self.handle_key('w')
                    elif key == ord('s'):
                        self.handle_key('s')
                    elif key in [ord(str(i)) for i in range(1, 10)]:
                        self.handle_key(chr(key))
                        
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    
        curses.wrapper(inner_loop)

    def handle_key(self, key: str):
        """Handle key press events."""
        if key == "right" or key == "k":
            self.next_image()
        elif key == "left" or key == "j":
            self.prev_image()
        elif key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            idx = int(key) - 1
            if self.images[self.current_idx]["folder_masks"]:
                if idx < len(self.images[self.current_idx]["folder_masks"]):
                    self.selected_folder_idx = idx
                    self.draw_current()
        elif key == "z":
            self.keep_zip()
        elif key == "f":
            self.update_with_folder_mask()
        elif key == "m":
            self.update_with_mean()
        elif key == "u":
            self.update_with_union()
        elif key == "b":
            self.replace_with_black()
        elif key == "w":
            self.replace_with_white()
        elif key == "s":
            self.save_backup()
        elif key == "q":
            self.quit()

        if self.on_keypress:
            self.on_keypress(key, self.images[self.current_idx])

    def next_image(self):
        """Go to next image."""
        self.current_idx = (self.current_idx + 1) % len(self.images)
        self.selected_folder_idx = 0
        self.draw_current()

    def prev_image(self):
        """Go to previous image."""
        self.current_idx = (self.current_idx - 1) % len(self.images)
        self.selected_folder_idx = 0
        self.draw_current()

    def save_temp_image(self, img: Image.Image, prefix: str) -> str:
        """Save image to temp file for Rich to display."""
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=prefix)
        img.save(temp_file.name)
        return temp_file.name

    def draw_current(self):
        """Draw current image and masks using Rich with rich-pixels."""
        self.console.clear()
        img_info = self.images[self.current_idx]

        term_width = get_terminal_width()
        img = resize_for_terminal(Image.open(img_info["image_path"]).convert("RGB"), term_width)

        zip_mask = None
        zip_status = "not_in_zip"
        zip_exists = img_info.get("zip_mask_exists", False)

        if zip_exists:
            try:
                with zipfile.ZipFile(MASK_ZIP, "r") as zf:
                    mask_data = zf.read(img_info["zip_mask_path"])
                    zip_mask = Image.open(io.BytesIO(mask_data)).convert("L")
                    zip_mask = resize_for_terminal(zip_mask, term_width)
                    zip_status = get_mask_status(np.array(zip_mask))
            except:
                zip_status = "error"
        else:
            zip_status = "not_in_zip"

        folder_masks = img_info["folder_masks"]
        folder_mask_imgs = []
        for m in folder_masks:
            arr = load_folder_mask_as_array(m["path"])
            if arr is not None:
                resized = resize_for_terminal(Image.fromarray(arr, mode="L"), term_width)
                folder_mask_imgs.append(resized)

        mean_mask = compute_mean_mask(folder_masks)
        if mean_mask is not None:
            mean_mask = resize_for_terminal(Image.fromarray(mean_mask), term_width)

        self.console.print(
            Panel(
                f"[bold cyan]{img_info['image_name']}[/bold cyan] | "
                f"[yellow]ZIP: {zip_status}[/yellow] | "
                f"[magenta]Folder: {len(folder_masks)}[/magenta] | "
                f"[green]Selected: {self.selected_folder_idx + 1 if folder_masks else 'N/A'}[/green]",
                title=f"[{self.current_idx + 1}/{len(self.images)}] {img_info['class_name']}",
                border_style="blue",
            )
        )

        self.console.print()
        self.console.print("[bold]Original Image[/bold]")
        self.console.print(Pixels.from_image(img))
        self.console.print()

        self.console.print(
            Panel(
                f"[bold]ZIP Mask[/bold]\n({zip_status})", border_style="yellow" if zip_mask else "red", padding=(0, 1)
            )
        )
        if zip_mask:
            self.console.print(Pixels.from_image(zip_mask))
            self.console.print()
        else:
            self.console.print("[dim](No ZIP mask)[/dim]")
            self.console.print()

        if folder_mask_imgs:
            self.console.print(
                Panel(
                    f"[bold]Folder Mask[/bold]\n({folder_masks[0]['version']})", border_style="magenta", padding=(0, 1)
                )
            )
            self.console.print(Pixels.from_image(folder_mask_imgs[0]))
            self.console.print()
        else:
            self.console.print(Panel(f"[bold]Folder Masks[/bold]\n(NONE)", border_style="red", padding=(0, 1)))
            self.console.print()

        if len(folder_mask_imgs) > 1:
            self.console.print(
                Panel(
                    f"[bold]Folder Mask 2[/bold]\n({folder_masks[1]['version']})",
                    border_style="magenta",
                    padding=(0, 1),
                )
            )
            self.console.print(Pixels.from_image(folder_mask_imgs[1]))
            self.console.print()

        if mean_mask is not None:
            self.console.print(Panel(f"[bold]Mean Mask[/bold]", border_style="cyan", padding=(0, 1)))
            self.console.print(Pixels.from_image(mean_mask))
            self.console.print()

        shortcuts = self.get_contextual_shortcuts()
        self.console.print(Panel(f"[bold]{shortcuts}[/bold]", border_style="blue", padding=(0, 2)))

    def keep_zip(self):
        """Keep ZIP mask (log action)."""
        img_info = self.images[self.current_idx]

        if img_info.get("zip_mask_exists", False):
            zip_status = self.zip_masks.get(img_info["zip_mask_path"], {}).get("status", "unknown")
        else:
            zip_status = "not_in_zip"

        log_action("keep_zip", img_info, zip_status, zip_status)

        if zip_status == "not_in_zip":
            self.console.print(
                f"[yellow][{self.current_idx + 1}] No ZIP mask to keep: {img_info['image_name']}[/yellow]"
            )
        else:
            self.console.print(f"[green][{self.current_idx + 1}] Kept ZIP mask: {img_info['image_name']}[/green]")

    def update_with_folder_mask(self):
        """Update ZIP with selected folder mask (or add if not in ZIP)."""
        img_info = self.images[self.current_idx]
        folder_masks = img_info["folder_masks"]

        if not folder_masks:
            self.console.print("[red]No folder masks available[/red]")
            return

        if self.selected_folder_idx >= len(folder_masks):
            self.selected_folder_idx = 0

        selected = folder_masks[self.selected_folder_idx]
        new_mask_arr = load_folder_mask_as_array(selected["path"])

        if new_mask_arr is None:
            self.console.print("[red]Failed to load folder mask[/red]")
            return

        if img_info.get("zip_mask_exists", False):
            old_status = self.zip_masks.get(img_info["zip_mask_path"], {}).get("status", "unknown")
        else:
            old_status = "not_in_zip"

        new_status = get_mask_status(new_mask_arr)
        action_type = "add_with_folder_mask" if old_status == "not_in_zip" else "update_with_folder_mask"

        if update_mask_in_zip(MASK_ZIP, img_info["zip_mask_path"], new_mask_arr):
            log_action(action_type, img_info, old_status, new_status, [selected["version"]])
            self.zip_masks[img_info["zip_mask_path"]] = {"status": new_status}
            img_info["zip_mask_exists"] = True
            self.action_count += 1

            action_verb = "Added" if old_status == "not_in_zip" else "Updated"
            self.console.print(
                f"[green][{self.current_idx + 1}] {action_verb} ZIP with folder mask: {img_info['image_name']}"
                f" ({selected['version']})[/green]"
            )
            self.draw_current()

    def update_with_mean(self):
        """Update ZIP with mean of folder masks (or add if not in ZIP)."""
        img_info = self.images[self.current_idx]
        folder_masks = img_info["folder_masks"]

        if not folder_masks:
            self.console.print("[red]No folder masks available[/red]")
            return

        mean_mask = compute_mean_mask(folder_masks)
        if mean_mask is None:
            self.console.print("[red]Failed to compute mean mask[/red]")
            return

        if img_info.get("zip_mask_exists", False):
            old_status = self.zip_masks.get(img_info["zip_mask_path"], {}).get("status", "unknown")
        else:
            old_status = "not_in_zip"

        new_status = get_mask_status(mean_mask)
        action_type = "add_with_mean_folder_mask" if old_status == "not_in_zip" else "update_with_mean_folder_mask"

        if update_mask_in_zip(MASK_ZIP, img_info["zip_mask_path"], mean_mask):
            versions = [m["version"] for m in folder_masks]
            log_action(action_type, img_info, old_status, new_status, versions)
            self.zip_masks[img_info["zip_mask_path"]] = {"status": new_status}
            img_info["zip_mask_exists"] = True
            self.action_count += 1

            action_verb = "Added" if old_status == "not_in_zip" else "Updated"
            self.console.print(
                f"[green][{self.current_idx + 1}] {action_verb} ZIP with mean mask: {img_info['image_name']}[/green]"
            )
            self.draw_current()

    def update_with_union(self):
        """Update ZIP with union of folder masks (or add if not in ZIP)."""
        img_info = self.images[self.current_idx]
        folder_masks = img_info["folder_masks"]

        if not folder_masks:
            self.console.print("[red]No folder masks available[/red]")
            return

        union_mask = compute_union_mask(folder_masks)
        if union_mask is None:
            self.console.print("[red]Failed to compute union mask[/red]")
            return

        if img_info.get("zip_mask_exists", False):
            old_status = self.zip_masks.get(img_info["zip_mask_path"], {}).get("status", "unknown")
        else:
            old_status = "not_in_zip"

        new_status = get_mask_status(union_mask)
        action_type = "add_with_union_folder_mask" if old_status == "not_in_zip" else "update_with_union_folder_mask"

        if update_mask_in_zip(MASK_ZIP, img_info["zip_mask_path"], union_mask):
            versions = [m["version"] for m in folder_masks]
            log_action(action_type, img_info, old_status, new_status, versions)
            self.zip_masks[img_info["zip_mask_path"]] = {"status": new_status}
            img_info["zip_mask_exists"] = True
            self.action_count += 1

            action_verb = "Added" if old_status == "not_in_zip" else "Updated"
            self.console.print(
                f"[green][{self.current_idx + 1}] {action_verb} ZIP with union mask: {img_info['image_name']}[/green]"
            )
            self.draw_current()

    def replace_with_black(self):
        """Replace ZIP mask with all-black (or add if not in ZIP)."""
        img_info = self.images[self.current_idx]

        img = Image.open(img_info["image_path"])
        new_mask_arr = np.zeros((img.height, img.width), dtype=np.uint8)

        if img_info.get("zip_mask_exists", False):
            old_status = self.zip_masks.get(img_info["zip_mask_path"], {}).get("status", "unknown")
        else:
            old_status = "not_in_zip"

        action_type = "add_with_black" if old_status == "not_in_zip" else "replace_with_black"

        if update_mask_in_zip(MASK_ZIP, img_info["zip_mask_path"], new_mask_arr):
            log_action(action_type, img_info, old_status, "black")
            self.zip_masks[img_info["zip_mask_path"]] = {"status": "black"}
            img_info["zip_mask_exists"] = True
            self.action_count += 1

            action_verb = "Added" if old_status == "not_in_zip" else "Replaced"
            self.console.print(
                f"[green][{self.current_idx + 1}] {action_verb} with black mask: {img_info['image_name']}[/green]"
            )
            self.draw_current()

    def replace_with_white(self):
        """Replace ZIP mask with all-white (or add if not in ZIP)."""
        img_info = self.images[self.current_idx]

        img = Image.open(img_info["image_path"])
        new_mask_arr = np.full((img.height, img.width), 255, dtype=np.uint8)

        if img_info.get("zip_mask_exists", False):
            old_status = self.zip_masks.get(img_info["zip_mask_path"], {}).get("status", "unknown")
        else:
            old_status = "not_in_zip"

        action_type = "add_with_white" if old_status == "not_in_zip" else "replace_with_white"

        if update_mask_in_zip(MASK_ZIP, img_info["zip_mask_path"], new_mask_arr):
            log_action(action_type, img_info, old_status, "white")
            self.zip_masks[img_info["zip_mask_path"]] = {"status": "white"}
            img_info["zip_mask_exists"] = True
            self.action_count += 1

            action_verb = "Added" if old_status == "not_in_zip" else "Replaced"
            self.console.print(
                f"[green][{self.current_idx + 1}] {action_verb} with white mask: {img_info['image_name']}[/green]"
            )
            self.draw_current()

    def save_backup(self):
        """Create a manual backup of the current ZIP state."""
        img_info = self.images[self.current_idx]
        label = f"after_{img_info['class_name']}"

        backup_path = create_manual_backup(MASK_ZIP, BACKUP_DIR, label=label, action_count=self.action_count)
        self.console.print(f"[green][{self.current_idx + 1}] Backup saved: {backup_path}[/green]")

    def quit(self):
        """Quit the application."""
        self.running = False
        self.console.print("[bold red]Quitting...[/bold red]")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Interactive Mask Editor for TinyImageNet")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of the image index cache")
    parser.add_argument(
        "--filter",
        choices=["all", "problematic"],
        default="problematic",
        help="Filter images: all or problematic (black/white/no zip mask, default)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Mask Editor - TinyImageNet")
    print("=" * 60)

    # Check terminal supports inline images
    console = check_terminal_image_support()

    # Verify paths
    if not os.path.exists(MASK_ZIP):
        print(f"ERROR: ZIP file not found: {MASK_ZIP}")
        sys.exit(1)

    if not os.path.exists(IMAGE_ROOT):
        print(f"ERROR: Image root not found: {IMAGE_ROOT}")
        sys.exit(1)

    # Create backup
    print("\n[Step 1/4] Creating backup...")
    backup_path = create_mask_backup(MASK_ZIP, BACKUP_DIR)
    print(f"  Backup created: {backup_path}")

    # Analyze ZIP masks
    print("\n[Step 2/4] Analyzing ZIP masks...")

    zip_masks, zip_mask_paths = analyze_zip_masks(MASK_ZIP)
    black, white, other = count_mask_statuses(zip_masks)
    print(f"\n  Results:")
    print(f"    Black masks:  {black}")
    print(f"    White masks:  {white}")
    print(f"    Other masks:  {other}")
    print(f"    Total in ZIP: {len(zip_masks)}")

    # Build or load image index (including images with NO mask in ZIP)
    print("\n[Step 3/4] Building image index...")

    # Try to load from cache first (unless --rebuild-index is set)
    cache_result = None if args.rebuild_index else load_index_cache()
    if cache_result is not None:
        images, cached_zip_paths = cache_result
        print(f"  Loaded {len(images)} images from cache")
    else:
        # Build from scratch
        images = build_image_index(IMAGE_ROOT, SPLIT, zip_mask_paths)
        print(f"  Built {len(images)} images")

        # Save to cache for next time
        save_index_cache(images, zip_mask_paths)

    # Count statistics
    with_zip_mask = sum(1 for img in images if img["zip_mask_exists"])
    without_zip_mask = sum(1 for img in images if not img["zip_mask_exists"])
    with_folder = sum(1 for img in images if img["folder_masks"])

    print(f"  Images with ZIP mask: {with_zip_mask}")
    print(f"  Images WITHOUT ZIP mask: {without_zip_mask}")
    print(f"  Images with folder masks: {with_folder}")

    # Filter to problematic images if requested
    if args.filter == "problematic":
        print(f"\n  Filtering to problematic images...")
        original_count = len(images)
        images = filter_problematic_images(images, zip_masks)
        print(f"  Filtered: {original_count} -> {len(images)} images")

    # Initialize log file
    print(f"\n  Log file: {LOG_FILE}")

    # Launch editor
    print("\n[Step 4/4] Starting editor...")
    print("-" * 60)
    print("Controls:")
    print("  j/k or ←/→ : Previous/Next image")
    print("  1-9        : Select folder mask variant")
    print("  z          : Keep ZIP mask")
    print("  f/m/u      : Update ZIP with folder/mean/union mask")
    print("  b/w        : Replace with black/white mask")
    print("  s          : Save manual backup")
    print("  q          : Quit")
    print("-" * 60)

    print("\nDone! Starting editor...")

    # Launch the Rich TUI
    viz = MaskVisualizer(images, zip_masks)


if __name__ == "__main__":
    main()
