import io
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import get_worker_info

from openmixup.utils.logger import get_root_logger

from ..registry import DATASOURCES


@DATASOURCES.register_module
class MaskImageFolder(object):
    """Folder-based image dataset with masks from ZIP file.

    This data source reads images from a directory and corresponding masks
    from a ZIP file. The masks are never extracted to disk.

    Expected image directory structure:
        root/
            train/  (or val/)
                <class_name1>/
                    img1.jpg
                    img2.jpg
                <class_name2>/
                    img3.jpg

    Masks are expected in a ZIP file with parallel structure:
        <class_name1>/
            img1.png
            img2.png
        <class_name2>/
            img3.png

    Args:
        root (str): Root directory containing class subdirectories.
        mask_zip (str): Path to ZIP file containing masks.
        split (str): Dataset split in ['train', 'val', 'test'].
        return_label (bool): Whether to return class labels. Default: True.
        mask_suffix (str): Suffix for mask files. Default: '.png'.
            For example, if image is 'img.jpg', mask would be 'img.png'.
    """

    CLASSES = None

    def __init__(
        self,
        root,
        mask_zip,
        split="train",
        return_label=True,
        mask_suffix=".png",
        classes_file=None,
        force_resize=False,
    ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.root = root
        self.mask_zip = mask_zip
        self.split = split
        self.return_label = return_label
        self.mask_suffix = mask_suffix
        self.force_resize = force_resize

        split_root = os.path.join(self.root, self.split)
        if not os.path.exists(split_root):
            raise ValueError(f"Data split directory not found: {split_root}")

        all_classes = sorted([p.name for p in Path(split_root).iterdir() if p.is_dir()])
        if classes_file is not None:
            with open(classes_file) as f:
                allowed = {l.strip() for l in f if l.strip()}
            self.CLASSES = [c for c in all_classes if c in allowed]
            if len(self.CLASSES) == 0:
                raise ValueError(f"No matching classes found for classes_file: {classes_file}")
        else:
            self.CLASSES = all_classes
        self.fns = []
        self.labels = []

        for label, cls_name in enumerate(self.CLASSES):
            cls_dir = Path(split_root) / cls_name
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
                    self.fns.append(str(img_file))
                    self.labels.append(label)

        if len(self.fns) == 0:
            raise ValueError(f"No images found in {split_root}")

        # Validate that the zip file exists (but don't open it yet)
        if self.mask_zip and not os.path.exists(self.mask_zip):
            raise ValueError(f"Mask ZIP file not found: {self.mask_zip}")

        # Dict of zipfile objects indexed by worker_id - each worker gets its own
        self._zip_files = {}
        self.logger = get_root_logger()
        self.logger.info(f"Initialize mask image folder: folder={self.root}/{self.split} masks={self.mask_zip}")

    def _get_worker_id(self):
        """Get the current worker id."""
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    def _get_mask_path(self, img_path):
        """Get the mask path inside the ZIP file."""
        img_name = os.path.basename(img_path)
        img_name_without_ext = os.path.splitext(img_name)[0]
        cls_name = os.path.basename(os.path.dirname(img_path))

        mask_name = img_name_without_ext + self.mask_suffix
        mask_path = os.path.join(cls_name, mask_name)

        return mask_path

    def _get_zip_file(self):
        """Get or create the zip file for the current worker."""
        if not self.mask_zip or not os.path.exists(self.mask_zip):
            return None

        worker_id = self._get_worker_id()

        # If we don't have a zip file for this worker, create one
        if worker_id not in self._zip_files:
            self._zip_files[worker_id] = zipfile.ZipFile(self.mask_zip, "r")

        return self._zip_files[worker_id]

    def _load_mask_from_zip(self, mask_path):
        """Load a mask from the ZIP file."""
        zf = self._get_zip_file()
        if zf is None:
            self.logger.error(f"Could not open zipfile {self.mask_zip}")
            return None

        try:
            mask_data = zf.read(mask_path)
            mask = Image.open(io.BytesIO(mask_data))
            return mask.convert("L")
        except Exception as e:
            self.logger.warning(f"Could not load mask {mask_path}: {e}")
            return None

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx]).convert("RGB")

        mask_path = self._get_mask_path(self.fns[idx])
        mask = self._load_mask_from_zip(mask_path)

        if mask is None:
            mask = Image.new("L", img.size, 255)
        # else:
        #     mask_arr = np.array(mask)
        #     if np.max(mask_arr) == 0 or np.mean(mask_arr) <= 0.01:
        #         mask = Image.new("L", img.size, 255)
        #         if np.max(mask_arr) == 0:
        #             logger.error(f"Got full zero mask on index {idx}: {self.fns[idx]}")
        #         else:
        #             logger.error(
        #                 f"Got almost full zero mask on index {idx}: mean={np.mean(mask_arr)} path={self.fns[idx]}"
        #             )

        if self.force_resize:
            mask = mask.resize(img.size, Image.NEAREST)

        # Check if sizes match
        if img.size != mask.size:
            self.logger.error(f"Size mismatch at loading: img={img.size}, mask={mask.size}, path={self.fns[idx]}")

        if self.return_label:
            return img, mask, self.labels[idx]
        else:
            return img, mask

    def close(self):
        """Close all ZIP files."""
        for zf in self._zip_files.values():
            zf.close()
        self._zip_files = {}

    def __del__(self):
        self.close()
