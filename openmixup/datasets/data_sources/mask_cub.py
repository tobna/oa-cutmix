import io
import os
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import get_worker_info

from openmixup.utils.logger import get_root_logger

from ..registry import DATASOURCES


@DATASOURCES.register_module
class MaskCUB2011(object):
    """CUB-200-2011 dataset with masks from ZIP file.

    Args:
        root (str): Root directory containing 'CUB_200_2011' folder.
        mask_zip (str): Path to ZIP file containing masks.
        split (str): Dataset split in ['train', 'test'].
        return_label (bool): Whether to return class labels. Default: True.
        mask_suffix (str): Suffix for mask files. Default: '.png'.
    """

    BASE_FOLDER = "CUB_200_2011"

    def __init__(self, root, mask_zip, split="train", return_label=True, mask_suffix=".png", force_resize=False):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.root = Path(root) / self.BASE_FOLDER
        self.mask_zip = mask_zip
        self.split = split
        self.return_label = return_label
        self.mask_suffix = mask_suffix
        self.force_resize = force_resize
        self._zip_files = {}

        if not self.root.exists():
            raise ValueError(f"Data root not found: {self.root}")

        paths = pd.read_csv(self.root / "images.txt", sep=" ", names=["id", "path"])
        labels = pd.read_csv(self.root / "image_class_labels.txt", sep=" ", names=["id", "label"])
        splits = pd.read_csv(self.root / "train_test_split.txt", sep=" ", names=["id", "is_training"])
        data = paths.merge(labels, on="id")
        data = data.merge(splits, on="id")
        classes_file = self.root / "classes.txt"
        if classes_file.exists():
            classes_df = pd.read_csv(classes_file, sep=" ", names=["id", "name"])
            self.CLASSES = classes_df["name"].tolist()
        else:
            self.CLASSES = list(range(len(labels["label"].unique())))

        if split == "train":
            self.data = data[data.is_training == 1].reset_index(drop=True)
        else:
            self.data = data[data.is_training == 0].reset_index(drop=True)
        self.labels = (self.data["label"] - 1).tolist()

        if len(self.data) == 0:
            raise ValueError(f"No images found for split: {split}")

        self.logger = get_root_logger()

        if self.mask_zip is None:
            self.logger.warning("MaskCUB2011: mask_zip is None, all masks will be white (foreground=all)")
        elif not os.path.exists(self.mask_zip):
            raise ValueError(f"Mask ZIP file not found: {self.mask_zip}")

        self.logger.info(f"Initialize MaskCUB2011: root={self.root}, masks={self.mask_zip}, split={split}")

    def _get_worker_id(self):
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    def _get_zip_file(self):
        if not self.mask_zip or not os.path.exists(self.mask_zip):
            return None
        worker_id = self._get_worker_id()
        if worker_id not in self._zip_files:
            self._zip_files[worker_id] = zipfile.ZipFile(self.mask_zip, "r")
        return self._zip_files[worker_id]

    def _load_mask_from_zip(self, mask_path):
        zf = self._get_zip_file()
        if zf is None:
            return None
        try:
            mask_data = zf.read(mask_path)
            mask = Image.open(io.BytesIO(mask_data))
            return mask.convert("L")
        except Exception as e:
            self.logger.warning(f"Could not load mask {mask_path}: {e}")
            return None

    def get_length(self):
        return len(self.data)

    def get_sample(self, idx):
        sample = self.data.iloc[idx]
        path = self.root / "images" / sample.path
        label = sample.label - 1

        img = Image.open(path).convert("RGB")

        stem = os.path.splitext(sample.path)[0]
        mask_path = stem + self.mask_suffix
        mask = self._load_mask_from_zip(mask_path)

        if mask is None:
            mask = Image.new("L", img.size, 255)

        if self.force_resize:
            mask = mask.resize(img.size, Image.NEAREST)

        if img.size != mask.size:
            self.logger.error(f"Size mismatch at idx {idx}: img={img.size}, mask={mask.size}, path={path}")

        if self.return_label:
            return img, mask, label
        return img, mask

    def close(self):
        for zf in self._zip_files.values():
            zf.close()
        self._zip_files = {}

    def __del__(self):
        self.close()
