import io
import os
import zipfile

from PIL import Image
from torch.utils.data import get_worker_info
from torchvision.datasets import StanfordCars

from openmixup.utils.logger import get_root_logger

from ..registry import DATASOURCES


@DATASOURCES.register_module
class MaskCars(object):
    """Stanford Cars dataset with foreground masks from a ZIP file.

    Args:
        root (str): Root directory containing 'stanford_cars' folder.
        mask_zip (str): Path to ZIP file containing masks.
        split (str): Dataset split in ['train', 'test'].
        return_label (bool): Whether to return class labels. Default: True.
        mask_suffix (str): Suffix for mask files. Default: '.png'.
    """

    VALID_SPLITS = ["train", "test"]

    def __init__(
        self,
        root,
        mask_zip,
        split="train",
        return_label=True,
        mask_suffix=".png",
    ):
        assert split in self.VALID_SPLITS, f"Invalid split: {split}"
        self.return_label = return_label
        self.mask_zip = mask_zip
        self.mask_suffix = mask_suffix
        self._zip_files = {}

        self.dataset = StanfordCars(
            root=root,
            split=split,
            download=False,
        )
        self.CLASSES = self.dataset.classes
        self.labels = [s[1] for s in self.dataset._samples]

        self.logger = get_root_logger()

        if self.mask_zip is None:
            self.logger.warning(
                "MaskCars: mask_zip is None, all masks will be white "
                "(foreground=all)"
            )
        elif not os.path.exists(self.mask_zip):
            raise ValueError(f"Mask ZIP file not found: {self.mask_zip}")

        self.logger.info(
            f"Initialize MaskCars: root={root}, masks={self.mask_zip}, "
            f"split={split}"
        )

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

    def _load_mask_from_zip(self, mask_key):
        zf = self._get_zip_file()
        if zf is None:
            return None
        try:
            mask_data = zf.read(mask_key)
            mask = Image.open(io.BytesIO(mask_data))
            return mask.convert("L")
        except Exception as e:
            self.logger.warning(f"Could not load mask {mask_key}: {e}")
            return None

    def get_length(self):
        return len(self.dataset)

    def get_sample(self, idx):
        img_path, label = self.dataset._samples[idx]

        img = Image.open(img_path).convert("RGB")

        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_key = stem + self.mask_suffix
        mask = self._load_mask_from_zip(mask_key)

        if mask is None:
            mask = Image.new("L", img.size, 255)

        if img.size != mask.size:
            self.logger.error(
                f"Size mismatch at idx {idx}: img={img.size}, "
                f"mask={mask.size}, path={img_path}"
            )

        if self.return_label:
            return img, mask, label
        return img, mask

    def close(self):
        for zf in self._zip_files.values():
            zf.close()
        self._zip_files = {}

    def __del__(self):
        self.close()
