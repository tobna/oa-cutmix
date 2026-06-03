import io
import os
import zipfile

import numpy as np
from PIL import Image
from torch.utils.data import get_worker_info

from openmixup.utils.logger import get_root_logger

from ..registry import DATASOURCES
from .cifar import CIFAR100


@DATASOURCES.register_module
class MaskCIFAR100(CIFAR100):
    """CIFAR100 with masks from ZIP file.

    Args:
        root (str): Dataset root path containing 'cifar-100-python'.
        mask_zip (str): Path to ZIP file containing masks.
        split (str): Dataset split in ['train', 'test'].
        return_label (bool): Whether to return class labels. Default: True.
        *args, **kwargs: Passed to parent CIFAR100 class.
    """

    def __init__(self, root, *args, mask_zip, split="train", return_label=True, **kwargs):
        super().__init__(root, *args, split=split, return_label=True, **kwargs)
        self.mask_zip = mask_zip
        self._zip_files = {}
        self.return_label = return_label

        if not os.path.exists(mask_zip):
            raise ValueError(f"Mask ZIP file not found: {mask_zip}")

        logger = get_root_logger()
        logger.info(f"Initialize MaskCIFAR100: root={root}, masks={mask_zip}, split={split}")

    def _get_worker_id(self):
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    def _get_zip_file(self):
        worker_id = self._get_worker_id()
        if worker_id not in self._zip_files:
            self._zip_files[worker_id] = zipfile.ZipFile(self.mask_zip, "r")
        return self._zip_files[worker_id]

    def _load_mask_from_zip(self, mask_path):
        logger = get_root_logger()
        try:
            zf = self._get_zip_file()
            mask_data = zf.read(mask_path)
            mask = Image.open(io.BytesIO(mask_data))
            return mask.convert("L")
        except Exception as e:
            logger.warning(f"Could not load mask {mask_path}: {e}")
            return None

    def get_sample(self, idx):
        img, label = super().get_sample(idx)
        logger = get_root_logger()

        mask_path = f"{label}/{idx:06d}.png"
        mask = self._load_mask_from_zip(mask_path)

        if mask is None:
            mask = Image.new("L", img.size, 255)
        # else:
        #     mask_arr = np.array(mask)
        #     if np.max(mask_arr) == 0 or np.mean(mask_arr) <= 0.01:
        #         mask = Image.new("L", img.size, 255)
        #         if np.max(mask_arr) == 0:
        #             logger.error(f"Got full zero mask on index {idx}")
        #         else:
        #             logger.error(f"Got almost full zero mask on index {idx}: mean={np.mean(mask_arr)}")

        if img.size != mask.size:
            logger.error(f"Size mismatch at idx {idx}: img={img.size}, mask={mask.size}")

        if self.return_label:
            return img, mask, label
        else:
            return img, mask

    def close(self):
        for zf in self._zip_files.values():
            zf.close()
        self._zip_files = {}

    def __del__(self):
        self.close()
