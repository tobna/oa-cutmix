# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

from PIL import Image

from openmixup.utils import build_from_cfg
from openmixup.utils.logger import get_root_logger

from ..registry import PIPELINES

# Transforms that affect the mask (geometric transforms that must be applied identically)
MASK_AFFECTING_TRANSFORMS = {
    "CenterCrop",
    "CenterCropForEfficientNet",
    "CenterCrop_mmcls",
    "ElasticTransform",
    "FiveCrop",
    "LinearTransformation",
    "Pad",
    "Perspective",
    "PlaceCrop",
    "RandomAffine",
    "RandomCrop",
    "RandomCrop_mmcls",
    "RandomFlip",
    "RandomFlip_mmcls",
    "RandomHorizontalFlip",
    "RandomPerspective",
    "RandomPosterize",
    "RandomResizedCrop",
    "RandomResizedCropForEfficient",
    "RandomResizedCropPair",  # Custom class with apply_pair method
    "RandomResizedCropWithTwoCrop",
    "RandomResizedCrop_mmcls",
    "RandomRotation",
    "RandomVerticalFlip",
    "Resize",
    "ResizePair",
    "ResizeShare",
    "Resize_mmcls",
    "Rotate",
    "Shear",
    "TenCrop",
    "Translate",
}

ILLEGAL_TRANSFORMS = {"RandomResizedCrop", "Resize", "RandomPerspective", "RandomResizedCropWithTwoCrop"}


class BuildCompose(object):
    """Compose a data pipeline with a sequence of transforms.
    *** Modified torchvision Compose ***

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be callable or a dict, but got {type(transform)}")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string

    def apply_pair(self, img, mask):
        """Apply transforms to a pair of image and mask.

        For transforms that affect the mask (geometric), applies them identically to both.
        For color-only transforms, applies them only to the image.

        Args:
            img (PIL.Image): Input image (RGB).
            mask (PIL.Image): Input mask (grayscale).

        Returns:
            tuple: (transformed_img, transformed_mask)
        """
        from PIL import Image as PILImage

        logger = get_root_logger()

        img = img.convert("RGB")
        mask = mask.convert("L")

        for t in self.transforms:
            t_name = t.__class__.__name__

            old_img_size = img.size if isinstance(img, Image.Image) else img.shape[-2:]  # get size for pil and tensor
            old_mask_size = (
                mask.size if isinstance(mask, Image.Image) else mask.shape[-2:]
            )  # get size for pil and tensor

            # Check if transform has its own apply_pair method
            if hasattr(t, "apply_pair"):
                img, mask = t.apply_pair(img, mask)
            # Check if transform affects the mask (geometric transforms)
            elif t_name in ILLEGAL_TRANSFORMS:
                raise ValueError(
                    f"It is not possible to pass the image + mask through {t_name}. Use another augmentation. For"
                    f" example, use {t_name}Pair instead of {t_name}"
                )
            elif t_name in MASK_AFFECTING_TRANSFORMS:
                # Apply to both image and mask as RGBA
                img.putalpha(mask)
                img = t(img)
                r, g, b, a = img.split()
                img = PILImage.merge("RGB", (r, g, b))
                mask = a
            elif t_name == "ToTensor":  # Convert mask and image to tensor individually
                img = t(img)
                mask = t(mask)
            else:
                # Color-only transform - apply only to image
                img = t(img)

            if img is None:
                return None

            # Check if sizes match after transform
            img_size = img.size if isinstance(img, Image.Image) else img.shape[-2:]  # get size for pil and tensor
            mask_size = mask.size if isinstance(mask, Image.Image) else mask.shape[-2:]  # get size for pil and tensor
            if img_size != mask_size:
                logger.error(
                    f"Size mismatch after transform '{t_name}' | old img size: {old_img_size}, old mask size:"
                    f" {old_mask_size} | new img size: {img_size}, new mask size: {mask_size}, image type: {type(img)}"
                )

        return img, mask
