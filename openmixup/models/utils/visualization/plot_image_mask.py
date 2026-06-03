# Copyright (c) OpenMMLab. All rights reserved.
"""Image-Mask visualization utilities for debugging.

This module provides functions to visualize image-mask pairs by:
1. Unnormalizing the images
2. Plotting each pair as: image only, mask only, masked image (mask as alpha)
"""

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch

# Default ImageNet normalization values
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default CIFAR-10 normalization values
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.201)


def unnormalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD, to_rgb=True, inplace=False):
    """Unnormalize a tensor image or batch of images.

    Args:
        tensor (torch.Tensor or np.ndarray): Image tensor of shape
            (C, H, W) or (B, C, H, W) in range [0, 1] or normalized.
        mean (tuple): Mean values for each channel. Defaults to ImageNet.
        std (tuple): Std values for each channel. Defaults to ImageNet.
        to_rgb (bool): Whether to convert from BGR to RGB. Defaults to True.
        inplace (bool): Whether to modify the tensor in place. Defaults to False.

    Returns:
        np.ndarray: Unnormalized image(s) in range [0, 255] as uint8.
    """
    # Convert mean and std to numpy arrays (required by mmcv.imdenormalize)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    print(f"[DEBUG unnormalize_tensor] mean={mean}, std={std}, to_rgb={to_rgb}")

    # Helper function to denormalize a single image
    def _denorm_single(img):
        print(f"[DEBUG unnormalize_tensor] BEFORE: min={img.min():.4f}, max={img.max():.4f}, shape={img.shape}")

        # Check if image is already in [0, 1] range (not normalized)
        if img.min() >= 0 and img.max() <= 1:
            print(f"[DEBUG unnormalize_tensor] Image appears to be in [0, 1] range, scaling to [0, 255]")
            img = (img * 255).astype(np.uint8)
        else:
            # Assume image is normalized with mean/std, apply denormalization
            img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb)
            print(f"[DEBUG unnormalize_tensor] After imdenormalize (before uint8): min={img.min():.4f}, max={img.max():.4f}")
            # Scale from [0, 1] to [0, 255] before converting to uint8
            img = (img * 255).astype(np.uint8)

        print(f"[DEBUG unnormalize_tensor] AFTER: min={img.min()}, max={img.max()}, dtype={img.dtype}")
        return np.ascontiguousarray(img)

    if isinstance(tensor, torch.Tensor):
        # Move to CPU and convert to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        # Handle batch dimension
        if tensor.dim() == 4:
            # Batch of images: (B, C, H, W)
            imgs = []
            for i in range(tensor.size(0)):
                img = tensor[i].numpy().transpose(1, 2, 0)
                imgs.append(_denorm_single(img))
            return imgs
        elif tensor.dim() == 3:
            # Single image: (C, H, W)
            img = tensor.numpy().transpose(1, 2, 0)
            return _denorm_single(img)
        else:
            raise ValueError(f"Expected tensor of shape (C, H, W) or (B, C, H, W), got {tensor.shape}")
    elif isinstance(tensor, np.ndarray):
        if tensor.ndim == 4:
            # Batch of images: (B, H, W, C) or (B, C, H, W)
            imgs = []
            for i in range(tensor.shape[0]):
                if tensor.shape[1] == 3 or tensor.shape[1] == 1:
                    # (B, C, H, W) format
                    img = tensor[i].transpose(1, 2, 0)
                else:
                    # (B, H, W, C) format
                    img = tensor[i]
                imgs.append(_denorm_single(img))
            return imgs
        elif tensor.ndim == 3:
            # Single image: (C, H, W) or (H, W, C)
            if tensor.shape[0] == 3 or tensor.shape[0] == 1:
                # (C, H, W) format
                img = tensor.transpose(1, 2, 0)
            else:
                # (H, W, C) format
                img = tensor
            return _denorm_single(img)
        else:
            raise ValueError(
                f"Expected array of shape (C, H, W), (H, W, C), (B, C, H, W), or (B, H, W, C), got {tensor.shape}"
            )
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


def unnormalize_mask(mask):
    """Unnormalize a mask tensor to [0, 255] uint8.

    Args:
        mask (torch.Tensor or np.ndarray): Mask of shape (H, W), (1, H, W),
            (B, H, W), or (B, 1, H, W). Values can be in [0, 1] or [0, 255].

    Returns:
        np.ndarray: Unnormalized mask in range [0, 255] as uint8.
    """
    if isinstance(mask, torch.Tensor):
        if mask.is_cuda:
            mask = mask.cpu()
        mask = mask.numpy()

    # Check if mask is in [0, 1] range (including edge case of floats)
    is_normalized = mask.max() <= 1.0 and mask.min() >= 0.0

    # Handle dimensions
    if mask.ndim == 4:
        # (B, 1, H, W) or (B, H, W)
        if mask.shape[1] == 1:
            mask = mask.squeeze(1)
        masks = []
        for i in range(mask.shape[0]):
            m = mask[i]
            if is_normalized:
                m = (m * 255).astype(np.uint8)
            else:
                m = m.astype(np.uint8)
            masks.append(np.ascontiguousarray(m))
        return masks
    elif mask.ndim == 3:
        # (1, H, W) or (C, H, W) or (H, W)
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.shape[0] > 1 and mask.shape[0] <= 3:
            # Assume this is actually an image, not a mask
            raise ValueError(f"Mask appears to be an image with shape {mask.shape}")
        if is_normalized:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        return np.ascontiguousarray(mask)
    elif mask.ndim == 2:
        # (H, W)
        if is_normalized:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        return np.ascontiguousarray(mask)
    else:
        raise ValueError(f"Expected mask of shape (H, W), (1, H, W), (B, H, W), or (B, 1, H, W), got {mask.shape}")


def create_masked_image(img, mask):
    """Create a masked image with mask as alpha channel.

    Args:
        img (np.ndarray): Image of shape (H, W, 3) in range [0, 255] as uint8.
        mask (np.ndarray): Mask of shape (H, W) in range [0, 255] as uint8.

    Returns:
        np.ndarray: RGBA image of shape (H, W, 4) with mask as alpha channel.
    """
    # Ensure mask is the right shape
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    # Normalize mask to [0, 1] for alpha
    alpha = mask.astype(np.float32) / 255.0

    # Create RGBA image
    rgba = np.concatenate([img, alpha], axis=2)

    return rgba


def plot_image_mask_pairs(
    images, masks, mean=IMAGENET_MEAN, std=IMAGENET_STD, titles=None, show=False, save_path=None, figsize_per_pair=15
):
    """Plot image-mask pairs with three views per pair:
    image only, mask only, masked image.

    Args:
        images (torch.Tensor or np.ndarray): Batch of images of shape
            (B, C, H, W) or (B, H, W, C). Values can be normalized or in [0, 1].
        masks (torch.Tensor or np.ndarray): Batch of masks of shape
            (B, 1, H, W), (B, H, W), (1, H, W), or (H, W).
        mean (tuple): Mean values for denormalization. Defaults to ImageNet.
        std (tuple): Std values for denormalization. Defaults to ImageNet.
        titles (list or None): List of titles for each pair. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
        save_path (str or None): Path to save the figure. Defaults to None.
        figsize_per_pair (int): Base figure width per image pair.
            Defaults to 15.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    # Unnormalize images
    imgs = unnormalize_tensor(images, mean=mean, std=std)

    # Unnormalize masks
    masks_list = unnormalize_mask(masks)

    # Ensure we have lists
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    if isinstance(masks_list, np.ndarray):
        masks_list = [masks_list]

    num_pairs = len(imgs)
    assert (
        len(masks_list) == num_pairs
    ), f"Number of images ({num_pairs}) must match number of masks ({len(masks_list)})"

    # Create figure
    fig, axes = plt.subplots(num_pairs, 3, figsize=(3 * figsize_per_pair, num_pairs * figsize_per_pair / 2))

    # Handle single image case
    if num_pairs == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_pairs):
        img = imgs[i]
        mask = masks_list[i]

        # Image only
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(titles[i] + "\nImage" if titles else f"Image {i+1}")
        axes[i, 0].axis("off")

        # Mask only
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(titles[i] + "\nMask" if titles else f"Mask {i+1}")
        axes[i, 1].axis("off")

        # Masked image (apply mask directly to RGB channels)
        # Foreground (mask=255) shows image, background (mask=0) shows black
        alpha = (mask / 255.0).astype(np.float32)  # (H, W)
        masked_img = (img * alpha[..., np.newaxis]).astype(np.uint8)  # Apply mask to RGB
        axes[i, 2].imshow(masked_img)
        axes[i, 2].set_title(titles[i] + "\nMasked" if titles else f"Masked {i+1}")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig


def plot_single_image_mask_pair(image, mask, mean=IMAGENET_MEAN, std=IMAGENET_STD, title="", show=True, save_path=None):
    """Plot a single image-mask pair with three views.

    Args:
        image (torch.Tensor or np.ndarray): Image of shape (C, H, W), (H, W, C),
            or single image in batch format (1, C, H, W).
        mask (torch.Tensor or np.ndarray): Mask of shape (1, H, W), (H, W),
            or single mask in batch format (1, 1, H, W).
        mean (tuple): Mean values for denormalization. Defaults to ImageNet.
        std (tuple): Std values for denormalization. Defaults to ImageNet.
        title (str): Title for the image. Defaults to ''.
        show (bool): Whether to display the plot. Defaults to True.
        save_path (str or None): Path to save the figure. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    # Handle single images (add batch dimension if needed)
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

    if isinstance(mask, torch.Tensor):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3 and mask.shape[0] != 1:
            mask = mask.unsqueeze(0)
    elif isinstance(mask, np.ndarray):
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=(0, 1))
        elif mask.ndim == 3 and mask.shape[0] != 1:
            mask = np.expand_dims(mask, axis=0)

    titles = [title] if title else None

    return plot_image_mask_pairs(
        images=image, masks=mask, mean=mean, std=std, titles=titles, show=show, save_path=save_path
    )


# Convenience function with shorter name
vis_image_mask = plot_image_mask_pairs
vis_single_pair = plot_single_image_mask_pair
