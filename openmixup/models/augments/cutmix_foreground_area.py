import sys

import numpy as np
import torch
from torch.nn import functional as F

from openmixup.models.utils.visualization import plot_image_mask_pairs


def _no_repeat_shuffle_idx(batch_size_this, ignore_failure=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_shuffle = torch.randperm(batch_size_this, device=device)
    idx_original = torch.tensor([i for i in range(batch_size_this)], device=device)
    idx_repeat = False
    for i in range(10):
        if (idx_original == idx_shuffle).any():
            idx_repeat = True
            idx_shuffle = torch.randperm(batch_size_this, device=device)
        else:
            idx_repeat = False
            break
    if idx_repeat and not ignore_failure:
        idx_shift = np.random.randint(1, batch_size_this - 1)
        idx_shuffle = torch.tensor([(i + idx_shift) % batch_size_this for i in range(batch_size_this)], device=device)
    return idx_shuffle


def _convert_binary_mask(mask):
    """Convert binary mask from (0, 255) to (0, 1) float."""
    if mask.max() > 1.0:
        mask = mask / 255.0
    return mask


def _rand_bbox(size, lam):
    """Generate random box by lam."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def _compute_foreground_area(mask, bbx1, bby1, bbx2, bby2):
    """Compute foreground object area within a bounding box region.

    Args:
        mask: Binary mask tensor of shape (N, 1, H, W) with values in [0, 1]
        bbx1, bby1, bbx2, bby2: Bounding box coordinates

    Returns:
        foreground_area: Tensor of shape (N,) with foreground pixel counts
    """
    if mask.shape[1] != 1:
        mask = mask.unsqueeze(1)

    return mask[:, :, bbx1:bbx2, bby1:bby2].sum(dim=(1, 2, 3))


def _compute_total_foreground_area(mask):
    """Compute total foreground object area in the entire image.

    Args:
        mask: Binary mask tensor of shape (N, 1, H, W) with values in [0, 1]

    Returns:
        total_foreground_area: Tensor of shape (N,) with total foreground pixel counts
    """
    if mask.shape[1] != 1:
        mask = mask.unsqueeze(1)

    return mask.sum(dim=(1, 2, 3))


def gaussian_blur(tensor, kernel_size=11, sigma=1.0):
    """
    Apply Gaussian blur on a single channel tensor of shape (bs, h, w).

    Args:
        tensor:      (bs, h, w)
        kernel_size: int, must be odd
        sigma:       float, standard deviation of Gaussian

    Returns:
        blurred tensor of shape (bs, h, w)
    """

    # --- 1. Build 1D Gaussian kernel ---
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss_1d = torch.exp(-(x**2) / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    # --- 2. Build 2D Gaussian kernel via outer product ---
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]  # (k, k)

    # --- 3. Reshape to conv2d weight format: (out_ch, in_ch, kH, kW) ---
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    kernel = kernel.to(tensor.device)

    # --- 4. Add channel dim: (bs, h, w) -> (bs, 1, h, w) ---
    tensor = tensor.unsqueeze(1)

    # --- 5. Pad to preserve spatial dimensions ---
    pad = kernel_size // 2
    tensor = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")

    # --- 6. Apply convolution ---
    blurred = F.conv2d(tensor, kernel)  # (bs, 1, h, w)

    return blurred.squeeze(1)  # (bs, h, w)


@torch.no_grad()
def cutmix_foreground_area(
    img,
    gt_label,
    mask=None,
    alpha=1.0,
    lam=None,
    fg_weight=1.0,
    dist_mode=False,
    eps=1e-7,
    _debug_visualize=False,
    mode="absolute",
    mask_blur_sigma=0.0,
    **kwargs,
):
    r"""CutMix with foreground object area-based label weighting.

    Uses the visible foreground object area in each CutMix region to compute
    the mixing weight for labels, rather than using box area.

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        mask (Tensor): Object masks of shape (N, 1, H, W) with values in [0, 1]
            or binary (0, 255). If None, falls back to standard CutMix.
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        fg_weight (float): Weight for foreground-area-based label computation.
            1.0 = use only foreground-based weights (default).
            0.0 = use only default lambda-based weights.
            Values in (0, 1) interpolate between the two.
        dist_mode (bool): Whether to do cross gpus index shuffling.

    Returns:
        img: Mixed images of shape (N, C, H, W) (mask not included in output)
        gt_label: Tuple of (y_a, y_b, weight_a, weight_b) where weights are
            based on foreground object area
    """
    if mask is None:
        raise ValueError(
            "cutmix_foreground_area requires a valid mask tensor, but got None. "
            "Ensure your dataset returns masks and they are passed to the mixup function."
        )

    mask = _convert_binary_mask(mask)

    if mask.shape[1] != 1:
        mask = mask.unsqueeze(1)

    if mask.shape[2] != img.shape[2] or mask.shape[3] != img.shape[3]:
        mask = torch.nn.functional.interpolate(mask, size=(img.shape[2], img.shape[3]), mode="nearest")

    if mask_blur_sigma > 0.0:
        k = int(6 * mask_blur_sigma + 1) | 1
        mask = gaussian_blur(mask, k, mask_blur_sigma)

    # ============================================
    # DEBUG: Visualize input before CutMix
    # ============================================
    if _debug_visualize:
        print("[DEBUG] Visualizing input images and masks...")
        plot_image_mask_pairs(
            img, mask, titles=[f"Input {i}" for i in range(img.size(0))], show=False, save_path="debug_input.png"
        )

    if lam is None:
        lam = np.random.beta(alpha, alpha)

    if not dist_mode:
        rand_index = _no_repeat_shuffle_idx(img.size(0), ignore_failure=True)
        if len(img.size()) == 4:
            img_ = img[rand_index]
            mask_ = mask[rand_index]
        else:
            raise ValueError("cutmix_foreground_area does not support 5D input")

        y_a = gt_label
        y_b = gt_label[rand_index]

        _, _, h, w = img.shape
        bbx1, bby1, bbx2, bby2 = _rand_bbox(img.size(), lam)

        img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]

        # Compute foreground areas
        fg_area_a_inside = _compute_foreground_area(mask, bbx1, bby1, bbx2, bby2)
        fg_area_b_inside = _compute_foreground_area(mask_, bbx1, bby1, bbx2, bby2)

        # Total foreground area
        total_fg_a = _compute_total_foreground_area(mask)
        total_fg_b = _compute_total_foreground_area(mask_)

        if mode == "relative":
            mode = 0.0
        elif mode == "absolute":
            mode = 1.0
        else:
            assert (
                isinstance(mode, float) and 0.0 <= mode <= 1.0
            ), f"Only modes allowed are 'absolute', 'relative' or a float between 0 and 1, but got {mode}."

        if mode == 1.0:
            # Pure absolute: skip relative computations
            fg_visible_a = total_fg_a - fg_area_a_inside
            fg_visible_b = fg_area_b_inside
            total_fg_visible = (fg_visible_a + fg_visible_b).clamp(min=eps)
            fg_weight_a = (fg_visible_a / total_fg_visible).unsqueeze(1).clamp(0.0, 1.0)
            no_fg_visible = (fg_visible_a + fg_visible_b) <= eps
            fg_weight_a = torch.where(no_fg_visible.unsqueeze(1), torch.ones_like(fg_weight_a) * 0.5, fg_weight_a)
        elif mode == 0.0:
            # Pure relative: skip absolute computations
            rel_fg_area_a_inside = fg_area_a_inside / (total_fg_a + eps)
            rel_fg_area_b_inside = fg_area_b_inside / (total_fg_b + eps)
            rel_fg_vis_a = 1.0 - rel_fg_area_a_inside
            rel_fg_vis_b = rel_fg_area_b_inside
            rel_total_fg_vis = (rel_fg_vis_a + rel_fg_vis_b).clamp(min=eps)
            fg_weight_a = (rel_fg_vis_a / rel_total_fg_vis).unsqueeze(1).clamp(0.0, 1.0)
        else:
            # Blended: compute both branches
            fg_visible_a = total_fg_a - fg_area_a_inside
            fg_visible_b = fg_area_b_inside
            total_fg_visible = (fg_visible_a + fg_visible_b).clamp(min=eps)
            abs_weight_a = (fg_visible_a / total_fg_visible).unsqueeze(1).clamp(0.0, 1.0)
            no_fg_visible = (fg_visible_a + fg_visible_b) <= eps
            abs_weight_a = torch.where(no_fg_visible.unsqueeze(1), torch.ones_like(abs_weight_a) * 0.5, abs_weight_a)
            rel_fg_area_a_inside = fg_area_a_inside / (total_fg_a + eps)
            rel_fg_area_b_inside = fg_area_b_inside / (total_fg_b + eps)
            rel_fg_vis_a = 1.0 - rel_fg_area_a_inside
            rel_fg_vis_b = rel_fg_area_b_inside
            rel_total_fg_vis = (rel_fg_vis_a + rel_fg_vis_b).clamp(min=eps)
            rel_weight_a = (rel_fg_vis_a / rel_total_fg_vis).unsqueeze(1).clamp(0.0, 1.0)
            fg_weight_a = mode * abs_weight_a + (1 - mode) * rel_weight_a

        # Default lambda-based weights
        lam = 1.0 - (bbx2 - bbx1) * (bby2 - bby1) / (w * h)

        if fg_weight == 1.0:
            weight_a = fg_weight_a
        else:
            # Blend between foreground-based and lambda-based weights
            weight_a = fg_weight * fg_weight_a + (1.0 - fg_weight) * lam

        # ============================================
        # DEBUG: Visualize output after CutMix
        # ============================================
        if _debug_visualize:
            print("[DEBUG] Visualizing output images and masks after CutMix...")
            mask[:, :, bbx1:bbx2, bby1:bby2] = mask_[:, :, bbx1:bbx2, bby1:bby2]
            plot_image_mask_pairs(
                img,
                mask,
                titles=[
                    f"Output {i} ({weight_a[i].item():.2f} / {1- weight_a[i].item():.2f} lambda:{lam}/{1-lam})"
                    for i in range(img.size(0))
                ],
                show=False,
                save_path="debug_output.png",
            )
            print("[DEBUG] Exiting after debug visualization.")
            sys.exit(0)

        return img, (y_a, y_b, weight_a)

    else:
        raise ValueError("cutmix_foreground_area does not support dist_mode yet")
