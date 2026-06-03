import numpy as np
import torch
import torch.nn.functional as F

from .cutmix_foreground_area import _convert_binary_mask


@torch.no_grad()
def mask_attentivemix(
    img,
    gt_label,
    mask=None,
    alpha=1.0,
    lam=None,
    dist_mode=False,
    grid_size=32,
    top_k=6,
    mode="absolute",
    fg_weight=1.0,
    eps=1e-7,
    return_mask=False,
    **kwargs,
):
    """Mask-guided AttentiveMix using object masks instead of feature attention.

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
        gt_label (Tensor): Ground-truth labels (one-hot).
        mask (Tensor): Object masks of shape (N, 1, H, W) with values in [0, 1].
        alpha (float): Beta distribution parameter.
        lam (float): Mixing ratio. If None, sample from Beta(alpha, alpha).
        grid_size (int): Size of the coarse grid used to aggregate mask saliency.
        top_k (int): Number of grid cells to keep for the base image.
        mode (str | float): "absolute", "relative", or float to blend weights.
        fg_weight (float): Blend weight between foreground-aware and area-based lambda.
        eps (float): Numerics guard.
        return_mask (bool): Whether to return the binary mix mask along with the image.
    """

    if mask is None:
        raise ValueError("mask_attentivemix requires `mask` argument with object masks.")
    if dist_mode:
        raise ValueError("mask_attentivemix currently does not support distributed mode.")

    mask = _convert_binary_mask(mask)
    if mask.shape[1] != 1:
        mask = mask.unsqueeze(1)

    if mask.shape[2:] != img.shape[2:]:
        mask = F.interpolate(mask, size=img.shape[2:], mode="nearest")

    bs, _, h, w = img.shape
    if lam is None:
        lam = np.random.beta(alpha, alpha)

    rand_index = torch.randperm(bs).to(img.device)
    img_b = img[rand_index]
    mask_b = mask[rand_index]
    y_a = gt_label
    y_b = gt_label[rand_index]

    grid_h = max(1, h // grid_size)
    grid_w = max(1, w // grid_size)
    if grid_h == 0 or grid_w == 0:
        raise ValueError("grid_size is too large for current image resolution.")

    def _downsample(sal):
        return F.adaptive_avg_pool2d(sal, (grid_h, grid_w))

    sal_a = _downsample(mask)
    sal_b = _downsample(mask_b)

    sal_adv = (sal_a - sal_b).view(bs, -1)

    att_grid = sal_adv.size(1)
    if top_k is None:
        top_k = max(1, min(att_grid, int(att_grid * lam)))
    else:
        top_k = max(1, min(att_grid, top_k))

    k = sal_adv.topk(top_k, dim=1)
    _, att_idx = k
    mask_cells = torch.zeros(bs, att_grid, device=img.device)
    mask_cells.scatter_(1, att_idx, 1)
    mix_mask = mask_cells.view(bs, 1, grid_h, grid_w)
    mix_mask = F.interpolate(mix_mask, size=img.shape[2:], mode="nearest")

    img_mix = mix_mask * img + (1 - mix_mask) * img_b

    fg_vis_a = (mix_mask * mask).sum(dim=(-1, -2))
    fg_vis_b = ((1 - mix_mask) * mask_b).sum(dim=(-1, -2))
    total_a = mask.sum(dim=(-1, -2)).clamp(min=eps)
    total_b = mask_b.sum(dim=(-1, -2)).clamp(min=eps)

    abs_lam = (fg_vis_a / (fg_vis_a + fg_vis_b).clamp(min=eps)).clamp(0.0, 1.0)
    rel_lam = ((fg_vis_a / total_a) / ((fg_vis_a / total_a) + (fg_vis_b / total_b)).clamp(min=eps)).clamp(0.0, 1.0)

    if mode == "absolute":
        base = abs_lam
    elif mode == "relative":
        base = rel_lam
    else:
        mix_ratio = float(mode)
        base = mix_ratio * abs_lam + (1 - mix_ratio) * rel_lam

    area_lam = mix_mask.mean(dim=(-1, -2))
    out_lam = fg_weight * base + (1 - fg_weight) * area_lam

    if return_mask:
        img_mix = (img_mix, mix_mask)

    return img_mix, (y_a, y_b, out_lam.unsqueeze(1))
