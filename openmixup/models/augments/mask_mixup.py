import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def mask_mixup(img,
               gt_label,
               mask=None,
               alpha=1.0,
               lam=None,
               dist_mode=False,
               **kwargs):
    r""" Mask-guided MixUp augmentation.

    Uses object masks to re-weight labels after mixup based on the visible
    region of each class.

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        mask (Tensor): Object masks of shape (N, 1, H, W) with values in [0, 1].
            Must NOT be None. Will raise an error if mask is None.
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
    
    Raises:
        ValueError: If mask is None.
    """
    if mask is None:
        raise ValueError(
            "mask_mixup requires a valid mask tensor, but got None. "
            "Ensure your dataset returns masks and they are passed to the mixup function."
        )
    
    if lam is None:
        lam = np.random.beta(alpha, alpha)

    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:
            img_ = img[rand_index]
            mask_ = mask[rand_index]
        else:
            raise ValueError("mask_mixup does not support 5D input (semi-supervised)")
        
        y_a = gt_label
        y_b = gt_label[rand_index]
        
        img = lam * img + (1 - lam) * img_
        
        mixed_mask = lam * mask + (1 - lam) * mask_
        
        mask_a = mask
        mask_b = mask_
        
        mask_overlap_a = (mixed_mask * mask_a).sum(dim=(1, 2, 3)) / (mask_a.sum(dim=(1, 2, 3)) + 1e-8)
        mask_overlap_b = (mixed_mask * mask_b).sum(dim=(1, 2, 3)) / (mask_b.sum(dim=(1, 2, 3)) + 1e-8)
        
        weight_a = mask_overlap_a.view(-1, 1)
        weight_b = mask_overlap_b.view(-1, 1)
        
        return img, (y_a, y_b, weight_a, weight_b)

    else:
        raise ValueError("mask_mixup does not support dist_mode yet")


def standard_mixup_fallback(img, gt_label, alpha=1.0, lam=None, dist_mode=False, **kwargs):
    """Fallback to standard mixup when no mask is provided."""
    if lam is None:
        lam = np.random.beta(alpha, alpha)

    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:
            img_ = img[rand_index]
        else:
            assert img.dim() == 5
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()

        y_a = gt_label
        y_b = gt_label[rand_index]
        img = lam * img + (1 - lam) * img_
        return img, (y_a, y_b, lam)
    else:
        raise ValueError("standard_mixup_fallback does not support dist_mode")
