import numpy as np
import sys
import torch
import torch.nn.functional as F

from openmixup.models.utils.visualization import plot_image_mask_pairs


@torch.no_grad()
def attentivemix_fga(
    img,
    gt_label,
    obj_mask,
    alpha=1.0,
    lam=None,
    dist_mode=False,
    features=None,
    grid_scale=32,
    top_k=6,
    eps=1e-6,
    mode="absolute",
    fg_weight=1.0,
    debug=False,
    return_mask=False,
    **kwargs,
):
    r"""AttentiveMix augmentation

    "Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning
    Based Image Classification (https://arxiv.org/abs/2003.13048)". In ICASSP, 2020.
        https://github.com/xden2331/attentive_cutmix

    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        features (tensor): Feature maps for attentive regions.
        grid_scale (float): The upsampling scale of attentive grids.
        top_k (int): Using top_k attentive regions in feature maps.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """

    # basic mixup args
    bs, _, att_size, _ = features.size()
    att_grid = att_size**2
    if att_size * grid_scale != img.size(2):
        grid_scale = img.size(2) / att_size
    if lam is None:
        lam = np.random.beta(alpha, alpha)
    # Notice: official attentivemix uses fixed lam by top_k, while attentivemix+
    #   in this repo uses lam\in\Beta(a,a) to choose top_k for better preformances.
    if top_k is None:
        top_k = min(max(1, int(att_grid * lam)), att_grid)

    if not dist_mode:
        # normal mixup process
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:  # [N, C, H, W]
            img_ = img[rand_index]
        else:
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # Notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        y_a = gt_label
        y_b = gt_label[rand_index]
        obj_mask_ = obj_mask[rand_index]
    else:
        raise ValueError("AttentiveMix cannot perform distributed mixup.")

    if debug:
        print("[DEBUG] Visualizing input images and masks...")
        plot_image_mask_pairs(
            img,
            obj_mask,
            titles=[f"Input {i}" for i in range(img.size(0))],
            show=False,
            save_path="test_output/attentivemix_fga_input.png",
        )

    # select top_k attentive regions
    features = features.mean(1)
    _, att_idx = features.view(bs, att_grid).topk(top_k)
    att_idx = torch.cat(
        [
            (att_idx // att_size).unsqueeze(1),
            (att_idx % att_size).unsqueeze(1),
        ],
        dim=1,
    )
    cutmix_mask = torch.zeros(bs, 1, att_size, att_size).cuda()
    for i in range(bs):
        cutmix_mask[i, 0, att_idx[i, 0, :], att_idx[i, 1, :]] = 1
    cutmix_mask = F.upsample(cutmix_mask, scale_factor=grid_scale, mode="nearest")
    lam = float(cutmix_mask[0, 0, ...].mean().cpu().numpy())
    img = cutmix_mask * img + (1 - cutmix_mask) * img_
    if return_mask:
        img = (img, cutmix_mask)

    img_a_vis = (cutmix_mask * obj_mask).sum(dim=(-1, -2))
    img_b_vis = ((1 - cutmix_mask) * obj_mask_).sum(dim=(-1, -2))
    total_a = obj_mask.sum(dim=(-1, -2)).clamp(min=eps)
    total_b = obj_mask_.sum(dim=(-1, -2)).clamp(min=eps)

    abs_lam = (img_a_vis / (img_a_vis + img_b_vis).clamp(min=eps)).clamp(0.0, 1.0)
    rel_lam = ((img_a_vis / total_a) / (img_a_vis / total_a + img_b_vis / total_b).clamp(min=eps)).clamp(0.0, 1.0)

    if abs_lam.isfinite().logical_not().any().item():
        print("img:", img.shape)
        print("y_a:", y_a)
        print("y_b:", y_b)
        print("visible a:", img_a_vis)
        print("total a:", total_a)
        print("visible b:", img_b_vis)
        print("total b:", total_b)
        print("absolute weight:", abs_lam)
        print("relative weight:", rel_lam, flush=True)
        raise ValueError("Got NaN/Inf value in lam.")

    if mode == "absolute":
        mode = 1.0
    elif mode == "relative":
        mode = 0.0
    else:
        assert (
            isinstance(mode, float) and 0.0 < mode < 1.0
        ), f"Mode needs to be 'absolute', 'relative' or float between 0 and 1, but got {mode}"

    out_weight = fg_weight * (mode * abs_lam + (1 - mode) * rel_lam) + (1 - fg_weight) * lam

    if debug:
        print("[DEBUG] Visualizing output images and masks after CutMix...")
        obj_mask = cutmix_mask * obj_mask + (1 - cutmix_mask) * obj_mask_
        plot_image_mask_pairs(
            img,
            obj_mask,
            titles=[
                f"Output {i}: abs=({abs_lam[i].item():.2f}/{1-abs_lam[i].item():.2f}),"
                f" rel=({rel_lam[i].item():.2f}/{1-rel_lam[i].item():.2f}), area=({lam:.2f}/{1-lam:.2f})"
                for i in range(img.size(0))
            ],
            show=False,
            save_path="test_output/attentivemix_fga_output.png",
        )
        print("[DEBUG] Exiting after debug visualization.")
        sys.exit(0)

    return img, (y_a, y_b, out_weight)
