_base_ = "r50_cutmix_fga_abs_l1p0.py"

data_source_cfg = dict(
    type="MaskCUB2011",
    root="data/CUB2011",
    mask_zip="/fscratch/nauen/datasets/cub200-masks/train_masks_intra_shuffle.zip",
    force_resize=True,
)

data = dict(
    train=dict(
        data_source=dict(split="train", **data_source_cfg),
    ),
)
