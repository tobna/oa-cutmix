_base_ = "deit_ti_cutmix_fga_abs_l1p0.py"

data_source_cfg = dict(
    type="MaskImageFolder",
    root="data/TinyImageNetHD",
    mask_zip="/ds-sds/images/tiny-imagenet/train_masks_inter_shuffle.zip",
    force_resize=True,
)

data = dict(
    train=dict(
        data_source=dict(split="train", **data_source_cfg),
    ),
)
