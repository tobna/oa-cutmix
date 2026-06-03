_base_ = "../deit_s_mixups.py"


data_source_cfg = dict(
    type="MaskImageFolder",
    root="data/TinyImageNetHD",
    mask_zip="/ds-sds/images/tiny-imagenet/train_masks.zip",
)

data = dict(
    train=dict(
        data_source=dict(split="train", **data_source_cfg),
    ),
)

# model settings
model = dict(
    alpha=1.0,
    mix_mode="cutmix_foreground_area",
    mix_args=dict(
        cutmix_foreground_area=dict(fg_weight=0.4, mode="absolute"),
    ),
)
