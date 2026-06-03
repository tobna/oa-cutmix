_base_ = "../r18_mixups.py"


data_source_cfg = dict(
    type="MaskImageFolder",
    root="data/ImageNet",
    mask_zip="/fscratch/nauen/datasets/tiny-imagenet-masks/train_masks.zip",
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
        cutmix_foreground_area=dict(fg_weight=1.0, mode=0.75),
    ),
)
