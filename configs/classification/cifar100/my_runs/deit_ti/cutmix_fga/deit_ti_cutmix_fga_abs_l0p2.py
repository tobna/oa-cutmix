_base_ = "../deit_ti_mixups_CE.py"


data_source_cfg = dict(
    type="MaskCIFAR100",
    root="data/cifar100/",
    mask_zip="/fscratch/nauen/datasets/cifar100-masks/train_masks_merged.zip",
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
        cutmix_foreground_area=dict(fg_weight=0.2, mode="absolute"),
    ),
)
