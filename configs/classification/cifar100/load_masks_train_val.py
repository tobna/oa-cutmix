_base_ = [
    "../_base_/datasets/cifar100/sz32_randaug_bs100.py",
    "../_base_/default_runtime.py",
]

train_source = dict(
    type="MaskCIFAR100",
    root="data/cifar100/",
    mask_zip="/fscratch/nauen/datasets/cifar100-masks/train_masks_merged.zip",
)
val_source = dict(
    type="MaskCIFAR100",
    root="data/cifar100/",
    mask_zip="/fscratch/nauen/datasets/cifar100-masks/test_masks.zip",
)

img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
test_pipeline = [dict(type="ToTensor"), dict(type="Normalize", **img_norm_cfg)]

data = dict(
    train=dict(
        data_source=dict(split="train", **train_source),
    ),
    val=dict(
        data_source=dict(split="test", **val_source),
        pipeline=test_pipeline,
        prefetch=False,
    ),
)

# model settings
model = dict(
    alpha=1.0,
    mix_mode="cutmix_foreground_area",
    mix_args=dict(
        cutmix_foreground_area=dict(fg_weight=1.0, mode="absolute"),
    ),
    head=dict(
        type="ClsMixupHead",
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            use_soft=True,
            use_sigmoid=False,
        ),
        multi_label=True,
    ),
)
