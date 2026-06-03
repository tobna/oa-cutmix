_base_ = [
    "../_base_/datasets/cub200/sz224_bs128.py",
    "../_base_/default_runtime.py",
]

train_source = dict(
    type="MaskCUB2011",
    root="data/CUB2011",
    mask_zip="/fscratch/nauen/datasets/cub200-masks/train_masks.zip",
)
val_source = dict(
    type="MaskCUB2011",
    root="data/CUB2011",
    mask_zip="/fscratch/nauen/datasets/cub200-masks/test_masks.zip",
)

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_pipeline = [
    dict(type="ResizePair", size=256),
    dict(type="CenterCrop", size=224),
]
test_pipeline.extend([dict(type="ToTensor"), dict(type="Normalize", **img_norm_cfg)])

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
