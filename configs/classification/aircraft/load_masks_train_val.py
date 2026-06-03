_base_ = [
    "../_base_/datasets/aircraft/sz224_bs32.py",
    "../_base_/default_runtime.py",
]

train_source = dict(
    type="MaskAircraft",
    root="data/Aircraft",
    mask_zip="/fscratch/nauen/datasets/aircraft-masks/train_masks.zip",
)
val_source = dict(
    type="MaskAircraft",
    root="data/Aircraft",
    mask_zip="/fscratch/nauen/datasets/aircraft-masks/test_masks.zip",
)

test_pipeline = [
    dict(type="ResizePair", size=256),
    dict(type="CenterCrop", size=224),
]

data = dict(
    train=dict(
        data_source=dict(split="trainval", **train_source),
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
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0, use_soft=True, use_sigmoid=False),
        multi_label=True,
    ),
)
