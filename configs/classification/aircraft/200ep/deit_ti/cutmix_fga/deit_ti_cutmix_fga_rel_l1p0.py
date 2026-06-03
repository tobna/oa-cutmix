_base_ = "../deit_ti_mixups.py"
data_source_cfg = dict(
    type="MaskAircraft",
    
    root="data/Aircraft",
    mask_zip="/fscratch/nauen/datasets/aircraft-masks/train_masks.zip",
    
)

data = dict(
    train=dict(
        data_source=dict(split="trainval", **data_source_cfg),
    ),
)

# model settings
model = dict(
    alpha=1.0,
    mix_mode="cutmix_foreground_area",
    mix_args=dict(
        cutmix_foreground_area=dict(fg_weight=1.0, mode="relative"),
    ),
    head=dict(
        type="ClsMixupHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0, use_soft=True, use_sigmoid=False),
        multi_label=True,
        extract_cls_token=True,
    ),
)
