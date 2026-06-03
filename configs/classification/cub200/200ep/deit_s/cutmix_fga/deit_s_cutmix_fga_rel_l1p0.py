_base_ = "../deit_s_mixups.py"
data_source_cfg = dict(
    type="MaskCUB2011",
    
    root="data/CUB2011",
    mask_zip="/fscratch/nauen/datasets/cub200-masks/train_masks.zip",
    
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
        cutmix_foreground_area=dict(fg_weight=1.0, mode="relative"),
    ),
    head=dict(
        type="ClsMixupHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0, use_soft=True, use_sigmoid=False),
        multi_label=True,
        extract_cls_token=True,
    ),
)
