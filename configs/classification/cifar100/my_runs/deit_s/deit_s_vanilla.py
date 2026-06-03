_base_ = "deit_s_mixups_CE.py"

# model settings
model = dict(
    type="Classification",
    pretrained=None,
    backbone=dict(
        type="VisionTransformer",
        arch="deit-small",
        img_size=224,
        patch_size=16,
        drop_path=0.1,
    ),
    head=dict(
        type="VisionTransformerClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        in_channels=384,
        num_classes=100,
    ),
)
