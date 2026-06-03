_base_ = "r18_mixups.py"

# model settings
model = dict(
    type="Classification",
    pretrained=None,
    backbone=dict(
        type="ResNet",
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        with_avg_pool=True,
        multi_label=False,
        in_channels=512,
        num_classes=1000,
    ),
)
