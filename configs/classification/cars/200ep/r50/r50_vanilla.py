_base_ = "r50_mixups.py"

# model settings
model = dict(
    type="Classification",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        with_avg_pool=True,
        multi_label=False,
        in_channels=2048,
        num_classes=196,
    ),
)
