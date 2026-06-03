_base_ = "r18_mixups.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="snapmix",
    head=dict(
        type="ClsMixupHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        with_avg_pool=True,
        in_channels=512,
        num_classes=200,
    ),
)
