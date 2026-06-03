_base_ = "r50_mixups_CE.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="snapmix",
    head=dict(
        type="ClsMixupHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        with_avg_pool=True,
        in_channels=2048,
        num_classes=100,
    ),
)
