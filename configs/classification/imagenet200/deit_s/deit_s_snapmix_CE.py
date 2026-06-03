_base_ = "deit_s_mixups.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="snapmix",
    head=dict(
        type="ClsMixupHead",
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1,
                  num_classes=200, mode="original", loss_weight=1.0),
        in_channels=384,
        num_classes=200,
        extract_cls_token=True,
    ),
)
