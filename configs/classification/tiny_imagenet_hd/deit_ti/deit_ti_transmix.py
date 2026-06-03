_base_ = "deit_ti_mixups.py"

# model settings
model = dict(
    alpha=[0.8, 1.0],
    mix_mode=["mixup", "transmix"],
    mix_prob=[0.2, 0.8],
    backbone=dict(return_attn=True),
)
