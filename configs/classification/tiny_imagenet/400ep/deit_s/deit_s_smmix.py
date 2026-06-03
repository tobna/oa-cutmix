_base_ = "deit_s_mixups.py"

# model settings
model = dict(
    type="MixUpClassification",
    pretrained=None,
    alpha=[1.0, 1.0],
    mix_mode=["smmix", "mixup"],
    mix_prob=[0.8, 0.2],
    debug=True,
    mix_args=dict(
        smmix=dict(side=8, min_side_ratio=0.25, max_side_ratio=0.75),
    ),
    backbone=dict(
        return_attn=True,
    ),
)

# interval for accumulate gradient
update_interval = 1

# fp16
use_fp16 = False
