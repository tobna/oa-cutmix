_base_ = "deit_ti_mixups_CE.py"

# model settings
model = dict(
    alpha=[1.0, 1.0],
    mix_mode=["tokenmix", "cutmix"],  # using TokenMix and Mixup
    mix_prob=[0.5, 0.5],
    debug=True,
    mix_args=dict(
        tokenmix=dict(mask_type="block", minimum_tokens=14),
    ),
    backbone=dict(
        return_attn=True,
    ),
)

# fp16
use_fp16 = False
