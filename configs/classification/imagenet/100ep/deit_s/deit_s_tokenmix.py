_base_ = "deit_s_mixups.py"

# model settings
model = dict(
    alpha=[1.0, 1.0],
    mix_mode=["tokenmix", "cutmix"],
    mix_prob=[0.5, 0.5],
    debug=True,
    mix_args=dict(
        tokenmix=dict(mask_type="block", minimum_tokens=8, token_w_h=8),
    ),
    backbone=dict(
        type="VisionTransformer",
        arch="deit-small",
        img_size=224,
        patch_size=16,
        drop_path=0.1,
        return_attn=True,
    ),
)

# fp16
use_fp16 = False
