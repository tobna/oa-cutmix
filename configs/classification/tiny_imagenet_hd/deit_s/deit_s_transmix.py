_base_ = "deit_s_mixups.py"

# model settings
model = dict(
    alpha=[0.8, 1.0],
    mix_mode=["mixup", "transmix"],
    mix_prob=[0.2, 0.8],
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
