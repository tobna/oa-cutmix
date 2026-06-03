_base_ = "deit_ti_mixups.py"

# model settings
model = dict(
    type="MixUpClassification",
    pretrained=None,
    alpha=[1.0, 1.0],
    mix_mode=["mixpro", "mixup"],
    mix_prob=[0.8, 0.2],
    debug=True,
    mix_args=dict(
        mixpro=dict(num_classes=200, smoothing=0.1, mask_patch_size=64, model_patch_size=16),
    ),
    backbone=dict(
        type="VisionTransformer",
        arch="deit-tiny",
        img_size=224,
        patch_size=16,
        drop_path=0.1,
        return_attn=True,
    ),
)
