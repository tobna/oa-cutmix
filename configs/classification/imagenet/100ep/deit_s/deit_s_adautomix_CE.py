_base_ = "deit_s_mixups.py"

# model settings
model = dict(
    type="AdAutoMix",
    pretrained=None,
    alpha=1.0,
    mix_samples=3,
    is_random=False,
    momentum=0.999,
    lam_margin=0.03,
    mixup_radio=0.5,
    beta_radio=0.3,
    debug=True,
    backbone=dict(
        type="VisionTransformer",
        arch="deit-small",
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=(5, 11),
    ),
    mix_block=dict(
        type="AdaptiveMask",
        in_channel=384,
        reduction=2,
        lam_concat=False,
        use_scale=True,
        unsampling_mode="nearest",
        scale_factor=8,
        frozen=False,
    ),
    head_one=dict(
        type="VisionTransformerClsHead",
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=1000, mode="original", loss_weight=1.0),
        in_channels=384,
        num_classes=1000,
    ),
    head_mix=dict(
        type="VisionTransformerClsHead",
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=1000, mode="original", loss_weight=1.0),
        in_channels=384,
        num_classes=1000,
    ),
    head_weights=dict(decent_weight=[], accent_weight=[], head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# interval for accumulate gradient
update_interval = 1

custom_hooks = [
    dict(
        type="SAVEHook",
        save_interval=500 * 20,
        iter_per_epoch=500,
    ),
    dict(
        type="CosineScheduleHook",
        end_momentum=0.99996,
        adjust_scope=[0.25, 1.0],
        warming_up="constant",
        update_interval=update_interval,
        interval=1,
    ),
]

# fp16
use_fp16 = False

# additional scheduler
addtional_scheduler = dict(
    policy="CosineAnnealing",
    by_epoch=False,
    min_lr=5e-6,
    paramwise_options=["mix_block"],
    warmup_iters=5,
    warmup_by_epoch=True,
    warmup_ratio=1e-5,
)
