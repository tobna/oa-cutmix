_base_ = "deit_ti_mixups_CE.py"

# value_neck_cfg
conv1x1 = dict(
    type="ConvNeck",
    in_channels=192,
    hid_channels=192,
    out_channels=1,
    num_layers=2,
    kernel_size=1,
    with_last_norm=False,
    norm_cfg=dict(type="BN"),
    with_last_dropout=0.1,
    with_avg_pool=False,
    with_residual=False,
)

# model settings
model = dict(
    type="AutoMixup",
    pretrained=None,
    alpha=2.0,
    momentum=0.999,
    mask_layer=2,
    mask_loss=0.1,
    mask_adjust=0,
    lam_margin=0.08,
    switch_off=1.0,
    mask_up_override=None,
    debug=True,
    backbone=dict(
        type="VisionTransformer",
        arch="deit-tiny",
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=(5, 11),
    ),
    mix_block=dict(
        type="PixelMixBlock",
        in_channels=192,
        reduction=2,
        use_scale=True,
        unsampling_mode=[
            "nearest",
        ],
        lam_concat=False,
        lam_concat_v=False,
        lam_mul=True,
        lam_residual=True,
        lam_mul_k=-1,
        value_neck_cfg=conv1x1,
        x_qk_concat=True,
        x_v_concat=False,
        mask_loss_mode="L1+Variance",
        mask_loss_margin=0.1,
        frozen=False,
    ),
    head_one=dict(
        type="VisionTransformerClsHead",
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=200, mode="original", loss_weight=1.0),
        in_channels=192,
        num_classes=200,
    ),
    head_mix=dict(
        type="VisionTransformerClsHead",
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=200, mode="original", loss_weight=1.0),
        in_channels=192,
        num_classes=200,
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
        type="CustomCosineAnnealingHook",
        attr_name="mask_loss",
        attr_base=0.1,
        min_attr=0.0,
        by_epoch=False,
        update_interval=update_interval,
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
    min_lr=1e-4,
    paramwise_options=["mix_block"],
    warmup_iters=20,
    warmup_by_epoch=True,
    warmup_ratio=1e-5,
)
