_base_ = "deit_ti_mixups_CE.py"

# model settings
model = dict(
    type="AdAutoMix",
    pretrained=None,
    alpha=1.0,
    mix_samples=3,  # mix samples number
    is_random=False,
    momentum=0.999,  # 0.999 to 0.999999
    lam_margin=0.03,
    mixup_radio=0.5,
    beta_radio=0.3,
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
        type="AdaptiveMask",
        in_channel=192,
        reduction=2,
        lam_concat=False,
        use_scale=True,
        unsampling_mode="nearest",
        scale_factor=16,  # 4 for r18 and rx50; 2 for wrn and 16 for vits
        frozen=False,
    ),
    head_one=dict(
        type="VisionTransformerClsHead",  # mixup CE + label smooth
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=100, mode="original", loss_weight=1.0),
        in_channels=192,
        num_classes=100,
    ),
    head_mix=dict(
        type="VisionTransformerClsHead",  # mixup CE + label smooth
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=100, mode="original", loss_weight=1.0),
        in_channels=192,
        num_classes=100,
    ),
    head_weights=dict(decent_weight=[], accent_weight=[], head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs128 x 1 accumulates = bs1024

custom_hooks = [
    dict(
        type="SAVEHook",
        save_interval=500 * 20,  # 20 ep
        iter_per_epoch=500,
    ),
    dict(
        type="CosineScheduleHook",
        end_momentum=0.99996,  # 0.999 to 0.99996
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
    min_lr=5e-6,  # 0.1 x lr
    paramwise_options=["mix_block"],
    warmup_iters=5,
    warmup_by_epoch=True,
    warmup_ratio=1e-5,
)
