_base_ = "r18_mixups.py"

# model settings
model = dict(
    type='AdAutoMix',
    pretrained=None,
    alpha=1.0,
    mix_samples=3,
    is_random=False,
    momentum=0.999,
    lam_margin=0.03,
    mixup_radio=0.5,
    beta_radio=0.4,
    debug=True,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(2, 3),  # 2:[b,256,28,28], 3:[b,512,14,14]
        style='pytorch'),
    mix_block=dict(
        type='AdaptiveMask',
        in_channel=256,
        reduction=2,
        lam_concat=False,
        use_scale=True,
        unsampling_mode='bilinear',
        scale_factor=16,  # 16 for r18 on 224x224 (14->224)
        frozen=False),
    head_one=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True,
        multi_label=False,
        in_channels=512,
        num_classes=200),
    head_mix=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True,
        multi_label=False,
        in_channels=512,
        num_classes=200),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(
        type='CosineScheduleHook',
        end_momentum=0.99996,
        adjust_scope=[0.25, 1.0],
        warming_up="constant",
        interval=1,
    ),
    dict(
        type='SAVEHook',
        iter_per_epoch=500,
        save_interval=500 * 20,
    ),
]

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options={'mix_block': dict(lr=0.1, momentum=0.9, weight_decay=0.0001)},
)

# fp16
use_fp16 = False

# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
