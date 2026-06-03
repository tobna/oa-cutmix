_base_ = [
    "../../../_base_/datasets/tiny_imagenet/sz64_randaug_bs256.py",
    "../../../_base_/default_runtime.py",
]

# value_neck_cfg
conv1x1 = dict(
    type="ConvNeck",
    in_channels=256,
    hid_channels=128,
    out_channels=1,
    num_layers=2,
    kernel_size=1,
    with_last_norm=False,
    norm_cfg=dict(type="BN"),
    with_last_dropout=0,
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
    mask_up_override=None,
    debug=True,
    backbone=dict(
        type="ResNet", depth=18, num_stages=4, out_indices=(2, 3), style="pytorch"  # 2:[b,256,28,28], 3:[b,512,14,14]
    ),
    mix_block=dict(
        type="PixelMixBlock",
        in_channels=256,
        reduction=2,
        use_scale=True,
        unsampling_mode=[
            "bilinear",
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
        scale_factor=16,  # 16 for r18 on 224x224 (14->224)
        frozen=False,
    ),
    head_one=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True,
        multi_label=True,
        in_channels=512,
        num_classes=200,
    ),
    head_mix=dict(
        type="ClsMixupHead",
        loss=dict(type="CrossEntropyLoss", use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True,
        multi_label=True,
        in_channels=512,
        num_classes=200,
    ),
    head_mix_k=dict(
        type="ClsMixupHead",
        loss=dict(type="CrossEntropyLoss", use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True,
        multi_label=True,
        neg_weight=0.5,
        in_channels=512,
        num_classes=200,
    ),
    head_weights=dict(head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(
        type="CosineScheduleHook",
        end_momentum=0.999999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        update_interval=1,
    ),
    dict(
        type="SAVEHook",
        iter_per_epoch=500,
        save_interval=500 * 20,
    ),
]

# optimizer
optimizer = dict(
    type="SGD",
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options={"mix_block": dict(lr=0.1, momentum=0.9, weight_decay=0.0001)},
)

# fp16
use_fp16 = False
optimizer_config = dict(update_interval=1, grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0.0)

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=400)
