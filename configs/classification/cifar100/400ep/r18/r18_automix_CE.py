_base_ = [
    "../../../_base_/datasets/cifar100/sz32_bs100.py",
    "../../../_base_/default_runtime.py",
]

# models settings
model = dict(
    type="AutoMixup",
    pretrained=None,
    alpha=2.0,
    momentum=0.999,
    mask_layer=2,
    mask_loss=0.1,
    mask_adjust=0,
    lam_margin=0.08,
    debug=True,
    backbone=dict(type="ResNet_Mix_CIFAR", depth=18, num_stages=4, out_indices=(2, 3), style="pytorch"),
    mix_block=dict(
        type="PixelMixBlock",
        in_channels=256,
        reduction=2,
        use_scale=True,
        unsampling_mode=[
            "nearest",
        ],
        lam_concat=False,
        lam_concat_v=False,
        lam_mul=False,
        lam_residual=False,
        lam_mul_k=-1,
        value_neck_cfg=None,
        x_qk_concat=False,
        x_v_concat=False,
        mask_loss_mode="L1",
        mask_loss_margin=0.1,
        frozen=False,
    ),
    head_one=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True,
        multi_label=False,
        in_channels=512,
        num_classes=100,
    ),
    head_mix=dict(
        type="ClsMixupHead",
        loss=dict(type="CrossEntropyLoss", use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True,
        multi_label=False,
        in_channels=512,
        num_classes=100,
    ),
    head_weights=dict(head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(type="CosineScheduleHook", end_momentum=0.999999, adjust_scope=[0.1, 1.0], warming_up="constant", interval=1),
    dict(
        type="SAVEHook",
        iter_per_epoch=500,
        save_interval=500 * 25,
    ),
]

# optimizer
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0.05)

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=400)
