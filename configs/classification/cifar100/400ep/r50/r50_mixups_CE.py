_base_ = [
    "../../../_base_/datasets/cifar100/sz32_bs100.py",
    "../../../_base_/default_runtime.py",
]

# model settings
model = dict(
    type="MixUpClassification",
    pretrained=None,
    alpha=1,
    mix_mode="mixup",
    mix_args=dict(
        alignmix=dict(eps=0.1, max_iter=100),
        attentivemix=dict(grid_size=32, top_k=None, beta=8),
        automix=dict(mask_adjust=0, lam_margin=0),
        fmix=dict(decay_power=3, size=(32, 32), max_soft=0.0, reformulate=False),
        gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.0, cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        manifoldmix=dict(layer=(0, 3)),
        puzzlemix=dict(
            transport=True,
            t_batch_size=None,
            t_size=4,
            block_num=5,
            beta=1.2,
            gamma=0.5,
            eta=0.2,
            neigh_size=4,
            n_labels=3,
            t_eps=0.8,
        ),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        samix=dict(mask_adjust=0, lam_margin=0.08),
        snapmix=dict(),
    ),
    backbone=dict(
        type="ResNet_Mix_CIFAR",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        with_avg_pool=True,
        multi_label=False,
        in_channels=2048,
        num_classes=100,
    ),
)

# additional hooks
custom_hooks = [
    dict(type="SAVEHook", iter_per_epoch=500, save_interval=12500),
]

# optimizer
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0.0)

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=400)
