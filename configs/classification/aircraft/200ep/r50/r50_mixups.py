_base_ = [
    "../../../_base_/datasets/aircraft/sz224_bs32.py",
    "../../../_base_/default_runtime.py",
]

# model settings — finetune from ImageNet-pretrained ResNet-18
model = dict(
    type="MixUpClassification",
    pretrained="torchvision://resnet50",
    alpha=1,
    mix_mode="mixup",
    mix_args=dict(
        alignmix=dict(eps=0.1, max_iter=100),
        attentivemix=dict(grid_size=32, top_k=None, beta=8),
        automix=dict(mask_adjust=0, lam_margin=0),
        fmix=dict(decay_power=3, size=(224, 224), max_soft=0.0, reformulate=False),
        gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.0, cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        manifoldmix=dict(layer=(0, 3)),
        puzzlemix=dict(
            transport=True,
            t_batch_size=32,
            t_size=-1,
            mp=None,
            block_num=4,
            beta=1.2,
            gamma=0.5,
            eta=0.2,
            neigh_size=4,
            n_labels=3,
            t_eps=0.8,
        ),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        samix=dict(mask_adjust=0, lam_margin=0.08),
        transmix=dict(mix_mode="cutmix"),
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
    head=dict(
        type="ClsHead",
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=100, mode="original", loss_weight=1.0),
        with_avg_pool=True,
        multi_label=False,
        in_channels=2048,
        num_classes=100,
    ),
)

# optimizer — lr linearly scaled from 0.001 @ bs16 to bs32
optimizer = dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=1e-6,
    warmup="linear",
    warmup_iters=5,
    warmup_by_epoch=True,
    warmup_ratio=1e-3,
)

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=200)
