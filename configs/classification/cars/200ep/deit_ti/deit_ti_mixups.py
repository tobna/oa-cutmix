_base_ = [
    "../../../_base_/datasets/cars/sz224_bs32.py",
    "../../../_base_/default_runtime.py",
]

# model settings — finetune from DeiT III Tiny ImageNet-1k pretrained
model = dict(
    type="MixUpClassification",
    pretrained="https://dl.fbaipublicfiles.com/deit/deit_3_tiny_224_1k.pth",
    alpha=[1, 0.8],
    mix_mode=["cutmix", "mixup"],
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
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True, interpolate_mode="bilinear"),
        samix=dict(mask_adjust=0, lam_margin=0.08),
        transmix=dict(mix_mode="cutmix"),
    ),
    backbone=dict(
        type="VisionTransformer",
        arch="deit-tiny",
        img_size=224,
        patch_size=16,
        drop_path=0.0,  # no stochastic depth for finetuning
        init_values=1e-6,  # LayerScale as used in DeiT III
    ),
    head=dict(
        type="VisionTransformerClsHead",
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, num_classes=196, mode="original", loss_weight=1.0),
        in_channels=192,
        num_classes=196,
    ),
    init_cfg=[
        dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
        dict(type="Constant", layer=["LayerNorm", "BatchNorm"], val=1.0, bias=0.0),
    ],
)

# optimizer — lr linearly scaled from 5e-4 @ bs512 to bs32
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_options={
        "(bn|ln|gn)(\d+)?.(weight|bias)": dict(weight_decay=0.0),
        "norm": dict(weight_decay=0.0),
        "bias": dict(weight_decay=0.0),
        "cls_token": dict(weight_decay=0.0),
        "pos_embed": dict(weight_decay=0.0),
    },
)

update_interval = 1

use_fp16 = False
optimizer_config = dict(grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    by_epoch=False,
    min_lr=1e-7,
    warmup="linear",
    warmup_iters=5,
    warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=200)
