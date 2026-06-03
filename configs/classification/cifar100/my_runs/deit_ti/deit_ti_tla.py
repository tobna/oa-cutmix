_base_ = "deit_ti_mixups_CE.py"


# model settings
model = dict(
    type="MixUpClassification",
    pretrained=None,
    alpha=1.0,
    mix_mode="tla",  # using Token Lableing Align and Mixup
    debug=True,
    mix_args=dict(
        alignmix=dict(eps=0.1, max_iter=100),
        attentivemix=dict(grid_size=32, top_k=None, beta=8),  # AttentiveMix+ in this repo (use pre-trained)
        automix=dict(mask_adjust=0, lam_margin=0),  # require pre-trained mixblock
        fmix=dict(decay_power=3, size=(224, 224), max_soft=0.0, reformulate=False),
        gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.0, cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        manifoldmix=dict(layer=(0, 3)),
        puzzlemix=dict(
            transport=True,
            t_batch_size=32,
            t_size=-1,  # adjust t_batch_size if CUDA out of memory
            mp=None,
            block_num=4,  # block_num<=4 and mp=2/4 for fast training
            beta=1.2,
            gamma=0.5,
            eta=0.2,
            neigh_size=4,
            n_labels=3,
            t_eps=0.8,
        ),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        samix=dict(mask_adjust=0, lam_margin=0.08),  # require pre-trained mixblock
        transmix=dict(mix_mode="cutmix"),
        mixpro=dict(num_classes=100, smoothing=0.1, mask_patch_size=64, model_patch_size=16),
    ),
    head=dict(
        _delete_=True,
        type="DistillationVisionTransformerClsHead",  # mixup CE + label smooth
        loss=dict(type="DistillationLoss", distillation_type="none", alpha=0.5, tau=1.0),
        in_channels=192,
        num_classes=100,
    ),
    backbone=dict(
        type="DistilledVisionTransformer",
        arch="deit-tiny",
        img_size=224,
        patch_size=16,
        drop_path=0.1,
        with_dis_token=True,
    ),
)

# fp16
use_fp16 = False
