_base_ = "../r50_mixups_CE.py"


data_source_cfg = dict(
    type="MaskCIFAR100",
    root="data/cifar100/",
    mask_zip="/fscratch/nauen/datasets/cifar100-masks/train_masks_merged.zip",
)

data = dict(
    train=dict(
        data_source=dict(split="train", **data_source_cfg),
    ),
)

# model settings
model = dict(
    alpha=1.0,
    mix_mode="cutmix_foreground_area",
    mix_args=dict(
        cutmix_foreground_area=dict(fg_weight=1.0, mode="absolute"),
    ),
    head=dict(
        type="ClsMixupHead",  # soft CE decoupled mixup
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            use_soft=True,
            use_sigmoid=False,
            use_mix_decouple=True,  # decouple mixup CE
        ),
        multi_label=True,
        two_hot=False,
        two_hot_scale=1,  # not two-hot
        lam_scale_mode="pow",  # mode pow with thr 1 and idx 1 means weights for labels a and b are both set to 1, i.e. weight vector sums to 2
        lam_thr=1,
        lam_idx=1,  # lam rescale, default as linear
        eta_weight=dict(eta=0.1, mode="both", thr=0.5),  # hyperparams for decoupled loss mode
    ),
)

# fp16
use_fp16 = False
