_base_ = "deit_s_mixups_CE.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="cutmix",
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
        lam_scale_mode="none",  # use pure eta as weigth
        eta_weight=dict(eta=0.1, mode="both", thr=0.5),  # hyperparams for decoupled loss mode
        extract_cls_token=True,
    ),
)

# fp16
use_fp16 = False
