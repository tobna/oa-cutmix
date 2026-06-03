_base_ = "r50_mixups_CE.py"

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
        lam_scale_mode="none",  # mode pow with thr 1 and idx 1 means weights for labels a and b are both set to 1, i.e. weight vector sums to 2
        eta_weight=dict(eta=0.1, mode="both", thr=0.5),  # hyperparams for decoupled loss mode
    ),
)

# fp16
use_fp16 = False
