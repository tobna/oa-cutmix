_base_ = [
    "../../../../../_base_/datasets/cifar100/sz224_randaug_bs100.py",
    "../../../../../_base_/default_runtime.py",
]

# model settings
model = dict(
    type="MergeMix",
    pretrained=None,
    alpha=1.0,
    merge_num=98,
    mask_leaked=False,
    lam_scale=True,
    tome_in_mix=False,
    debug=False,
    backbone=dict(
        type="ToMeVisionTransformer",
        arch="deit-tiny",
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=(8, 11),
        return_attn=True,
    ),
)
