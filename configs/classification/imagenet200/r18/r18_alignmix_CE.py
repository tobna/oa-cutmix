_base_ = "r18_mixups.py"

# halved batch size + gradient accumulation to match effective bs=256
data = dict(imgs_per_gpu=128)
optimizer_config = dict(grad_clip=dict(max_norm=5.0), update_interval=2)

# model settings
model = dict(
    alpha=[2, 2],  # list of alpha
    mix_mode=["manifoldmix", "alignmix"],  # AlignMix switches to {'mixup' or 'manifoldmix'}
    backbone=dict(
        type="ResNet_Mix",  # standard ImageNet stem (stride-2 + maxpool) → 7x7 features
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style="pytorch",
    ),
)
