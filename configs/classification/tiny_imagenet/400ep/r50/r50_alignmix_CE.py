_base_ = "r50_mixups.py"

# model settings
model = dict(
    alpha=[2, 2],  # list of alpha
    mix_mode=["mixup", "alignmix"],  # AlignMix switches to {'mixup' or 'manifoldmix'}
    backbone=dict(
        type="ResNet_Mix",  # standard ImageNet stem
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style="pytorch",
    ),
)
