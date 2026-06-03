_base_ = "deit_ti_mixups.py"

# model settings
model = dict(
    alpha=[2, 2],
    mix_mode=["mixup", "alignmix"],
)
