_base_ = "deit_ti_mixups_CE.py"

# model settings
model = dict(
    alpha=0.8,
    mix_mode="mixup",
)
