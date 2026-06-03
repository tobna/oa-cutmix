_base_ = "r50_mixups_CE.py"

# model settings
model = dict(
    alpha=0.8,
    mix_mode="mixup",
)
