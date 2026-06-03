_base_ = "r18_mixups_CE.py"

# model settings
model = dict(
    alpha=0.8,
    mix_mode="mixup",
)
