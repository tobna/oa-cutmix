_base_ = "r50_mixups.py"

# model settings
model = dict(
    alpha=0.8,
    mix_mode="mixup",
)
