_base_ = "r50_mixups_CE.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="vanilla",
)
