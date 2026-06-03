_base_ = "r18_mixups_CE.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="gridmix",
)
