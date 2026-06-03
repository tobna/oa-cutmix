_base_ = "r50_mixups_CE.py"

# model settings
model = dict(
    pretrained=None,
    alpha=1.0,
    mix_mode="gridmix",
)
