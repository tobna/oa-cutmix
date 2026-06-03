_base_ = "r18_mixups.py"

# model settings
model = dict(
    pretrained=None,
    alpha=1.0,
    mix_mode="gridmix",
)
