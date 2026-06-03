_base_ = "deit_s_mixups_CE.py"

# model settings
model = dict(
    pretrained=None,
    alpha=1.0,
    mix_mode="tokenmix",
)
