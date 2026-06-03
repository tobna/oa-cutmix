_base_ = "r50_mixups.py"

# model settings
model = dict(
    pretrained=None,
    alpha=1.0,
    mix_mode="cutmix_stupid",
    stupidity=0.8,
)
