_base_ = "deit_s_mixups.py"

# model settings
model = dict(
    alpha=2.0,
    mix_mode="cutmix",
)
