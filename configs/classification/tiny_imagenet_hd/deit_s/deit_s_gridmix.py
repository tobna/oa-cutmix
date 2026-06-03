_base_ = "deit_s_mixups.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="gridmix",
)
