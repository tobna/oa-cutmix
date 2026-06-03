_base_ = "r18_mixups_CE.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="resizemix",
    mix_args=dict(
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
    ),
)
