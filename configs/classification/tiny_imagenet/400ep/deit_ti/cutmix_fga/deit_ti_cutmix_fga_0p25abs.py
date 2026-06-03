_base_ = "deit_ti_cutmix_fga_abs_l1p0.py"


# model settings
model = dict(
    alpha=1.0,
    mix_mode="cutmix_foreground_area",
    mix_args=dict(
        cutmix_foreground_area=dict(fg_weight=1.0, mode=0.25),
    ),
)
