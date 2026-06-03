_base_ = "deit_ti_attentivemix_fga_abs.py"

data = dict(
    imgs_per_gpu=10,
)

# model settings
model = dict(
    mix_args=dict(
        attentivemix_fga=dict(mode="absolute", fg_weight=1.0, debug=True),
    ),
)
