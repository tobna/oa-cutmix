_base_ = "deit_s_attentivemix_fga_abs.py"

# model settings
model = dict(
    mix_args=dict(
        attentivemix_fga=dict(mode="relative", fg_weight=1.0),
    ),
)
