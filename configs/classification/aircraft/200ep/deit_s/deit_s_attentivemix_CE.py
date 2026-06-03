_base_ = "deit_s_mixups.py"

# model settings
model = dict(
    pretrained_k="torchvision://resnet50",
    alpha=2,
    mix_mode="attentivemix",
    backbone_k=dict(
        type="ResNet", depth=50, num_stages=4, out_indices=(3,), style="pytorch"
    ),
)
