_base_ = "r18_mixups.py"

# model settings
model = dict(
    pretrained=None,
    pretrained_k="torchvision://resnet18",
    alpha=2,
    mix_mode="attentivemix",
    backbone_k=dict(
        type="ResNet", depth=18, num_stages=4, out_indices=(3,), style="pytorch"
    ),
)
