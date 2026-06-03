_base_ = "../deit_ti_mixups.py"

data_source_cfg = dict(
    type="MaskImageFolder",
    root="data/TinyImageNetHD",
    mask_zip="/ds-sds/images/tiny-imagenet/train_masks.zip",
)

data = dict(
    train=dict(
        data_source=dict(split="train", **data_source_cfg),
    ),
)

# model settings
model = dict(
    pretrained=None,
    pretrained_k="torchvision://resnet50",
    alpha=2,  # float or list
    mix_mode="attentivemix_fga",
    mix_args=dict(
        attentivemix_fga=dict(mode="absolute", fg_weight=1.0),
    ),
    backbone_k=dict(  # PyTorch pre-trained R-18 is required for attentivemix+
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
)
