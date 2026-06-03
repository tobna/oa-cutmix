# dataset settings
data_source_cfg = dict(type="ImageFolder", root="data/ImageNet")

dataset_type = "ClassificationDataset"
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type="RandomResizedCropPair", size=224, interpolation=3),  # bicubic
    dict(type="RandomHorizontalFlip"),
]
test_pipeline = [
    dict(type="Resize", size=256, interpolation=3),  # 0.85
    dict(type="CenterCrop", size=224),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type="ToTensor"), dict(type="Normalize", **img_norm_cfg)])

data = dict(
    imgs_per_gpu=512,  # V100: 64 x 4gpus = bs256
    workers_per_gpu=16,  # according to total cpus cores, usually 4 workers per 32~128 imgs
    train=dict(
        type=dataset_type,
        data_source=dict(split="train", **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(split="val", **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ),
)

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    imgs_per_gpu=512,
    workers_per_gpu=16,
    eval_param=dict(topk=(1, 5)),
)

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
