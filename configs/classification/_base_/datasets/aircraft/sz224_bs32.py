# dataset settings
data_source_cfg = dict(type="Aircraft", root="data/Aircraft")

dataset_type = "ClassificationDataset"
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type="ResizePair", size=256),
    dict(type="RandomResizedCropPair", size=224, scale=[0.5, 1.0]),
    dict(type="RandomHorizontalFlip"),
    dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # dict(type="RandomGrayscale", p=0.2),
]
test_pipeline = [
    dict(type="Resize", size=256),
    dict(type="CenterCrop", size=224),
]

# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type="ToTensor"), dict(type="Normalize", **img_norm_cfg)])
test_pipeline.extend([dict(type="ToTensor"), dict(type="Normalize", **img_norm_cfg)])

data = dict(
    imgs_per_gpu=32,  # larger batch OK with 100 classes and 6667 trainval images
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(split="trainval", **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(split="test", **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ),
)

# validation hook
evaluation = dict(initial=False, interval=1, imgs_per_gpu=100, workers_per_gpu=4, eval_param=dict(topk=(1, 5)))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
