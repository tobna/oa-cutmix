# configs/classification/_base_/datasets/tiny_imagenet_hd/sz64_bs100.py
# This config uses the ImageFolder data source with ClassificationDataset.

data_source_cfg = dict(type="ImageFolder", root="data/TinyImageNetHD")

dataset_type = "ClassificationDataset"
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type="RandomResizedCropPair", size=224, interpolation=3),  # bicubic
    dict(type="RandomHorizontalFlip"),
]
test_pipeline = []
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type="ToTensor"), dict(type="Normalize", **img_norm_cfg)])
test_pipeline.extend([dict(type="ToTensor"), dict(type="Normalize", **img_norm_cfg)])

data = dict(
    imgs_per_gpu=100,  # 100 x 1gpu = 100
    workers_per_gpu=4,
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
    imgs_per_gpu=100,
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)),
    save_best="auto",
)

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
