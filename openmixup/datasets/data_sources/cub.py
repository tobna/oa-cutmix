import os
from pathlib import Path

import pandas as pd
from PIL import Image

from ..registry import DATASOURCES


@DATASOURCES.register_module
class CUB2011(object):
    """CUB-200-2011 dataset.

    Args:
        root (str): Root directory containing 'CUB_200_2011' folder.
        split (str): Dataset split in ['train', 'test'].
        return_label (bool): Whether to return class labels. Default: True.
    """

    BASE_FOLDER = "CUB_200_2011"

    def __init__(self, root, split="train", return_label=True):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.root = Path(root) / self.BASE_FOLDER
        self.split = split
        self.return_label = return_label

        if not self.root.exists():
            raise ValueError(f"Data root not found: {self.root}")

        paths = pd.read_csv(self.root / "images.txt", sep=" ", names=["id", "path"])
        labels = pd.read_csv(self.root / "image_class_labels.txt", sep=" ", names=["id", "label"])
        splits = pd.read_csv(self.root / "train_test_split.txt", sep=" ", names=["id", "is_training"])
        data = paths.merge(labels, on="id")
        data = data.merge(splits, on="id")
        classes_file = self.root / "classes.txt"
        if classes_file.exists():
            classes_df = pd.read_csv(classes_file, sep=" ", names=["id", "name"])
            self.CLASSES = classes_df["name"].tolist()
        else:
            self.CLASSES = list(range(len(labels["label"].unique())))

        if split == "train":
            self.data = data[data.is_training == 1].reset_index(drop=True)
        else:
            self.data = data[data.is_training == 0].reset_index(drop=True)
        self.labels = (self.data["label"] - 1).tolist()

        if len(self.data) == 0:
            raise ValueError(f"No images found for split: {split}")

    def get_length(self):
        return len(self.data)

    def get_sample(self, idx):
        sample = self.data.iloc[idx]
        path = self.root / "images" / sample.path
        label = sample.label - 1
        img = Image.open(path).convert("RGB")

        if self.return_label:
            return img, label
        return img
