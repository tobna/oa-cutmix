from torchvision.datasets import FGVCAircraft

from ..registry import DATASOURCES


@DATASOURCES.register_module
class Aircraft(object):
    """FGVC-Aircraft dataset via torchvision.

    Args:
        root (str): Root directory containing 'fgvc-aircraft-2013b' folder.
        split (str): Dataset split in ['train', 'val', 'trainval', 'test'].
        return_label (bool): Whether to return class labels. Default: True.
    """

    VALID_SPLITS = ["train", "val", "trainval", "test"]

    def __init__(self, root, split="trainval", return_label=True):
        assert split in self.VALID_SPLITS, f"Invalid split: {split}"
        self.return_label = return_label
        self.dataset = FGVCAircraft(
            root=root,
            split=split,
            annotation_level="variant",
            download=False,
        )
        self.CLASSES = self.dataset.classes
        self.labels = self.dataset._labels

    def get_length(self):
        return len(self.dataset)

    def get_sample(self, idx):
        img, label = self.dataset[idx]
        if self.return_label:
            return img, label
        return img
