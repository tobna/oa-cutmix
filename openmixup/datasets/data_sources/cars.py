from torchvision.datasets import StanfordCars

from ..registry import DATASOURCES


@DATASOURCES.register_module
class Cars(object):
    """Stanford Cars dataset via torchvision.

    Args:
        root (str): Root directory containing 'stanford_cars' folder.
        split (str): Dataset split in ['train', 'test'].
        return_label (bool): Whether to return class labels. Default: True.
    """

    VALID_SPLITS = ["train", "test"]

    def __init__(self, root, split="train", return_label=True):
        assert split in self.VALID_SPLITS, f"Invalid split: {split}"
        self.return_label = return_label
        self.dataset = StanfordCars(
            root=root,
            split=split,
            download=False,
        )
        self.CLASSES = self.dataset.classes
        self.labels = [s[1] for s in self.dataset._samples]

    def get_length(self):
        return len(self.dataset)

    def get_sample(self, idx):
        img, label = self.dataset[idx]
        if self.return_label:
            return img, label
        return img
