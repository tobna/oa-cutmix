import os
from pathlib import Path
from PIL import Image

from ..registry import DATASOURCES


@DATASOURCES.register_module
class ImageFolder(object):
    """Folder-based image dataset for classification.

    This data source reads images directly from a folder hierarchy where
    each subdirectory in the root represents a class.

    Expected layout:
        root/
            <class_name1>/
                img1.jpg
                img2.jpg
            <class_name2>/
                img3.jpg
                img4.jpg

    Args:
        root (str): Root directory containing class subdirectories.
        split (str): Dataset split in ['train', 'val', 'test'].
        return_label (bool): Whether to return class labels. Default: True.
    """

    CLASSES = None

    def __init__(self, root, split='train', return_label=True, classes_file=None):
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        self.root = root
        self.split = split
        self.return_label = return_label

        split_root = os.path.join(self.root, self.split)
        if not os.path.exists(split_root):
            raise ValueError(f"Data split directory not found: {split_root}")

        all_classes = sorted([p.name for p in Path(split_root).iterdir() if p.is_dir()])
        if classes_file is not None:
            with open(classes_file) as f:
                allowed = {l.strip() for l in f if l.strip()}
            self.CLASSES = [c for c in all_classes if c in allowed]
            if len(self.CLASSES) == 0:
                raise ValueError(f"No matching classes found for classes_file: {classes_file}")
        else:
            self.CLASSES = all_classes
        self.fns = []
        self.labels = []
        
        for label, cls_name in enumerate(self.CLASSES):
            cls_dir = Path(split_root) / cls_name
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
                    self.fns.append(str(img_file))
                    self.labels.append(label)
        
        if len(self.fns) == 0:
            raise ValueError(f"No images found in {split_root}")

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx]).convert('RGB')
        if self.return_label:
            return img, self.labels[idx]
        else:
            return img
