import os
import mmcv
import numpy as np
from PIL import Image

from ..registry import DATASOURCES
from .image_list import ImageList


@DATASOURCES.register_module
class ImageListMask(ImageList):
    """ImageList with mask support for mask-based mixup.

    The `ImageListMask` extends ImageList to also load object masks alongside
    images. Masks are expected to be in a parallel directory structure:
    
        /path/to/images/class1/img1.jpg
        /path/to/masks/class1/img1.png

    Or with a prefix/suffix:
    
        /path/to/images/class1/img1.jpg
        /path/to/masks/class1/img1_mask.png

    Args:
        root (str): Path to the dataset.
        list_file (str): Path to the txt list file.
        splitor (str): Splitor between file names and the class id.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='pillow')``.
        return_label (bool): Whether to return the class id.
        mask_root (str): Path to the masks directory. If None, assumes masks
            are in the same directory as images with .png extension.
        mask_suffix (str): Suffix to add to filename to get mask filename.
            Defaults to '' (e.g., img.jpg -> img.png).
            Set to '_mask' for img.jpg -> img_mask.png.
    """

    CLASSES = None

    def __init__(self,
                 root,
                 list_file,
                 splitor=" ",
                 file_client_args=dict(backend='pillow'),
                 return_label=True,
                 mask_root=None,
                 mask_suffix=''):
        super(ImageListMask, self).__init__(
            root, list_file, splitor, file_client_args, return_label)
        
        self.mask_root = mask_root if mask_root is not None else root
        self.mask_suffix = mask_suffix

    def _get_mask_path(self, img_path):
        """Get the corresponding mask path for an image path."""
        directory = os.path.dirname(img_path)
        basename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(basename)[0]
        ext = '.png'
        
        if self.mask_root == self.root:
            mask_path = os.path.join(directory, name_without_ext + self.mask_suffix + ext)
        else:
            rel_path = os.path.relpath(img_path, self.root)
            mask_path = os.path.join(self.mask_root, rel_path)
            mask_path = os.path.splitext(mask_path)[0] + self.mask_suffix + ext
        
        return mask_path

    def get_sample(self, idx):
        img = super(ImageListMask, self).get_sample(idx)
        
        mask_path = self._get_mask_path(self.fns[idx])
        
        if self.backend == 'pillow':
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = mask.convert('L')
            else:
                mask = Image.new('L', img.size, 255)
        else:
            if os.path.exists(mask_path):
                mask_bytes = self.file_client.get(mask_path)
                mask = mmcv.imfrombytes(mask_bytes, flag='grayscale')
                if mask is None:
                    mask = np.full((img.height, img.width), 255, dtype=np.uint8)
                else:
                    mask = Image.fromarray(mask.astype(np.uint8))
            else:
                mask = Image.new('L', img.size, 255)

        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return (img, mask, target)
        else:
            return (img, mask)
