import argparse
import io
from zipfile import ZipFile

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("zip", type=str, help="Zipfile to inspect")
parser.add_argument("--file-ending", default="png", type=str, help="Image file ending")
args = parser.parse_args()


total_masks = close_zero_masks = zero_masks = 0
with ZipFile(args.zip, "r") as zf:
    for filename in tqdm(zf.namelist()):
        if not filename.endswith(args.file_ending):
            continue
        mask_data = zf.read(filename)
        mask = Image.open(io.BytesIO(mask_data))
        mask = mask.convert("L")
        total_masks += 1
        mask_arr = np.array(mask)
        if np.max(mask_arr) == 0:
            zero_masks += 1
        elif np.mean(mask_arr) < 0.01:
            close_zero_masks += 1

print(
    f"total: {total_masks}, ==0: {zero_masks} ({round(zero_masks / total_masks * 100)}%) <.01: {close_zero_masks}"
    f" ({round(close_zero_masks/total_masks*100)}%)"
)
