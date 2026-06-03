import argparse
import io
from zipfile import ZipFile

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs="+", type=str)
parser.add_argument("output", type=str)
args = parser.parse_args()

print(f"start merging zips: {args}")
assert args.output not in args.inputs
input_zips = [ZipFile(input, "r") for input in args.inputs]
img_count = [0] * (len(args.inputs) + 1)
files = set()
for inp_zip in tqdm(input_zips, desc="Gather filenames"):
    files.update(list(inp_zip.namelist()))
with ZipFile(args.output, "w") as out_zf:
    for filename in tqdm(files, desc="Collect Files"):
        saved_file = False
        file_in = len(input_zips)
        for i, in_zf in enumerate(input_zips):
            try:
                mask_data = in_zf.read(filename)
            except KeyError:
                continue
            mask = Image.open(io.BytesIO(mask_data))
            mask = mask.convert("L")
            mask_arr = np.array(mask)
            if i < file_in:
                file_in = i
            if np.max(mask_arr) > 0:
                buf = io.BytesIO()
                mask.save(buf, format="PNG")
                out_zf.writestr(filename, buf.getvalue())
                img_count[i] += 1
                saved_file = True
                break
        if not saved_file:
            if file_in == len(input_zips):
                print(f"Mask in no file: {filename}")
            else:
                mask_data = input_zips[file_in].read(filename)  # read the white image from the first zip
                mask = Image.open(io.BytesIO(mask_data))
                mask = mask.convert("L")
                buf = io.BytesIO()
                mask.save(buf, format="PNG")
                out_zf.writestr(filename, buf.getvalue())
                img_count[-1] += 1
for in_zf in input_zips:
    in_zf.close()
total_masks = sum(img_count)
print(
    "merged zips towards:"
    f" {[[zf, count, count/total_masks] for zf, count in zip(args.inputs + ['zero masks'], img_count)]}"
)
