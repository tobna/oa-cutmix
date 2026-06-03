#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from transformers import Sam3Model, Sam3Processor

try:
    # Import grounded_segmentation
    sys.path.append(str(Path(__file__).parent))
    from grounded_segmentation import grounded_segmentation
except Exception as e:
    print(f"WARNING: Failed to import grounded sam {type(e)}: {e}")


def load_class_mapping(csv_path):
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) >= 2:
                wordnet_id = row[0].strip()
                class_name = row[1].strip()
                hypernym = row[2].strip() if len(row) > 2 else None
                mapping[wordnet_id] = (class_name, hypernym)
    return mapping


def get_prompt(class_name, hypernym):
    if hypernym:
        return f"a {class_name}, {hypernym}"
    return f"a {class_name}"


def cub_class_name_from_folder(folder_name):
    """Extract human-readable name from CUB folder, e.g.
    '001.Black_footed_Albatross' -> 'Black footed Albatross'."""
    parts = folder_name.split(".", 1)
    name = parts[1] if len(parts) == 2 else parts[0]
    return name.replace("_", " ")


def aircraft_class_name_from_variant(variant):
    """Return a prompt-friendly name for an aircraft variant."""
    return variant


def image_iterator(args, id=0, processes=1):
    if args.dataset == "cifar100":
        train = args.split != "test"
        cifar = datasets.CIFAR100(root=args.input_path, train=train, download=False)
        for idx, (img, label) in enumerate(cifar):
            yield img.convert("RGB"), str(label), idx
    elif args.dataset == "aircraft":
        split = args.split if args.split != "all" else "trainval"
        aircraft = datasets.FGVCAircraft(
            root=args.input_path,
            split=split,
            annotation_level="variant",
            download=False,
        )
        image_files = aircraft._image_files
        labels = aircraft._labels
        classes = aircraft.classes
        if processes > 1:
            start_idx = round(len(image_files) / processes * id)
            end_idx = round(len(image_files) / processes * (id + 1))
            image_files = image_files[start_idx:end_idx]
            labels = labels[start_idx:end_idx]
        for img_path, label in zip(image_files, labels):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            class_id = classes[label]
            yield Image.open(img_path).convert("RGB"), class_id, stem
    elif args.dataset == "cars":
        from torchvision.datasets import StanfordCars

        split = args.split if args.split != "all" else "train"
        cars = StanfordCars(
            root=args.input_path,
            split=split,
            download=False,
        )
        samples = cars._samples
        classes = cars.classes
        if processes > 1:
            start_idx = round(len(samples) / processes * id)
            end_idx = round(len(samples) / processes * (id + 1))
            samples = samples[start_idx:end_idx]
        for img_path, label in samples:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            class_id = classes[label]
            yield Image.open(img_path).convert("RGB"), class_id, stem
    elif args.dataset == "cub200":
        import pandas as pd

        cub_root = Path(args.input_path) / "CUB_200_2011"
        paths = pd.read_csv(cub_root / "images.txt", sep=" ", names=["id", "path"])
        splits_df = pd.read_csv(
            cub_root / "train_test_split.txt",
            sep=" ",
            names=["id", "is_training"],
        )
        data = paths.merge(splits_df, on="id").reset_index(drop=True)
        if args.split == "train":
            data = data[data.is_training == 1].reset_index(drop=True)
        elif args.split == "test":
            data = data[data.is_training == 0].reset_index(drop=True)
        if processes > 1:
            start_idx = round(len(data) / processes * id)
            end_idx = round(len(data) / processes * (id + 1))
            data = data.iloc[start_idx:end_idx]
        for _, row in data.iterrows():
            img_path = cub_root / "images" / row.path
            # class_id is the folder name, e.g. "001.Black_footed_Albatross"
            class_id = row.path.split("/")[0]
            # image_id is the full relative path; mask will be at
            # image_id + ".png" to match MaskCUB2011 datasource
            yield Image.open(img_path).convert("RGB"), class_id, row.path
    else:
        input_path = Path(args.input_path)
        classes = sorted(input_path.iterdir())
        if processes > 1:
            start_idx = round(len(classes) / processes * id)
            end_idx = round(len(classes) / processes * (id + 1))
            classes = classes[start_idx:end_idx]
        for class_dir in classes:
            if class_dir.is_dir():
                class_id = class_dir.name
                for img_path in find_images_in_directory(class_dir):
                    yield Image.open(img_path).convert("RGB"), class_id, img_path.stem


SAM3_MODEL = None
SAM3_PROCESSOR = None


def process_sam3(image, prompt, device, threshold, mask_threshold, target_size):
    global SAM3_MODEL, SAM3_PROCESSOR
    if SAM3_MODEL is None:
        print("Loading SAM3 model...")
        SAM3_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(device)
        SAM3_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
        print(f"Model loaded on {device}")

    inputs = SAM3_PROCESSOR(images=[image], text=[prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = SAM3_MODEL(**inputs)

    results = SAM3_PROCESSOR.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results.get("masks")

    if masks is not None and len(masks) > 0:
        if hasattr(masks, "cpu"):
            masks = masks.cpu().numpy()
        masks = masks.squeeze()
        if masks.ndim == 3:
            combined_mask = np.any(masks > 0, axis=0).astype(np.uint8)
        else:
            assert masks.ndim == 2, f"Got mask shape {masks.shape}"
            combined_mask = (masks > 0).astype(np.uint8)
        mask_img = Image.fromarray(combined_mask * 255, mode="L")
    else:
        mask_img = Image.new("L", target_size[::-1], 0)
    return mask_img


def find_images_in_directory(input_path, extensions=None):
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(input_path).glob(f"*{ext}"))
        image_paths.extend(Path(input_path).glob(f"*{ext.upper()}"))

    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description="Detect objects in datasets using SAM3")
    parser.add_argument("--input-path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output-file", type=str, required=True, help="Output zip file path")
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagefolder",
        choices=["imagefolder", "cifar100", "cub200", "aircraft", "cars"],
        help="Dataset type: imagefolder, cifar100, cub200, aircraft, or cars",
    )
    parser.add_argument(
        "--split", type=str, default="all", choices=["train", "test", "all", "trainval"], help="Dataset split"
    )
    parser.add_argument(
        "--mapping-file", type=str, default=None, help="CSV file mapping class IDs to names and hypernyms"
    )
    parser.add_argument(
        "--custom-prompt", type=str, default=None, help="Custom prompt to use for all classes (overrides mapping)"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold (default: 0.5)")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask threshold (default: 0.5)")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu), auto-detect if not specified"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing (default: 1)")
    parser.add_argument("--print-every", type=int, default=500, help="Print progress every N images (default: 500)")
    parser.add_argument("--min-size", type=int, default=224, help="Min size (longer edge) for upscaling")
    parser.add_argument("--detector", choices=["sam3", "grounded-sam"], default="sam3")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--processes", type=int, default=1)

    args = parser.parse_args()
    assert args.batch_size == 1

    print(f"Segmentation args: {args}")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.dataset == "cifar100":
        if args.mapping_file is None:
            script_dir = Path(__file__).parent
            args.mapping_file = script_dir / "cifar100_classnames.csv"
    elif args.dataset in ("cub200", "aircraft", "cars"):
        # Prompts are derived from class names; no CSV required
        args.mapping_file = None
    else:
        if args.mapping_file is None:
            script_dir = Path(__file__).parent
            args.mapping_file = script_dir / "imagenet_classnames_hypernyms.csv"

    class_mapping = {}
    if args.mapping_file is not None:
        if not os.path.exists(args.mapping_file):
            print(f"Error: Mapping file not found: {args.mapping_file}")
            sys.exit(1)
        print(f"Loading class mapping from: {args.mapping_file}")
        class_mapping = load_class_mapping(args.mapping_file)
        print(f"Loaded {len(class_mapping)} class mappings")

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if args.dataset == "cifar100":
        num_classes = 100
        missing = set(str(i) for i in range(100)) - set(class_mapping.keys())
        if missing:
            print(f"Warning: missing class IDs in mapping: {missing}")
    elif args.dataset == "cub200":
        num_classes = 200
    elif args.dataset == "aircraft":
        num_classes = 100
    elif args.dataset == "cars":
        num_classes = 196
    else:
        class_dirs = [d.name for d in input_path.iterdir() if d.is_dir()]
        num_classes = len(class_dirs)
        missing = set(class_dirs) - set(class_mapping.keys())
        if missing:
            print(f"Warning: missing class IDs in mapping: {missing}")
    print(f"Found {num_classes} classes")

    if args.processes > 0:
        args.output_file = args.output_file.replace(".zip", f"_{args.id}_of_{args.processes}.zip")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    print(f"Writing to {args.output_file}")

    total_images = 0
    start_time = time.time()
    processed_classes = set()

    with zipfile.ZipFile(args.output_file, "w", zipfile.ZIP_STORED) as zipf:
        for image, class_id, image_id in image_iterator(args, id=args.id, processes=args.processes):
            processed_classes.add(class_id)

            if args.custom_prompt:
                prompt = args.custom_prompt
            elif args.dataset == "cub200":
                class_name = cub_class_name_from_folder(class_id)
                prompt = get_prompt(class_name, "bird")
            elif args.dataset == "aircraft":
                class_name = aircraft_class_name_from_variant(class_id)
                prompt = get_prompt(class_name, "aircraft")
            elif args.dataset == "cars":
                prompt = get_prompt(class_id, "car")
            else:
                if class_id not in class_mapping:
                    print(f"Warning: No mapping for {class_id}, skipping")
                    continue
                class_name, hypernym = class_mapping[class_id]
                prompt = get_prompt(class_name, hypernym)

            orig_size = image.size
            was_resized = False

            if max(orig_size) < args.min_size:
                was_resized = True
                if orig_size[0] > orig_size[1]:
                    new_w = args.min_size
                    new_h = int(orig_size[1] * args.min_size / orig_size[0])
                else:
                    new_h = args.min_size
                    new_w = int(orig_size[0] * args.min_size / orig_size[1])
                image = image.resize((new_w, new_h), Image.BILINEAR)
                target_size = (new_h, new_w)
            else:
                target_size = (orig_size[1], orig_size[0])

            try:
                if args.detector == "sam3":
                    mask_img = process_sam3(image, prompt, device, args.threshold, args.mask_threshold, target_size)
                elif args.detector == "grounded-sam":
                    _, detections = grounded_segmentation(
                        image, [prompt], threshold=args.threshold, polygon_refinement=True
                    )
                    if len(detections) == 0:
                        mask_img = Image.new("L", target_size[::-1], 0)
                        # print(f"WARNING: No detection for {class_id}/{image_id}.png")
                    elif len(detections) == 1:
                        mask = detections[0].mask
                        mask_img = Image.fromarray(mask).convert("L")
                    else:
                        mask = np.any([detect.mask > 0 for detect in detections], axis=0)
                        mask_img = Image.fromarray(mask).convert("L")

                else:
                    raise NotImplementedError(f"Unknown detector pipeline {args.detector}")

                if was_resized:
                    mask_img = mask_img.resize(orig_size, Image.NEAREST)

                buf = BytesIO()
                mask_img.save(buf, format="PNG")

                if args.dataset == "cifar100":
                    zipf.writestr(f"{class_id}/{image_id:06d}.png", buf.getvalue())
                elif args.dataset == "cub200":
                    # Strip original extension then save as .png, e.g.
                    # "001.Black_footed_Albatross/img.jpg" -> "…/img.png"
                    stem = os.path.splitext(image_id)[0]
                    zipf.writestr(f"{stem}.png", buf.getvalue())
                elif args.dataset in ("aircraft", "cars"):
                    # image_id is already the stem (no extension), matching
                    # MaskAircraft/MaskCars which looks up "<stem>.png"
                    zipf.writestr(f"{image_id}.png", buf.getvalue())
                else:
                    zipf.writestr(f"{class_id}/{image_id}.png", buf.getvalue())

                total_images += 1

            except Exception as e:
                print(f"Error on {class_id}/{image_id}: {e}")
                raise e

            if total_images % args.print_every == 0:
                elapsed = time.time() - start_time
                rate = total_images / elapsed
                if args.dataset == "cifar100":
                    total = 10000 if args.split == "test" else 50000
                    eta = (total - total_images) / rate
                elif args.dataset == "cub200":
                    eta = (11788 - total_images) / rate
                elif args.dataset == "aircraft":
                    eta = (10000 - total_images) / rate
                elif args.dataset == "cars":
                    eta = (8144 - total_images) / rate
                else:
                    eta = -1
                print(f"[imgs: {total_images}] {elapsed/60:.1f}min elapsed, ETA: {eta/60:.1f}min")

    total_time = time.time() - start_time
    print(f"\nDone! Output written to: {args.output_file}")
    print(f"Processed {total_images} images in {len(processed_classes)} classes in {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
