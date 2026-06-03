#!/usr/bin/env python3
"""
Analyze CutMix example outputs.

For each sample in each pair, computes the absolute label error:
    |lam_area - lam_fg|

This measures how much the area-based mixing coefficient (used as the
soft label in standard CutMix) deviates from the foreground-aware lambda.

Output is sorted by absolute label error (descending).

Usage:
    python tools/analyze_cutmix_examples.py \
        /fscratch/nauen/openmixup_work_dirs/cutmix_examples
"""

import argparse
import json
import os
from pathlib import Path


CLASSNAMES_FILE = Path(__file__).parent / "prepare_data" / "imagenet_classnames.txt"


def load_classnames(path=CLASSNAMES_FILE):
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            synset, name = line.split(",", 1)
            mapping[synset.strip()] = name.strip()
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Analyze CutMix example label errors")
    parser.add_argument("out_dir", help="Root output directory from cutmix_examples.py")
    parser.add_argument(
        "--sort",
        choices=["error", "lam_area", "lam_fg"],
        default="error",
        help="Sort key (default: absolute label error, descending)",
    )
    args = parser.parse_args()

    classnames = load_classnames()

    # Collect records grouped by pair
    pairs = {}  # pair_name -> list of sample records

    for pair_name in sorted(os.listdir(args.out_dir)):
        pair_dir = os.path.join(args.out_dir, pair_name)
        pair_meta_path = os.path.join(pair_dir, "metadata.json")
        if not os.path.isdir(pair_dir) or not os.path.isfile(pair_meta_path):
            continue

        with open(pair_meta_path) as f:
            pair_meta = json.load(f)

        synset_a = pair_meta.get("class_a", "")
        synset_b = pair_meta.get("class_b", "")
        class_a = classnames.get(synset_a, synset_a)
        class_b = classnames.get(synset_b, synset_b)
        label_a = pair_meta["label_a"]
        label_b = pair_meta["label_b"]

        samples = []
        for sample_name in sorted(os.listdir(pair_dir)):
            sample_dir = os.path.join(pair_dir, sample_name)
            sample_meta_path = os.path.join(sample_dir, "metadata.json")
            if not os.path.isdir(sample_dir) or not os.path.isfile(sample_meta_path):
                continue

            with open(sample_meta_path) as f:
                sample_meta = json.load(f)

            lam_area = sample_meta["lam_area"]
            lam_fg = sample_meta["lam_fg"]
            abs_error = abs(lam_area - lam_fg)

            samples.append(dict(
                sample=sample_name,
                lam_area=lam_area,
                lam_fg=lam_fg,
                abs_error=abs_error,
            ))

        pairs[pair_name] = dict(
            class_a=class_a,
            class_b=class_b,
            label_a=label_a,
            label_b=label_b,
            samples=samples,
        )

    # Sort pairs by mean of top-3 error samples (descending)
    def pair_sort_key(item):
        samples = item[1]["samples"]
        top3 = sorted(samples, key=lambda s: s["abs_error"], reverse=True)[:3]
        return sum(s["abs_error"] for s in top3) / len(top3) if top3 else 0.0

    sorted_pairs = sorted(pairs.items(), key=pair_sort_key, reverse=True)

    # Column widths
    col = dict(sample=9, lam_area=9, lam_fg=9, abs_error=10)
    row_width = (
        col["sample"] + 1 + col["lam_area"] + 1 + col["lam_fg"] + 1 + col["abs_error"]
    )

    all_errors = []
    for pair_name, pdata in sorted_pairs:
        samples = pdata["samples"]
        top3_errors = sorted(
            [s["abs_error"] for s in samples], reverse=True
        )[:3]
        pair_top3_mean = sum(top3_errors) / len(top3_errors) if top3_errors else 0.0
        all_errors.extend(s["abs_error"] for s in samples)

        print("=" * row_width)
        print(
            f"{pair_name}  A: {pdata['class_a']} (cls {pdata['label_a']})  "
            f"B: {pdata['class_b']} (cls {pdata['label_b']})  "
            f"[top-3 mean |err|: {pair_top3_mean:.4f}]"
        )
        print("-" * row_width)
        header = (
            f"{'sample':<{col['sample']}} "
            f"{'lam_area':>{col['lam_area']}} "
            f"{'lam_fg':>{col['lam_fg']}} "
            f"{'|err|':>{col['abs_error']}}"
        )
        print(header)
        print("-" * row_width)

        for s in sorted(samples, key=lambda x: x["abs_error"], reverse=True):
            print(
                f"{s['sample']:<{col['sample']}} "
                f"{s['lam_area']:>{col['lam_area']}.4f} "
                f"{s['lam_fg']:>{col['lam_fg']}.4f} "
                f"{s['abs_error']:>{col['abs_error']}.4f}"
            )

    print("=" * row_width)
    print(f"\nTotal samples: {len(all_errors)}")
    if all_errors:
        print(f"Mean |error|: {sum(all_errors)/len(all_errors):.4f}")
        print(f"Max  |error|: {max(all_errors):.4f}")
        print(f"Min  |error|: {min(all_errors):.4f}")


if __name__ == "__main__":
    main()
