#!/usr/bin/env python
"""
Convert grab_all_results output to LaTeX tables for time and memory.

Usage:
    python tools/summary/results_to_latex_time_mem.py <input_file> [--output-time <time.tex>] [--output-mem <mem.tex>]

Generates separate tables for time per epoch and memory usage.
"""

import argparse
import re
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert grab_all_results to LaTeX table for time/memory.")
    parser.add_argument("input", type=str, help="Input file (csv, json, excel).")
    parser.add_argument("--output-time", "-t", type=str, default=None, help="Output .tex file for time.")
    parser.add_argument("--output-mem", "-m", type=str, default=None, help="Output .tex file for memory.")
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Datasets to exclude.",
    )
    parser.add_argument(
        "--include-only",
        type=str,
        nargs="*",
        default=None,
        help="Only include these datasets.",
    )
    parser.add_argument(
        "--no-booktabs",
        action="store_true",
        help="Don't use booktabs.",
    )
    args = parser.parse_args()
    return args


def format_model_name(model_name):
    """Format model name for LaTeX: remove underscores, capitalize properly.
    Examples: resnet_18 -> ResNet18, deit_tiny -> DeiT-Ti, resnet50 -> ResNet50
    """
    name = model_name.replace("_", " ").replace("-", " ")
    parts = name.split()
    
    if not parts:
        return model_name
    
    formatted_parts = []
    for part in parts:
        if part.lower() in ["resnet", "resnext", "deit", "vit", "swin", "convnext", "efficientnet", "mobilenet"]:
            if part.lower() == "deit":
                formatted_parts.append("DeiT")
            else:
                capitalized = part.capitalize() if part.islower() else part
                formatted_parts.append(capitalized)
        elif part.isdigit():
            formatted_parts.append(part)
        elif part.lower() in ["ti", "tiny", "t", "s", "small", "m", "medium", "b", "base", "l", "large"]:
            size_map = {
                "ti": "-Ti", "tiny": "-Tiny", "t": "-T", "s": "-S",
                "small": "-S", "m": "-M", "medium": "-M", "b": "-B",
                "base": "-B", "l": "-L", "large": "-L"
            }
            formatted_parts.append(size_map.get(part.lower(), "-" + part.capitalize()))
        else:
            formatted_parts.append(part.capitalize())
    
    result = "".join(formatted_parts)
    
    result = result.replace("Resnet", "ResNet")
    result = result.replace("Resnext", "ResNeXt")
    result = result.replace("Convnext", "ConvNeXt")
    result = result.replace("Efficientnet", "EfficientNet")
    result = result.replace("Mobilenet", "MobileNet")
    
    return result


def format_dataset_name(dataset_name):
    """Format dataset name for LaTeX: remove underscores, capitalize properly.
    Examples: cifar_100 -> CIFAR100, tiny_imagenet -> TinyImageNet
    """
    name = dataset_name.replace("_", " ").replace("-", " ")
    parts = name.split()
    
    if not parts:
        return dataset_name
    
    known_datasets = {
        "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
        "tinyimagenet": "TinyImageNet", "imagenet": "ImageNet",
        "stl10": "STL-10", "mnist": "MNIST",
        "cifar": "CIFAR", "svhn": "SVHN"
    }
    
    lower_name = name.lower().replace("-", "")
    if lower_name in known_datasets:
        return known_datasets[lower_name]
    
    if "hd" in lower_name and "tiny" in lower_name:
        return "TinyImageNet-HD"
    elif "tiny" in lower_name and "imagenet" in lower_name:
        return "TinyImageNet"
    else:
        formatted = "".join(p.capitalize() for p in parts)
        formatted = formatted.replace("Cifar", "CIFAR")
        formatted = formatted.replace("Stl10", "STL-10")
    
    return formatted


def parse_model_size(model_name):
    """Parse model name to extract base type and size for sorting."""
    model_name = model_name.lower()
    
    size_map = {
        'ti': 0, 'tiny': 0,
        't': 1,
        's': 2, 'small': 2,
        'm': 3, 'medium': 3,
        'b': 4, 'base': 4,
        'l': 5, 'large': 5,
    }
    
    num_match = re.search(r'(\d+)', model_name)
    if num_match:
        size = int(num_match.group(1))
    else:
        size = None
        for kw, val in size_map.items():
            if kw in model_name:
                size = val
                break
        if size is None:
            size = 0
    
    base_types = ['resnet', 'resnext', 'deit', 'vit', 'swin', 'convnext', 'efficientnet', 'mobilenet']
    base = None
    for bt in base_types:
        if bt in model_name:
            base = bt
            break
    
    if base is None:
        base = model_name.split('_')[0] if '_' in model_name else model_name
    
    return base, size


def sort_models(models):
    """Sort models by base type alphabetically, then by size."""
    return sorted(models, key=lambda m: (parse_model_size(m)[0], parse_model_size(m)[1]))


def format_cell(mean_val, std_val, is_best=False, is_second=False, unit=""):
    """Format a cell with mean ± std or just mean, with best/second highlighted."""
    if pd.isna(mean_val):
        return ""
    
    if std_val is not None and not pd.isna(std_val) and std_val != 0:
        formatted = f"{mean_val:.2f} \\pm {std_val:.2f}"
    else:
        formatted = f"{mean_val:.2f}"
    
    if unit:
        formatted += f" {unit}"
    
    if is_best:
        formatted = f"\\mathbf{{{formatted}}}"
    elif is_second:
        formatted = f"\\underline{{{formatted}}}"
    
    return formatted


def build_table(df, datasets, methods, col_name, unit="", highlight_best=False, no_booktabs=False):
    """Build a LaTeX table for a specific metric column."""
    
    # Build column format
    col_format = "l"
    header1 = ["Method"]
    header2 = ["\\multirow{2.5}{*}"]
    
    for ds in datasets:
        ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
        if not ds_models:
            continue
        col_format += "c" * len(ds_models)
        header1.append(f"\\multicolumn{{{len(ds_models)}}}{{c}}{{\\textbf{{{format_dataset_name(ds)}}}}}")
        header2.extend([f"\\textbf{{{format_model_name(m)}}}" for m in ds_models])

    # Pre-compute best and second best for each column
    best_second = {}
    if highlight_best:
        for ds_idx, ds in enumerate(datasets):
            ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
            for m_idx, m in enumerate(ds_models):
                col_idx = 1 + sum(len(sort_models(df[df["dataset"] == d]["model"].unique().tolist())) 
                                  for d in datasets[:ds_idx]) + m_idx
                
                col_values = []
                for method_idx, method in enumerate(methods):
                    subset = df[(df["dataset"] == ds) & (df["mix_method"] == method) & (df["model"] == m)]
                    if not subset.empty:
                        val = subset[col_name].values[0]
                        if not pd.isna(val):
                            col_values.append((method_idx, val))
                
                if len(col_values) >= 2:
                    # For time: lower is better, for memory: lower is better
                    sorted_values = sorted(col_values, key=lambda x: x[1])
                    best_method_idx = sorted_values[0][0]
                    second_method_idx = sorted_values[1][0]
                    best_second[(best_method_idx, col_idx)] = (True, False)
                    best_second[(second_method_idx, col_idx)] = (False, True)
                elif len(col_values) == 1:
                    best_method_idx = col_values[0][0]
                    best_second[(best_method_idx, col_idx)] = (True, False)

    # Build data rows
    latex_rows = []
    for row_idx, method in enumerate(methods):
        row_data = [method]
        col_idx = 0
        for ds in datasets:
            ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
            for m in ds_models:
                col_idx += 1
                subset = df[(df["dataset"] == ds) & (df["mix_method"] == method) & (df["model"] == m)]
                if subset.empty:
                    row_data.append("")
                else:
                    mean_val = subset[col_name].values[0]
                    std_val = subset.get(f"{col_name}_std", None)
                    if std_val is not None:
                        std_val = std_val.values[0] if len(std_val) > 0 else None
                    else:
                        std_val = None
                    is_best, is_second = best_second.get((row_idx, col_idx), (False, False))
                    row_data.append(format_cell(mean_val, std_val, is_best, is_second, unit))

        latex_rows.append(" & ".join(row_data) + " \\\\")

    # Build full LaTeX table
    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    
    if no_booktabs:
        lines.append("\\hline")
    else:
        lines.append("\\toprule")
    
    lines.append(" & ".join(header1) + " \\\\")
    
    # Build clines/cmidrules for each dataset group
    col_positions = []
    current_col = 2
    for ds in datasets:
        ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
        if ds_models:
            col_positions.append((current_col, current_col + len(ds_models) - 1))
            current_col += len(ds_models)
    
    if no_booktabs:
        for start, end in col_positions:
            lines.append(f"\\cline{{{start}-{end}}}")
    else:
        for start, end in col_positions:
            lines.append(f"\\cmidrule(lr){{{start}-{end}}}")
    
    lines.append(" & ".join(header2) +" \\\\")
    
    if no_booktabs:
        lines.append("\\hline")
    else:
        lines.append("\\midrule")
    
    lines.extend(latex_rows)
    
    if no_booktabs:
        lines.append("\\hline")
    else:
        lines.append("\\bottomrule")
    
    lines.append("\\end{tabular}")
    
    return "\n".join(lines)


def main():
    args = parse_args()
    input_path = Path(args.input)

    # Load data
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix == ".json":
        df = pd.read_json(input_path)
    elif input_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        print(f"Error: Unsupported file format {input_path.suffix}")
        return

    # Filter datasets
    if args.include_only:
        df = df[df["dataset"].isin(args.include_only)]
    for ds in args.exclude:
        df = df[df["dataset"] != ds]

    if df.empty:
        print("No data after filtering.")
        return

    # Log which models are found per dataset
    print("Datasets and their models found in the data:")
    for ds in sorted(df["dataset"].unique()):
        ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
        print(f"  {ds}: {ds_models}")
    print()

    # Get sorted datasets and methods
    datasets = sorted(df["dataset"].unique())
    methods = sorted(df["mix_method"].unique())

    # Check available columns
    print("Available columns:", list(df.columns))
    print()

    # Build time table
    time_col = "mean_time_per_epoch"
    if time_col in df.columns:
        time_table = build_table(
            df, datasets, methods, 
            col_name=time_col, 
            unit="min",  # time is in minutes
            highlight_best=True,
            no_booktabs=args.no_booktabs
        )
        if args.output_time:
            Path(args.output_time).write_text(time_table)
            print(f"Time table saved to {args.output_time}")
        else:
            print("Time per epoch table:")
            print(time_table)
    else:
        print(f"Column '{time_col}' not found in data")

    # Build memory table
    mem_col = "max_memory"
    if mem_col in df.columns:
        mem_table = build_table(
            df, datasets, methods,
            col_name=mem_col,
            unit="MB",
            highlight_best=True,
            no_booktabs=args.no_booktabs
        )
        if args.output_mem:
            Path(args.output_mem).write_text(mem_table)
            print(f"Memory table saved to {args.output_mem}")
        else:
            print("\nMax memory table:")
            print(mem_table)
    else:
        print(f"Column '{mem_col}' not found in data")


if __name__ == "__main__":
    main()
