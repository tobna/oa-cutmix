#!/usr/bin/env python
"""
Convert grab_all_results output to a LaTeX table.

Usage:
    python tools/summary/results_to_latex.py <input_file> [--output <output.tex>] [--mean-rank]

The output table has method as the first column, then for each dataset a group of
columns with model names as subheaders. Uses multicolumn for dataset headers.
Use --mean-rank to add a column showing the mean rank across all dataset-model
combinations (1=best, lower is better).
"""

import argparse
import re
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert grab_all_results to LaTeX table.")
    parser.add_argument("input", type=str, help="Input file (csv, json, excel).")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .tex file.")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="method",
        choices=["method", "mean-rank", "mean-acc"],
        help="Sort order for rows.",
    )
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
    parser.add_argument(
        "--mean-rank",
        action="store_true",
        help="Add a column showing mean rank across all dataset-model combinations (1=best).",
    )
    parser.add_argument(
        "--mean-acc",
        action="store_true",
        help=(
            "Add a column showing mean accuracy across all dataset-model combinations (only for methods with complete"
            " data)."
        ),
    )
    parser.add_argument(
        "--ece",
        action="store_true",
        help="Treat values as ECE scores (lower is better, instead of accuracy where higher is better).",
    )
    parser.add_argument("--std", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--speed-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to CSV produced by grab_speed_results.py. Adds a speed column matched by the 'method' column.",
    )
    parser.add_argument(
        "--speed-metric",
        type=str,
        default="ms",
        choices=["img/s", "ms"],
        help="Speed metric to show: 'img/s' (throughput) or 'ms' (mean latency). Default: img/s.",
    )
    parser.add_argument(
        "--speed-arch",
        type=str,
        default=None,
        metavar="FAMILY",
        help=(
            "Architecture family to prefer when matching speed rows "
            "(e.g. 'resnet', 'deit'). Auto-detected from results data "
            "when omitted; falls back to other architectures if the "
            "preferred one has no entry for a method."
        ),
    )
    parser.add_argument(
        "--group-dynamic",
        action="store_true",
        help=(
            "Group methods into dynamic (high-latency) and static "
            "(low-latency) sections. Requires --speed-csv. Dynamic "
            "methods appear first, each group sorted by mean accuracy."
        ),
    )
    parser.add_argument(
        "--dynamic-threshold",
        type=float,
        default=10.0,
        metavar="MS",
        help=(
            "Latency threshold (ms) to classify a method as dynamic. "
            "Methods with latency >= this value are considered dynamic. "
            "Default: 10.0"
        ),
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
                "ti": "-Ti",
                "tiny": "-Tiny",
                "t": "-T",
                "s": "-S",
                "small": "-S",
                "m": "-M",
                "medium": "-M",
                "b": "-B",
                "base": "-B",
                "l": "-L",
                "large": "-L",
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
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "tinyimagenet": "TinyImageNet",
        "imagenet": "ImageNet",
        "imagenet200": "ImageNet200",
        "stl10": "STL-10",
        "mnist": "MNIST",
        "cifar": "CIFAR",
        "svhn": "SVHN",
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
    """Parse model name to extract base type and size for sorting.
    Returns (base_type, size_int) where size_int is the model size number.
    """
    model_name = model_name.lower()

    # Extract number from model name (e.g., resnet18 -> 18, deit_s -> 0 (small), deit_b -> 1 (base))
    size_map = {
        "ti": 0,
        "tiny": 0,
        "t": 1,  # deit_t
        "s": 2,
        "small": 2,
        "m": 3,
        "medium": 3,
        "b": 4,
        "base": 4,
        "l": 5,
        "large": 5,
    }

    # Try to find a number in the name
    num_match = re.search(r"(\d+)", model_name)
    if num_match:
        size = int(num_match.group(1))
    else:
        # Try to find size keyword
        size = None
        for kw, val in size_map.items():
            if kw in model_name:
                size = val
                break
        if size is None:
            size = 0  # default

    # Extract base type
    base_types = ["resnet", "resnext", "deit", "vit", "swin", "convnext", "efficientnet", "mobilenet"]
    base = None
    for bt in base_types:
        if bt in model_name:
            base = bt
            break

    if base is None:
        # Fallback: use first word
        base = model_name.split("_")[0] if "_" in model_name else model_name

    return base, size


def sort_models(models):
    """Sort models by base type alphabetically, then by size."""
    return sorted(models, key=lambda m: (parse_model_size(m)[0], parse_model_size(m)[1]))


def format_cell(mean_val, std_val, is_best=False, is_second=False, is_ece=False, include_std=True):
    """Format a cell with mean ± std or just mean, with best/second highlighted.
    For ECE, lower is better; for accuracy, higher is better.
    """
    if pd.isna(mean_val):
        return ""

    # Format the value
    if pd.isna(std_val) or std_val == 0 or std_val is None or not include_std:
        formatted = f"{mean_val:.2f}"
    else:
        formatted = f"{mean_val:.2f} \\pm {std_val:.2f}"

    # Apply highlighting
    if is_best:
        formatted = f"\\mathbf{{{formatted}}}"
    elif is_second:
        formatted = f"\\underline{{{formatted}}}"

    return "$" + formatted + "$"


def get_mean_val(subset, is_ece=False):
    """Extract mean value from a dataframe subset."""
    if subset.empty:
        return None
    if is_ece and "ece_mean" in subset.columns:
        return subset["ece_mean"].values[0]
    if "acc_mean" in subset.columns:
        return subset["acc_mean"].values[0]
    if "ece_mean" in subset.columns:
        return subset["ece_mean"].values[0]
    return subset["value"].values[0]


def get_std_val(subset, is_ece=False):
    """Extract std value from a dataframe subset."""
    if subset.empty:
        return None
    if is_ece and "ece_std" in subset.columns:
        return subset["ece_std"].values[0]
    if "acc_std" in subset.columns:
        return subset["acc_std"].values[0]
    if "ece_std" in subset.columns:
        return subset["ece_std"].values[0]
    if "value_std" in subset.columns:
        return subset["value_std"].values[0]
    return None


def arch_family(arch: str) -> str:
    """Map an architecture identifier to its family name.

    Examples: 'r50' -> 'resnet', 'r18' -> 'resnet', 'deit_ti' -> 'deit'.
    """
    if re.match(r"^r\d+$", arch.lower()):
        return "resnet"
    return arch.lower().split("_")[0]


def detect_arch_family(df: pd.DataFrame) -> str | None:
    """Infer dominant architecture family from the 'model' column."""
    from collections import Counter

    families = []
    for model in df["model"].unique():
        m = model.lower().replace("-", "_")
        if re.match(r"^r\d+$", m) or "resnet" in m:
            families.append("resnet")
        elif "deit" in m:
            families.append("deit")
        elif "vit" in m:
            families.append("vit")
        elif "swin" in m:
            families.append("swin")
        else:
            families.append(m.split("_")[0])
    if not families:
        return None
    return Counter(families).most_common(1)[0][0]


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

    # Load speed data if provided
    speed_lookup = {}  # method -> value (img/s or mean_ms)
    if args.speed_csv:
        speed_df = pd.read_csv(args.speed_csv)
        use_ms = args.speed_metric == "ms"
        has_arch_col = "architecture" in speed_df.columns

        target_family = args.speed_arch
        if target_family is None and has_arch_col:
            target_family = detect_arch_family(df)
            if target_family:
                print(
                    f"Auto-detected architecture family for speed matching: "
                    f"'{target_family}'"
                )

        for method, grp in speed_df.groupby("method"):
            if has_arch_col and target_family:
                preferred = grp[
                    grp["architecture"].apply(arch_family) == target_family
                ]
                matched = preferred if not preferred.empty else grp
                if preferred.empty:
                    fallback_arch = grp["architecture"].iloc[0]
                    print(
                        f"  Speed fallback for '{method}': no '{target_family}' "
                        f"entry, using '{fallback_arch}'"
                    )
            else:
                matched = grp

            if use_ms:
                speed_lookup[method] = float(matched["mean_ms"].min())
            else:
                speed_lookup[method] = int(matched["img/s"].max())
        # mixup+cutmix has no direct speed entry; synthesise as the mean
        # of the two component methods (already resolved per architecture).
        if "mixup+cutmix" not in speed_lookup:
            m_val = speed_lookup.get("mixup")
            c_val = speed_lookup.get("cutmix")
            if m_val is not None and c_val is not None:
                if use_ms:
                    speed_lookup["mixup+cutmix"] = (m_val + c_val) / 2
                else:
                    speed_lookup["mixup+cutmix"] = int((m_val + c_val) / 2)
        print(f"Loaded speed data for {len(speed_lookup)} methods from {args.speed_csv}")

    # Filter datasets
    if args.include_only:
        df = df[df["dataset"].isin(args.include_only)]
    for ds in args.exclude:
        df = df[df["dataset"] != ds]

    if df.empty:
        print("No data after filtering.")
        return

    # Log which models are found for each dataset (sorted)
    print("Datasets and their models found in the data:")
    for ds in sorted(df["dataset"].unique()):
        ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
        print(f"  {ds}: {ds_models}")
    print()

    # Get sorted unique datasets and models
    datasets = sorted(df["dataset"].unique())
    models = sort_models(df["model"].unique().tolist())

    # Get sorted methods
    methods = sorted(df["mix_method"].unique())

    # Pre-compute mean rank and mean accuracy for each method if corresponding flags are set or sorting requested
    # This is needed to sort by mean rank or mean accuracy
    # For ECE, lower is better; for accuracy, higher is better
    reverse_sort = not args.ece

    row_means = {}
    mean_acc_means = {}
    total_combos = sum(len(sort_models(df[df["dataset"] == ds]["model"].unique().tolist())) for ds in datasets)
    # Compute mean rank (existing logic)
    if args.mean_rank or args.sort_by == "mean-rank":
        col_rank_data = {}
        col_idx = 0
        for ds in datasets:
            ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
            for m in ds_models:
                col_idx += 1
                method_vals = []
                for method in methods:
                    subset = df[(df["dataset"] == ds) & (df["mix_method"] == method) & (df["model"] == m)]
                    if not subset.empty:
                        mean_val = get_mean_val(subset, args.ece)
                        if mean_val is not None and not pd.isna(mean_val):
                            method_vals.append((method, mean_val))
                if len(method_vals) >= 2:
                    sorted_vals = sorted(method_vals, key=lambda x: x[1], reverse=reverse_sort)
                    for rank, (method, _) in enumerate(sorted_vals, 1):
                        if method not in col_rank_data:
                            col_rank_data[method] = []
                        col_rank_data[method].append((rank - 1) / (len(method_vals) - 1))

        for method in methods:
            if method in col_rank_data and col_rank_data[method]:
                row_means[method] = sum(col_rank_data[method]) / len(col_rank_data[method]) * (len(methods) - 1) + 1
            else:
                row_means[method] = None

        # Sort methods by mean rank descending (best = lowest rank at bottom)
        if args.sort_by == "mean-rank" and row_means:

            def sort_key(m):
                rank = row_means.get(m)
                if rank is None:
                    return (False, float("inf"))  # None values at top
                return (True, -rank)  # Higher ranks (worse) at top, lower ranks (better) at bottom

            methods = sorted(methods, key=sort_key)
    # Compute mean accuracy across all dataset-model combos for methods with full data
    if args.mean_acc or args.sort_by == "mean-acc" or args.group_dynamic:
        # Gather values per method
        for method in methods:
            vals = []
            for ds in datasets:
                ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
                for m in ds_models:
                    subset = df[(df["dataset"] == ds) & (df["mix_method"] == method) & (df["model"] == m)]
                    if not subset.empty:
                        mean_val = get_mean_val(subset, args.ece)
                        if mean_val is not None and not pd.isna(mean_val):
                            vals.append(mean_val)
            if len(vals) == total_combos:
                mean_acc_means[method] = sum(vals) / len(vals)
            else:
                mean_acc_means[method] = None
        # Sort methods by mean accuracy descending (higher is better)
        if args.sort_by == "mean-acc" and mean_acc_means:

            def sort_key_acc(m):
                val = mean_acc_means.get(m)
                if val is None:
                    return float("inf" if args.ece else "-inf")
                return val

            methods = sorted(methods, key=sort_key_acc, reverse=args.ece)

    # Group into dynamic / static if requested
    dynamic_separator_idx = None
    if args.group_dynamic and speed_lookup:
        dynamic_methods = []
        static_methods = []
        for m in methods:
            lat = speed_lookup.get(m)
            if lat is not None and lat >= args.dynamic_threshold:
                dynamic_methods.append(m)
            else:
                static_methods.append(m)

        def _sort_group_by_mean_acc(m_list):
            def key(m):
                val = mean_acc_means.get(m)
                if val is None:
                    return float("inf" if args.ece else "-inf")
                return val
            return sorted(m_list, key=key, reverse=args.ece)

        dynamic_methods = _sort_group_by_mean_acc(dynamic_methods)
        static_methods = _sort_group_by_mean_acc(static_methods)
        methods = dynamic_methods + static_methods
        if dynamic_methods and static_methods:
            dynamic_separator_idx = len(dynamic_methods)

    rows = methods

    # Build column format and header
    col_format = "l"
    header1 = ["\\multirow{2.5}{*}{Method}"]
    header2 = [""]

    for ds in datasets:
        ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
        if not ds_models:
            continue
        col_format += "c" * len(ds_models)
        # Multicolumn for dataset name
        header1.append(f"\\multicolumn{{{len(ds_models)}}}{{c}}{{{format_dataset_name(ds)}}}")
        header2.extend([f"{format_model_name(m)}" for m in ds_models])

    if args.mean_rank or args.sort_by == "mean-rank":
        col_format += "c"
        header1.append("\\multirow{2.5}{*}{\\makecell{Mean\\\\Rank}}")
        header2.append("")
    if args.mean_acc or args.sort_by == "mean-acc":
        col_format += "c"
        header1.append("\\multirow{2.5}{*}{Mean}")
        header2.append("")
    if speed_lookup:
        col_format += "r"
        label = "latency" if args.speed_metric == "ms" else "throughput"
        header1.append(f"\\multirow{{1.5}}{{*}}{{{label}}}")
        header2.append(f"[{args.speed_metric}]")

    # Pre-compute best and second best for each column (model/dataset combination)
    # In normal mode: each column is a (dataset, model) combination
    best_second = {}  # key: (row_idx, col_idx), value: (is_best, is_second)

    # Find best and second best for each (dataset, model) column
    for ds_idx, ds in enumerate(datasets):
        ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
        for m_idx, m in enumerate(ds_models):
            col_idx = (
                1
                + sum(len(sort_models(df[df["dataset"] == d]["model"].unique().tolist())) for d in datasets[:ds_idx])
                + m_idx
            )

            # Get all values for this dataset-model combination
            col_values = []
            for method_idx, method in enumerate(methods):
                subset = df[(df["dataset"] == ds) & (df["mix_method"] == method) & (df["model"] == m)]
                if not subset.empty:
                    mean_val = get_mean_val(subset, args.ece)
                    if not pd.isna(mean_val):
                        col_values.append((method_idx, mean_val))

            if len(col_values) >= 2:
                sorted_values = sorted(col_values, key=lambda x: x[1], reverse=reverse_sort)
                best_method_idx = sorted_values[0][0]
                second_method_idx = sorted_values[1][0]
                best_second[(best_method_idx, col_idx)] = (True, False)
                best_second[(second_method_idx, col_idx)] = (False, True)
            elif len(col_values) == 1:
                best_method_idx = col_values[0][0]
                best_second[(best_method_idx, col_idx)] = (True, False)

    # Build data rows
    latex_rows = []

    # Find best and second best by mean rank (lower is better)
    if (args.mean_rank or args.sort_by == "mean-rank") and row_means:
        valid_means = [(name, val) for name, val in row_means.items() if val is not None]
        if len(valid_means) >= 2:
            sorted_means = sorted(valid_means, key=lambda x: x[1], reverse=args.ece)
            best_mean_rank = sorted_means[0][0]
            second_mean_rank = sorted_means[1][0]
        elif len(valid_means) == 1:
            best_mean_rank = valid_means[0][0]
            second_mean_rank = None
        else:
            best_mean_rank = second_mean_rank = None
    else:
        best_mean_rank = second_mean_rank = None

    if (args.mean_acc or args.sort_by == "mean-acc") and mean_acc_means:
        valid_means = [(name, val) for name, val in mean_acc_means.items() if val is not None]
        if len(valid_means) >= 2:
            sorted_means = sorted(valid_means, key=lambda x: x[1], reverse=not args.ece)
            best_mean_acc = sorted_means[0][0]
            second_mean_acc = sorted_means[1][0]
        elif len(valid_means) == 1:
            best_mean_acc = valid_means[0][0]
            second_mean_acc = None
        else:
            best_mean_acc = second_mean_acc = None
    else:
        best_mean_acc = second_mean_acc = None

    for row_idx, row in enumerate(rows):
        if dynamic_separator_idx is not None and row_idx == dynamic_separator_idx:
            latex_rows.append("\\midrule")
        method = row
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
                    mean_val = get_mean_val(subset, args.ece)
                    std_val = get_std_val(subset, args.ece)
                    is_best, is_second = best_second.get((row_idx, col_idx), (False, False))
                    row_data.append(format_cell(mean_val, std_val, is_best, is_second, args.ece, include_std=args.std))

        if args.mean_rank or args.sort_by == "mean-rank":
            mean_val = row_means.get(method)
            is_best = method == best_mean_rank
            is_second = method == second_mean_rank
            if mean_val is not None:
                row_data.append(format_cell(mean_val, None, is_best, is_second, include_std=args.std))
            else:
                row_data.append("")

        if args.mean_acc or args.sort_by == "mean-acc":
            mean_val = mean_acc_means.get(method)
            is_best = method == best_mean_acc
            is_second = method == second_mean_acc
            if mean_val is not None:
                row_data.append(format_cell(mean_val, None, is_best, is_second, include_std=args.std))
            else:
                row_data.append("")

        if speed_lookup:
            val = speed_lookup.get(method)
            if val is None:
                row_data.append("")
            elif args.speed_metric == "ms":
                row_data.append(f"{val:.2f}")
            else:
                row_data.append(f"{val:,}")

        latex_rows.append(" & ".join(row_data) + " \\\\")

    # Build full LaTeX table
    lines = []

    lines.append(f"\\begin{{tabular}}{{{col_format}}}")

    if args.no_booktabs:
        lines.append("\\hline")
    else:
        lines.append("\\toprule")

    # Header row 1: Method + Dataset multicolumns
    lines.append(" & ".join(header1) + " \\\\")

    # Build cmidrules for each dataset group
    col_positions = []
    current_col = 2  # Start after Method column
    for ds in datasets:
        ds_models = sort_models(df[df["dataset"] == ds]["model"].unique().tolist())
        if ds_models:
            col_positions.append((current_col, current_col + len(ds_models) - 1))
            current_col += len(ds_models)

    if args.no_booktabs:
        for start, end in col_positions:
            lines.append(f"\\cline{{{start}-{end}}}")
    else:
        for start, end in col_positions:
            lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    # Header row 2: Model names
    lines.append(" & ".join(header2) + " \\\\")

    if args.no_booktabs:
        lines.append("\\hline")
    else:
        lines.append("\\midrule")

    # Data rows
    lines.extend(latex_rows)

    if args.no_booktabs:
        lines.append("\\hline")
    else:
        lines.append("\\bottomrule")

    lines.append("\\end{tabular}")

    latex_content = "\n".join(lines)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(latex_content)
        print(f"LaTeX table saved to {output_path}")
    else:
        print(latex_content)


if __name__ == "__main__":
    main()
