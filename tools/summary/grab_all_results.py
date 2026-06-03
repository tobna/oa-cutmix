#!/usr/bin/env python
"""
Grab all model results from a base folder.

Extracts: mixup method, model, dataset, top/final accuracy from experiment logs.

Usage:
    python tools/summary/grab_all_results.py [PATH/to/base_dir] [metric_key]

Example:
    python tools/summary/grab_all_results.py /path/to/experiments head0_top1
"""

import argparse
import functools
import json
import os
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

DEFAULT_EPOCHS = {
    "cifar10": 100,
    "cifar100": 100,
    "imagenet": 100,
    "imnet": 100,
    "svhn": 100,
    "stl10": 200,
    "food101": 100,
    "flowers102": 100,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Grab all model results from a base folder.")
    parser.add_argument("path", type=str, help="Base folder path.")
    parser.add_argument(
        "key",
        type=str,
        nargs="?",
        default="auto",
        help="Metric key to extract (default: auto). Use 'auto' to autodetect metric ending with 'top1'.",
    )
    parser.add_argument("--mode", type=str, default="val", help="Mode to filter by (default: val).")
    parser.add_argument("--max", action="store_true", help="Extract max value instead of final value.")
    parser.add_argument(
        "--expected-epochs",
        type=int,
        default=None,
        help="Override target epochs (overrides auto-detection and dataset defaults).",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file to save results (optional).")
    parser.add_argument(
        "--aggregate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggregate duplicate setups (calculate mean and std).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging.",
    )
    parser.add_argument(
        "--ece",
        action="store_true",
        help="Extract ECE scores from test_calibration_*.log files instead of accuracy from .log.json files.",
    )
    parser.add_argument(
        "--fgsm",
        action="store_true",
        help="Extract FGSM adversarial top1 accuracy from test_fgsm_*.log files.",
    )
    parser.add_argument(
        "--pgd",
        action="store_true",
        help="Extract PGD adversarial top1 accuracy from test_pgd_*.log files.",
    )
    parser.add_argument(
        "--include-methods",
        type=str,
        nargs="*",
        default=None,
        help="Only include methods matching these regex patterns (e.g., 'mixup.*' 'cutmix.*').",
    )
    parser.add_argument(
        "--exclude-methods",
        type=str,
        nargs="*",
        default=None,
        help="Exclude methods matching these regex patterns (e.g., 'baseline' 'vanilla').",
    )
    args = parser.parse_args()
    return args


def _scandir_find(base_path: Path, suffix: str, max_workers: int = 32) -> list[Path]:
    """Recursively find files ending with suffix, scanning directories in parallel.

    Parallel scandir is critical on network filesystems (sshfs) where each
    directory listing is a separate round-trip.
    """
    results = []

    def scan_one(dir_path: str) -> tuple[list[Path], list[str]]:
        files, subdirs = [], []
        try:
            with os.scandir(dir_path) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        subdirs.append(entry.path)
                    elif entry.name.endswith(suffix):
                        files.append(Path(entry.path))
        except PermissionError:
            pass
        return files, subdirs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {executor.submit(scan_one, str(base_path))}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                files, subdirs = fut.result()
                results.extend(files)
                for d in subdirs:
                    pending.add(executor.submit(scan_one, d))

    return sorted(results)


def find_json_logs(base_path: Path) -> list[Path]:
    """Find all .log.json files recursively."""
    return _scandir_find(base_path, ".log.json")


def find_calibration_logs(base_path: Path) -> list[Path]:
    """Find all test_calibration_*.log files recursively."""
    # scandir suffix match is a simple endswith, so filter the prefix separately
    return sorted(f for f in _scandir_find(base_path, ".log") if f.name.startswith("test_calibration_"))


def find_adv_logs(base_path: Path, mode: str) -> list[Path]:
    """Find all test_fgsm_*.log or test_pgd_*.log files recursively."""
    prefix = f"test_{mode}_"
    return sorted(f for f in _scandir_find(base_path, ".log") if f.name.startswith(prefix))


def extract_ece_score(log_file: Path) -> float | None:
    """Extract ECE score from a calibration log file."""
    import re

    try:
        content = log_file.read_text()
        match = re.search(r"ECE score:\s*([\d.]+)%", content)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def extract_adv_score(log_file: Path) -> float | None:
    """Extract adversarial top1 accuracy from an fgsm/pgd log file."""
    try:
        content = log_file.read_text()
        match = re.search(r"\w+_top1:\s*([\d.]+)", content)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def find_log_file(json_file: Path) -> Path | None:
    """Find corresponding .log file with same timestamp prefix.
    Handles both "<name>.json" and "<name>.log.json" naming schemes.
    """
    # If the file ends with ".log.json", strip that suffix to get the base name
    if json_file.name.endswith(".log.json"):
        base_name = json_file.name[: -len(".log.json")]
    else:
        # Fallback to stem (removes the last suffix only)
        base_name = json_file.stem
    possible_log = json_file.parent / f"train_{base_name}.log"
    if possible_log.exists():
        return possible_log
    return None


@functools.lru_cache(maxsize=1024)
def _epochs_from_log_file(log_path: str) -> tuple[int, str] | None:
    """Cached: scan a .log file for max_epochs. Returns (epochs, source) or None."""
    log_file = Path(log_path)
    try:
        with open(log_file, "r") as f:
            for line in f:
                m = re.search(r"max:\s*(\d+)\s*epochs", line)
                if m:
                    return int(m.group(1)), f"from .log file ({log_file.name}) max line"
    except Exception:
        pass
    try:
        with open(log_file, "r") as f:
            for line in f:
                m = re.search(r"runner\s*=\s*dict\([^)]*max_epochs\s*=\s*(\d+)", line)
                if m:
                    return int(m.group(1)), f"from .log file ({log_file.name})"
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=1024)
def _epochs_from_config_dir(dir_path: str) -> tuple[int, str] | None:
    """Cached: scan *.py files in a directory for max_epochs. Returns (epochs, source) or None."""
    for config_file in Path(dir_path).glob("*.py"):
        try:
            content = config_file.read_text()
            match = re.search(r"runner\s*=\s*dict\([^)]*max_epochs\s*=\s*(\d+)", content)
            if match:
                return int(match.group(1)), f"from config ({config_file.name})"
        except Exception:
            continue
    return None


def extract_target_epochs(
    json_file: Path, base_path: Path, dataset: str = "unknown", override: int | None = None, debug: bool = False
) -> tuple[int, str]:
    """Extract target epochs from .log file, config file, folder name, or use dataset default.

    Returns:
        tuple: (target_epochs, source) where source describes where the value came from.
    """
    if override is not None:
        return override, "cli override"

    log_file = find_log_file(json_file)
    if log_file:
        result = _epochs_from_log_file(str(log_file))
        if result is not None:
            return result

    for parent in json_file.parents:
        result = _epochs_from_config_dir(str(parent))
        if result is not None:
            return result

        match = re.search(r"ep(\d+)", parent.name, re.IGNORECASE)
        if match:
            return int(match.group(1)), f"from folder name ({parent.name})"

    default_val = DEFAULT_EPOCHS.get(dataset, 100)
    return default_val, f"dataset default ({dataset}={default_val})"


def extract_metric(
    json_file: Path, key: str, mode: str = "val", get_max: bool = False
) -> tuple[float | None, int, float | None, int]:
    """Extract metric, last epoch, and mean time per epoch from a single json log file.
    If 'key' is "auto", the actual metric key is auto‑detected as any key ending with "top1".
    """
    auto_detect = key == "auto"
    detected_key = None if auto_detect else key
    values = []
    last_epoch = 0
    epoch_times = {}
    epoch_max_iter = {}
    max_memory = 0
    try:
        with open(json_file, "r") as f:
            for line in f:
                log = json.loads(line.strip())
                if auto_detect and detected_key is None and log.get("mode") == mode:
                    for k in log.keys():
                        if k.endswith("top1"):
                            detected_key = k
                            break
                if detected_key and log.get("mode") == mode and detected_key in log:
                    values.append(log[detected_key])
                if "epoch" in log:
                    last_epoch = max(last_epoch, log["epoch"])
                # Track memory usage (GPU memory in MB)
                if "memory" in log:
                    max_memory = max(max_memory, log["memory"])
                # Collect per-epoch iteration times and track max iteration index for scaling
                if log.get("mode") == "train" and "epoch" in log and "time" in log:
                    epoch = log["epoch"]
                    if epoch not in epoch_times:
                        epoch_times[epoch] = []
                        epoch_max_iter[epoch] = 0
                    epoch_times[epoch].append(log["time"])
                    if "iter" in log:
                        epoch_max_iter[epoch] = max(epoch_max_iter[epoch], log["iter"])
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return None, 0, None, 0

    if auto_detect and detected_key is None:
        return None, 0, None, 0
    if not values:
        return None, 0, None, 0

    value = max(values) if get_max else values[-1]

    mean_time = None
    if epoch_times:
        # Scale per-epoch summed times by the ratio of total iterations to logged iterations
        epoch_totals = []
        for epoch, times in epoch_times.items():
            logged_iters = len(times)
            total_iters = epoch_max_iter.get(epoch, logged_iters)
            scale = total_iters / logged_iters if logged_iters else 1
            epoch_total = sum(times) * scale
            epoch_totals.append(epoch_total)
        # Compute average epoch time in seconds, then convert to minutes
        mean_time_seconds = sum(epoch_totals) / len(epoch_totals)
        mean_time = mean_time_seconds / 60.0

    return value, last_epoch, mean_time, max_memory


def extract_info_from_path(json_file: Path) -> dict:
    """Extract mixup method, model, dataset from folder structure."""
    result = {"mix_method": "unknown", "model": "unknown", "dataset": "unknown", "config_name": "unknown"}

    parts = json_file.parts

    path_str = str(json_file).lower()

    result["dataset"] = extract_dataset(path_str)
    result["model"], result["config_name"] = extract_model_and_config(parts, path_str)
    result["mix_method"] = extract_mix_method(result["config_name"].split("_"))

    return result


def extract_dataset(path_str: str) -> str:
    """Extract dataset from path parts."""
    datasets = [
        "cifar100",
        "cifar10",
        "tiny_imagenet_hd",
        "tiny_imagenet",
        "imagenet200",
        "imagenet",
        "imnet",
        "svhn",
        "stl10",
        "food101",
        "flowers102",
        "cub200",
        "aircraft",
        "cars",
    ]

    for ds in datasets:
        if ds in path_str:
            return ds

    return "unknown"


def extract_model_and_config(parts: tuple, path_str: str) -> tuple[str, str]:
    """Extract model name and config name from experiment directory."""
    # Patterns that capture the specific model variant - more specific first
    model_patterns = [
        # DeiT variants
        (r"deit[_-]?ti[_-]?", "deit_ti"),
        (r"deit[_-]?s[_-]?", "deit_s"),
        (r"deit[_-]?t[_-]?", "deit_ti"),
        (r"deit[_-]?3[_-]?", "deit_3"),
        (r"deit[_-]?t[_-]$", "deit_t"),
        (r"deit[_-]?small[_-]?", "deit_s"),
        (r"deit[_-]?base[_-]?", "deit_b"),
        # Generic deit fallback
        (r"deit", "deit"),
        # ViT variants
        (r"vit[_-]?tiny", "vit_tiny"),
        (r"vit[_-]?small", "vit_small"),
        (r"vit[_-]?base", "vit_base"),
        (r"vit", "vit"),
        # ResNet specific variants
        (r"resnet[_-]?18", "resnet18"),
        (r"resnet[_-]?34", "resnet34"),
        (r"resnet[_-]?50", "resnet50"),
        (r"resnet[_-]?101", "resnet101"),
        (r"resnet[_-]?152", "resnet152"),
        # r18, r50 etc. standalone patterns
        (r"r18[_-]", "resnet18"),
        (r"r34[_-]", "resnet34"),
        (r"r50[_-]", "resnet50"),
        (r"r101[_-]", "resnet101"),
        (r"r152[_-]", "resnet152"),
        # rx50, rx101 for ResNeXt
        (r"rx50", "resnext50"),
        (r"rx101", "resnext101"),
        # Generic resnet
        (r"resnet[0-9]+", "resnet"),
        # ConvNeXt variants
        (r"convnext[_-]?tiny", "convnext_tiny"),
        (r"convnext[_-]?small", "convnext_small"),
        (r"convnext[_-]?base", "convnext_base"),
        (r"convnext", "convnext"),
        # Swin variants
        (r"swin[_-]?tiny", "swin_tiny"),
        (r"swin[_-]?small", "swin_small"),
        (r"swin[_-]?base", "swin_base"),
        (r"swin", "swin"),
        # EfficientNet
        (r"efficientnet[_-]?b0", "efficientnet_b0"),
        (r"efficientnet[_-]?b1", "efficientnet_b1"),
        (r"efficientnet[_-]?b2", "efficientnet_b2"),
        (r"efficientnet[_-]?b3", "efficientnet_b3"),
        (r"efficientnet[_-]?b4", "efficientnet_b4"),
        (r"efficientnet", "efficientnet"),
        # MobileNet
        (r"mobilenetv2", "mobilenetv2"),
        (r"mobilenetv3", "mobilenetv3"),
    ]

    exp_parts = [
        p for p in parts if not p.endswith(".log.json") and not re.match(r"test_(calibration|fgsm|pgd)_.*\.log$", p)
    ]
    if re.match(r"^\d{8}_\d{6}", exp_parts[-1]):
        # there is also the timestamp folder
        exp_parts = exp_parts[:-1]
    config_name = exp_parts[-1] if len(exp_parts) > 1 else exp_parts[0] if exp_parts else "unknown"

    for pattern, model_name in model_patterns:
        if re.search(pattern, path_str, re.IGNORECASE):
            return model_name, config_name

    # Exclude mixup-related terms when falling back
    excluded_terms = {"mixups", "mix", "ce", "baseline", "ft", "linear", "full", "ep", "sz", "bs", "lr"}
    for part in exp_parts:
        part_lower = part.lower()
        if re.match(r"^[a-z]+$", part) and len(part) > 2 and part_lower not in excluded_terms:
            return part_lower, config_name

    return "unknown", config_name


def extract_mix_method(parts: tuple) -> str:
    """Extract mixup method from experiment directory or config name.
    Files that have "_mixups" in the config name are interpreted as method "mixup+cutmix".
    """
    is_resnet = re.compile(r"^r\d\d+$").match(parts[0]) is not None
    if is_resnet:
        parts = parts[1:]
    else:
        parts = parts[2:]

    if parts[-1].upper() == "CE":
        parts = parts[:-1]

    method = "-".join(parts)
    if method in ("vanilla", "none", ""):
        return "baseline"
    if method == "mixups":
        return "mixup+cutmix"
    return method


def save_dataframe(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    """Save DataFrame to file."""
    if output_format == "csv":
        df.to_csv(output_path, index=False)
    elif output_format == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif output_format == "excel":
        df.to_excel(output_path, index=False)
    elif output_format == "pickle":
        df.to_pickle(output_path)


def filter_methods(
    df: pd.DataFrame,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter DataFrame by method names using regex patterns.

    Args:
        df: DataFrame with a 'mix_method' column.
        include_patterns: If provided, only keep methods matching at least one pattern.
        exclude_patterns: If provided, exclude methods matching any pattern.

    Returns:
        Filtered DataFrame.
    """
    if df.empty:
        return df

    original_len = len(df)

    if include_patterns:
        # Keep only methods matching at least one include pattern
        mask = df["path"].apply(lambda m: any(re.search(p, m, re.IGNORECASE) for p in include_patterns))
        df = df[mask]
        if original_len > len(df):
            print(f"  Filtered by include-methods: {original_len} -> {len(df)} results")

    if exclude_patterns:
        # Exclude methods matching any exclude pattern
        mask = ~df["path"].apply(lambda m: any(re.search(p, m, re.IGNORECASE) for p in exclude_patterns))
        excluded = original_len - len(df[mask])
        df = df[mask]
        if excluded > 0:
            print(f"  Filtered by exclude-methods: {original_len} -> {len(df)} results")

    return df


def main():
    args = parse_args()
    base_path = Path(args.path)

    if not base_path.exists():
        print(f"Error: Path '{base_path}' does not exist.")
        return

    if args.ece:
        log_files = find_calibration_logs(base_path)
        if not log_files:
            print(f"No test_calibration_*.log files found in '{base_path}'")
            return

        if args.debug:
            print("DEBUG: Detected calibration log files:")
            for f in log_files:
                print(f"  {f}")

        print(f"Found {len(log_files)} calibration log files in '{base_path}'")
        print("Extracting ECE scores")
        print("-" * 100)

        results = []

        for log_file in tqdm(log_files):
            info = extract_info_from_path(log_file)
            ece_score = extract_ece_score(log_file)

            if ece_score is not None:
                results.append(
                    {
                        "mix_method": info["mix_method"],
                        "model": info["model"],
                        "dataset": info["dataset"],
                        "config_name": info["config_name"],
                        "file": log_file.name,
                        "ece": ece_score,
                        "path": str(log_file),
                    }
                )

        df = pd.DataFrame(results)

        if df.empty:
            print("No ECE results found.")
            return

        print(f"Total experiments: {len(df)}")

        # Filter by method patterns
        if args.include_methods or args.exclude_methods:
            df = filter_methods(df, args.include_methods, args.exclude_methods)
            if df.empty:
                print("No results after method filtering.")
                return

        if args.aggregate:
            group_cols = ["mix_method", "model", "dataset", "config_name"]

            aggregated = df.groupby(group_cols, as_index=False).agg(
                ece_mean=("ece", "mean"),
                ece_std=("ece", "std"),
                n_runs=("ece", "count"),
            )

            final_df = aggregated
        else:
            final_df = df

        sort_columns = ["dataset", "model", "ece_mean"]
        present_cols = [c for c in sort_columns if c in final_df.columns]
        if present_cols:
            final_df = final_df.sort_values(by=present_cols).reset_index(drop=True)

        print("\n" + "=" * 100)
        print("Results (mean \\pm std):")
        print("=" * 100)
        print(final_df.to_string(index=False))

        print("-" * 100)
        print(f"Total: {len(final_df)} results")

        if args.output:
            output_path = Path(args.output)
            output_format = output_path.suffix.lstrip(".")
            if not isinstance(final_df, pd.DataFrame):
                final_df = pd.DataFrame(final_df)
            save_dataframe(df=final_df, output_path=output_path, output_format=output_format)
            print(f"Results saved to: {output_path}")

        return

    if args.fgsm or args.pgd:
        modes = [m for m, flag in [("fgsm", args.fgsm), ("pgd", args.pgd)] if flag]
        results = []

        for mode in modes:
            log_files = find_adv_logs(base_path, mode)
            if not log_files:
                print(f"No test_{mode}_*.log files found in '{base_path}'")
                continue

            if args.debug:
                print(f"DEBUG: Detected {mode.upper()} log files:")
                for f in log_files:
                    print(f"  {f}")

            print(f"Found {len(log_files)} {mode.upper()} log files in '{base_path}'")

            if args.include_methods or args.exclude_methods:
                before = len(log_files)
                if args.include_methods:
                    log_files = [
                        f for f in log_files if any(re.search(p, str(f), re.IGNORECASE) for p in args.include_methods)
                    ]
                if args.exclude_methods:
                    log_files = [
                        f
                        for f in log_files
                        if not any(re.search(p, str(f), re.IGNORECASE) for p in args.exclude_methods)
                    ]
                print(f"After path filtering: {before} -> {len(log_files)} files")

            print(f"Extracting {mode.upper()} adversarial top1 accuracy")
            print("-" * 100)

            for log_file in tqdm(log_files, desc=mode.upper()):
                info = extract_info_from_path(log_file)
                top1 = extract_adv_score(log_file)

                if top1 is not None:
                    results.append(
                        {
                            "mode": mode,
                            "mix_method": info["mix_method"],
                            "model": info["model"],
                            "dataset": info["dataset"],
                            "config_name": info["config_name"],
                            "file": log_file.name,
                            "top1": top1,
                            "error": round(100.0 - top1, 3),
                            "path": str(log_file),
                        }
                    )

        df = pd.DataFrame(results)

        if df.empty:
            print("No adversarial robustness results found.")
            return

        print(f"Total: {len(df)}")

        if args.aggregate:
            group_cols = ["mode", "mix_method", "model", "dataset", "config_name"]
            final_df = df.groupby(group_cols, as_index=False).agg(
                top1_mean=("top1", "mean"),
                top1_std=("top1", "std"),
                error_mean=("error", "mean"),
                error_std=("error", "std"),
                n_runs=("top1", "count"),
            )
        else:
            final_df = df

        sort_columns = ["mode", "dataset", "model", "error_mean" if args.aggregate else "error"]
        present_cols = [c for c in sort_columns if c in final_df.columns]
        if present_cols:
            final_df = final_df.sort_values(by=present_cols).reset_index(drop=True)

        print("\n" + "=" * 100)
        print("Adversarial robustness results (mean \\pm std):")
        print("=" * 100)
        print(final_df.to_string(index=False))
        print("-" * 100)
        print(f"Total: {len(final_df)} results")

        if args.output:
            output_path = Path(args.output)
            output_format = output_path.suffix.lstrip(".")
            if not isinstance(final_df, pd.DataFrame):
                final_df = pd.DataFrame(final_df)
            save_dataframe(df=final_df, output_path=output_path, output_format=output_format)
            print(f"Results saved to: {output_path}")

        return

    json_files = find_json_logs(base_path)

    if not json_files:
        print(f"No .log.json files found in '{base_path}'")
        return

    if args.debug:
        # Print detected files and their containing folders
        print("DEBUG: Detected .log.json files:")
        for f in json_files:
            print(f"  {f}")
        folder_set = {str(p.parent) for p in json_files}
        print("DEBUG: Containing folders:")
        for folder in sorted(folder_set):
            print(f"  {folder}")

    print(f"Found {len(json_files)} log files in '{base_path}'")

    # Pre-filter by path patterns before expensive per-file processing
    if args.include_methods or args.exclude_methods:
        before = len(json_files)
        if args.include_methods:
            json_files = [
                f for f in json_files if any(re.search(p, str(f), re.IGNORECASE) for p in args.include_methods)
            ]
        if args.exclude_methods:
            json_files = [
                f for f in json_files if not any(re.search(p, str(f), re.IGNORECASE) for p in args.exclude_methods)
            ]
        print(f"After path filtering: {before} -> {len(json_files)} files")

    print(f"Extracting metric: {args.key} (mode={args.mode}, max={args.max})")
    if args.expected_epochs:
        print(f"Using manual target epochs: {args.expected_epochs}")
    else:
        print(f"Dataset-specific default epochs: {DEFAULT_EPOCHS}")
    print("-" * 100)

    results = []
    incomplete_info = []
    auto_epochs = 0
    manual_epochs = 0

    def _process_file(json_file: Path) -> dict:
        info = extract_info_from_path(json_file)
        value, last_epoch, mean_time, max_mem = extract_metric(json_file, args.key, args.mode, args.max)
        target_epochs, epoch_source = extract_target_epochs(
            json_file, base_path, info["dataset"], args.expected_epochs, args.debug
        )
        return dict(
            info=info,
            value=value,
            last_epoch=last_epoch,
            mean_time=mean_time,
            max_mem=max_mem,
            target_epochs=target_epochs,
            epoch_source=epoch_source,
            json_file=json_file,
        )

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_process_file, f): f for f in json_files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            r = future.result()
            json_file = r["json_file"]
            info = r["info"]
            value = r["value"]
            last_epoch = r["last_epoch"]
            mean_time = r["mean_time"]
            max_mem = r["max_mem"]
            target_epochs = r["target_epochs"]
            epoch_source = r["epoch_source"]

            if args.debug:
                print(f"DEBUG: Processing file {json_file.name} (folder: {json_file.parent})")
                if value is None:
                    print(f"   ERROR: Could not extract metric '{args.key}'")
                else:
                    print(f"   metric={value}, last_epoch={last_epoch}, target_epochs={target_epochs} ({epoch_source})")

            is_complete = (value is not None) and (last_epoch >= target_epochs)
            if is_complete:
                results.append(
                    {
                        "mix_method": info["mix_method"],
                        "model": info["model"],
                        "dataset": info["dataset"],
                        "config_name": info["config_name"],
                        "file": json_file.name,
                        "value": value,
                        "mean_time_per_epoch": mean_time,
                        "max_memory": max_mem,
                        "last_epoch": last_epoch,
                        "target_epochs": target_epochs,
                        "path": str(json_file),
                    }
                )
            if not is_complete:
                incomplete_info.append(
                    {
                        "file": str(json_file.relative_to(base_path)),
                        "mix_method": info["mix_method"],
                        "model": info["model"],
                        "dataset": info["dataset"],
                        "last_epoch": last_epoch,
                        "target_epochs": target_epochs,
                        "epoch_source": epoch_source,
                    }
                )

    # if args.debug:
    #     print("\n" + "=" * 100)
    #     print("DEBUG: All experiments processed")
    #     print("=" * 100)
    #     for r in results:
    #         status = "COMPLETE" if r["complete"] else "INCOMPLETE"
    #         status = "COMPLETE"
    #         print(f"  [{status}] {r['file']}")
    #         print(f"           mix_method={r['mix_method']}, model={r['model']}, dataset={r['dataset']}")
    #         # Handle possible None value gracefully
    #         if r["value"] is None:
    #             value_str = "N/A"
    #         else:
    #             value_str = f"{r['value']:.2f}"
    #         print(
    #             f"           value={value_str}, last_epoch={r['last_epoch']}, target_epochs={r['target_epochs']}"
    #             f" ({r['epoch_source']})"
    #         )
    #         if not r["complete"]:
    #             print(f"           >>> INCOMPLETE: expected {r['target_epochs']} epochs, got {r['last_epoch']}")

    df = pd.DataFrame(results)

    if df.empty:
        print("No results found.")
        return

    print(f"Complete experiments: {len(df)}")
    print(f"Incomplete experiments: {len(incomplete_info)}")
    print(f"Target epochs - Auto-detected/from config: {manual_epochs}, Dataset defaults: {auto_epochs}")

    # Filter by method patterns
    if args.include_methods or args.exclude_methods:
        df = filter_methods(df, args.include_methods, args.exclude_methods)
        if df.empty:
            print("No results after method filtering.")
            return

    if args.debug and incomplete_info:
        print("\n" + "=" * 100)
        print(f"DEBUG: Incomplete experiments ({len(incomplete_info)})")
        print("=" * 100)
        for inc in incomplete_info:
            print(f"  {inc['file']}")
            print(f"    mix_method={inc['mix_method']}, model={inc['model']}, dataset={inc['dataset']}")
            print(
                f"    >>> INCOMPLETE: last_epoch={inc['last_epoch']}, target_epochs={inc['target_epochs']}"
                f" ({inc['epoch_source']})"
            )

    # Use all experiments (complete and incomplete) for aggregation/display
    # Preserve original complete flag for user reference
    all_df = df.copy()

    if all_df.empty:
        print("No experiments found.")
        return

    if args.aggregate:
        group_cols = ["mix_method", "model", "dataset", "config_name"]

        aggregated = all_df.groupby(group_cols, as_index=False).agg(
            value_mean=("value", "mean"),
            value_std=("value", "std"),
            n_runs=("value", "count"),
            mean_time_per_epoch=("mean_time_per_epoch", "mean"),
            mean_time_per_epoch_std=("mean_time_per_epoch", "std"),
            max_memory=("max_memory", "mean"),
            max_memory_std=("max_memory", "std"),
        )

        final_df = aggregated
    else:
        final_df = all_df

    # Rename metric column(s) to more descriptive names
    if "value" in final_df.columns:
        final_df = final_df.rename({"value": "acc"}, axis=1)
    if "value_mean" in final_df.columns:
        final_df = final_df.rename({"value_mean": "acc_mean"}, axis=1)
    if "value_std" in final_df.columns:
        final_df = final_df.rename({"value_std": "acc_std"}, axis=1)

    # Sort results by the subset of desired columns that actually exist in the DataFrame
    sort_columns = ["dataset", "model", "acc", "acc_mean"]
    present_cols = [c for c in sort_columns if c in final_df.columns]
    if present_cols:
        final_df = final_df.sort_values(by=present_cols).reset_index(drop=True)  # pyright: ignore

    print("\n" + "=" * 100)
    print("Results (mean \u00b1 std):")
    print("=" * 100)
    print(final_df.to_string(index=False))

    print("-" * 100)
    print(f"Total: {len(final_df)} results")

    if args.output:
        output_path = Path(args.output)
        output_format = output_path.suffix.lstrip(".")
        # Ensure final_df is a DataFrame for type safety
        if not isinstance(final_df, pd.DataFrame):
            final_df = pd.DataFrame(final_df)
        save_dataframe(df=final_df, output_path=output_path, output_format=output_format)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
