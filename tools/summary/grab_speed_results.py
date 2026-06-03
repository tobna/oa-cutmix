#!/usr/bin/env python
"""Parse benchmark_mix_speed output and display a summary table.

Usage:
    python tools/summary/grab_speed_results.py results.out
    python tools/summary/grab_speed_results.py results.out --csv speed.csv
    python tools/summary/grab_speed_results.py results.out --no-dedup
"""
import argparse
from pathlib import Path
import re

import pandas as pd


# Maps function name → canonical method prefix.
# A method "owns" a function when method starts with this prefix (and has no '+').
_FUNCTION_PREFIX = {
    "cutmix": "cutmix",
    "mixup": "mixup",
    "fmix": "fmix",
    "resizemix": "resizemix",
    "tokenmix": "tokenmix",
    "smmix": "smmix",
    "mixpro": "mixpro",
    "puzzlemix": "puzzlemix",
    "snapmix": "snapmix",
    "guidedmix": "guidedmix",
    "alignmix": "alignmix",
    "attentivemix": "attentivemix",
    "attentivemix_fga": "attentivemix-fga",
    "saliencymix": "saliencymix",
    "smoothmix": "smoothmix",
    "gridmix": "gridmix",
    "augmix": "augmix",
    "samix": "samix",
    "cutmix_foreground_area": "cutmix-fga",
    "cutmix_stupid": "cutmix-stupid",
    "mask_attentivemix": "mask-attentivemix",
    "mask_mixup": "mask-mixup",
}


_IS_RESNET = re.compile(r"^r\d\d+$")


def extract_architecture(parts) -> str:
    """Extract architecture identifier from config stem split on '_'.

    Examples:
        ['r50', 'cutmix', 'CE']      -> 'r50'
        ['deit', 'ti', 'cutmix']     -> 'deit_ti'
        ['deit', 's', 'mixup']       -> 'deit_s'
    """
    if _IS_RESNET.match(parts[0]):
        return parts[0]
    return "_".join(parts[:2])


def extract_mix_method(parts) -> str:
    """Extract mixup method from config stem split on '_'."""
    is_resnet = _IS_RESNET.match(parts[0]) is not None
    if is_resnet:
        parts = parts[1:]
    else:
        parts = parts[2:]

    if parts and parts[-1].upper() == "CE":
        parts = parts[:-1]

    method = "-".join(parts)
    if method in ("vanilla", "none", ""):
        return "baseline"
    if method == "mixups":
        return "mixup+cutmix"
    return method


def is_canonical(method: str, function: str) -> bool:
    """True when method is a primary config for function.

    Incidental appearances (e.g. cutmix timed inside a tokenmix config, or
    cutmix inside mixup+cutmix) are non-canonical.
    """
    if "+" in method:
        return False
    prefix = _FUNCTION_PREFIX.get(function, function.replace("_", "-"))
    return method == prefix or method.startswith(prefix + "-")


def dedup_incidental(df: pd.DataFrame) -> pd.DataFrame:
    """Drop incidental rows, keeping only canonical ones per function.

    For each unique function, rows whose method is unrelated to the function
    (e.g. cutmix measured inside the tokenmix config) are dropped when at
    least one canonical row exists.  Distinct canonical configs (e.g.
    cutmix-fga-abs-* vs cutmix-fga-rel-*) are all kept.
    """
    groups = []
    for func, grp in df.groupby("function", sort=False):
        canonical = grp[grp["method"].apply(lambda m: is_canonical(m, func))]
        groups.append(canonical if not canonical.empty else grp)
    return pd.concat(groups, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse benchmark_mix_speed output into a table."
    )
    parser.add_argument("file", type=str, help="benchmark_mix_speed output file")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Save results as CSV to this path.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable merging of incidental function measurements.",
    )
    args = parser.parse_args()
    args.file = Path(args.file)
    return args


def main():
    args = parse_args()

    config_re = re.compile(r"^SUMMARY.*config=(.*.py)$")
    data_re = re.compile(
        r"^(\S+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)$"
    )

    results = []
    with open(args.file, "r") as f:
        config = None
        for line in f:
            line = line.strip()
            m = config_re.match(line)
            if m:
                config = m.group(1)
                continue
            m = data_re.match(line)
            if m and config is not None:
                results.append(
                    {
                        "config": config,
                        "method": extract_mix_method(
                            Path(config).stem.split("_")
                        ),
                        "architecture": extract_architecture(
                            Path(config).stem.split("_")
                        ),
                        "function": m.group(1),
                        "mean_ms": float(m.group(2)),
                        "std": float(m.group(3)),
                        "median": float(m.group(4)),
                        "p95": float(m.group(5)),
                        "img/s": int(m.group(6)),
                    }
                )

    df = pd.DataFrame(results)
    if df.empty:
        print("No results found.")
        return

    if not args.no_dedup:
        df = dedup_incidental(df)

    print(df.to_string(index=False))

    if args.csv:
        csv_path = Path(args.csv)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
