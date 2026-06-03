"""Plot mean label error vs prediction accuracy and correct-class score.

For each prediction file, bins samples into 20 equal-size bins by mean
label error and computes per-bin accuracy and mean correct-class probability.
Produces separate curves for 'image as A', 'image as B', and their mean.
Also plots the accuracy/score improvement between paired model variants.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_and_align(label_err_path, pred_path):
    """Load label errors and predictions, align by index."""
    le = np.load(label_err_path)
    pr = np.load(pred_path)

    # Build index → row mapping for label errors
    le_idx_to_row = {idx: i for i, idx in enumerate(le["indices"])}

    # Select only samples present in both files
    mask = np.array([idx in le_idx_to_row for idx in pr["indices"]])
    pr_rows = np.where(mask)[0]
    le_rows = np.array([le_idx_to_row[pr["indices"][i]] for i in pr_rows])

    err_a = le["label_err_abs_a_mean"][le_rows]
    err_b = le["label_err_abs_b_mean"][le_rows]
    err_mean = (err_a + err_b) / 2.0
    fg_frac = le["fg_frac"][le_rows]

    is_correct = pr["is_correct"][pr_rows]
    correct_prob = pr["correct_prob"][pr_rows]

    # Return original indices so two models can be aligned to each other
    orig_indices = pr["indices"][pr_rows]
    return err_a, err_b, err_mean, fg_frac, is_correct, correct_prob, orig_indices


def bin_stats(error, is_correct, correct_prob, n_bins=20):
    """Divide into n_bins equal-size bins by error; return bin centres,
    accuracy, mean correct-class probability, and their std deviations."""
    order = np.argsort(error)
    bins = np.array_split(order, n_bins)

    centres, accs, scores, acc_stds, score_stds = [], [], [], [], []
    for b in bins:
        centres.append(error[b].mean())
        accs.append(is_correct[b].mean() * 100)
        scores.append(correct_prob[b].mean() * 100)
        n = len(b)
        acc_stds.append(is_correct[b].std() * 100)
        score_stds.append(correct_prob[b].std() * 100)

    return (
        np.array(centres),
        np.array(accs),
        np.array(scores),
        np.array(acc_stds),
        np.array(score_stds),
    )


def process_pred(label_err_path, pred_path, n_bins=20):
    err_a, err_b, err_mean, fg_frac, is_correct, correct_prob, indices = load_and_align(label_err_path, pred_path)
    results = {}
    for role, err in [
        ("Image as A", err_a),
        ("Image as B", err_b),
        ("Mean A+B", err_mean),
    ]:
        c, acc, score, acc_std, score_std = bin_stats(err, is_correct, correct_prob, n_bins)
        results[role] = (c, acc, score, acc_std, score_std)

    results_fg = {"_All": bin_stats(fg_frac, is_correct, correct_prob, n_bins)}

    # Also return per-sample data keyed by index for diff computation
    per_sample = {
        "err_a": err_a,
        "err_b": err_b,
        "err_mean": err_mean,
        "fg_frac": fg_frac,
        "is_correct": is_correct,
        "correct_prob": correct_prob,
        "indices": indices,
    }
    return results, results_fg, per_sample


def compute_diff(base_ps, fga_ps, n_bins=20):
    """Align two models by sample index, bin by label error, compute diff."""
    # Intersect indices
    base_idx_map = {idx: i for i, idx in enumerate(base_ps["indices"])}
    common_mask = np.array([idx in base_idx_map for idx in fga_ps["indices"]])
    fga_rows = np.where(common_mask)[0]
    base_rows = np.array([base_idx_map[fga_ps["indices"][i]] for i in fga_rows])

    acc_diff = fga_ps["is_correct"][fga_rows].astype(float) - base_ps["is_correct"][base_rows].astype(float)
    score_diff = fga_ps["correct_prob"][fga_rows] - base_ps["correct_prob"][base_rows]

    results = {}
    for role, err_key in [
        ("Image as A", "err_a"),
        ("Image as B", "err_b"),
        # ("Mean A+B", "err_mean"),
    ]:
        err = base_ps[err_key][base_rows]
        c, acc_d, score_d, acc_d_std, score_d_std = bin_stats(err, acc_diff, score_diff, n_bins)
        results[role] = (c, acc_d, score_d, acc_d_std, score_d_std)

    fg = base_ps["fg_frac"][base_rows]
    c, acc_d, score_d, acc_d_std, score_d_std = bin_stats(fg, acc_diff, score_diff, n_bins)
    results_fg = {"_All": (c, acc_d, score_d, acc_d_std, score_d_std)}

    return results, results_fg


def plot_panel(
    ax, centres_list, values_list, labels, ylabel, stds_list=None, xlabel="Mean label error (abs)", legend=True
):
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, (centres, values, label, color) in enumerate(zip(centres_list, values_list, labels, colors)):
        ax.plot(centres, values, marker="o", ms=4, label=label, color=color)
        if stds_list is not None:
            s = stds_list[i]
            ax.fill_between(centres, values - s, values + s, color=color, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def make_figure(results, pred_name, bands=True, xlabel="Mean label error (abs)", legend=True):
    labels = list(results.keys())
    centres_list = [results[l][0] for l in labels]
    acc_list = [results[l][1] for l in labels]
    score_list = [results[l][2] for l in labels]
    acc_std_list = [results[l][3] for l in labels] if bands else None
    score_std_list = [results[l][4] for l in labels] if bands else None

    fig_acc, ax_acc = plt.subplots(figsize=(3, 2))
    plot_panel(ax_acc, centres_list, acc_list, labels, "Accuracy", stds_list=acc_std_list, xlabel=xlabel, legend=legend)
    fig_acc.tight_layout(pad=0.1)

    fig_score, ax_score = plt.subplots(figsize=(3, 2))
    plot_panel(
        ax_score,
        centres_list,
        score_list,
        labels,
        "Mean correct-class prob.",
        stds_list=score_std_list,
        xlabel=xlabel,
        legend=legend,
    )
    fig_score.tight_layout(pad=0.1)

    return fig_acc, fig_score


def make_diff_figure(diff_results, base_name, fga_name, bands=True, xlabel="Mean label error (abs)", legend=True):
    labels = list(diff_results.keys())
    centres_list = [diff_results[l][0] for l in labels]
    acc_diff_list = [diff_results[l][1] for l in labels]
    score_diff_list = [diff_results[l][2] for l in labels]
    acc_std_list = [diff_results[l][3] for l in labels] if bands else None
    score_std_list = [diff_results[l][4] for l in labels] if bands else None

    fig_acc, ax_acc = plt.subplots(figsize=(2.8, 1.6))
    plot_panel(
        ax_acc,
        centres_list,
        acc_diff_list,
        labels,
        "Accuracy\nimprovement",
        stds_list=acc_std_list,
        xlabel=xlabel,
        legend=legend,
    )
    fig_acc.tight_layout(pad=0.0)

    fig_score, ax_score = plt.subplots(figsize=(2.5, 1.5))
    plot_panel(
        ax_score,
        centres_list,
        score_diff_list,
        labels,
        "Score\nimprovement",
        stds_list=score_std_list,
        xlabel=xlabel,
        legend=legend,
    )
    fig_score.tight_layout(pad=0.1)

    return fig_acc, fig_score


ROW_LABELS = ["DeiT-S", "ResNet-50"]
COL_LABELS = ["CIFAR-100", "TinyImageNet", "Aircraft", "Cars", "CUB-200"]


def detect_model_row(stem):
    """Return row index (0=DeiT-S, 1=ResNet-50) or None."""
    sl = stem.lower()
    if any(k in sl for k in ["deit_small", "deit-s", "deits", "deit_s", "deit_tiny"]):
        return 0
    if any(k in sl for k in ["resnet50", "r50", "res50", "resnet_50"]):
        return 1
    return None


def detect_dataset_col(stem):
    """Return col index (0=CIFAR100, 1=TinyImageNet, 2=Aircraft, 3=Cars) or None."""
    sl = stem.lower()
    if any(k in sl for k in ["cifar100", "cifar-100", "cifar_100"]):
        return 0
    if any(k in sl for k in ["tinyimagenet", "tiny_imagenet", "tiny-imagenet"]):
        return 1
    if any(k in sl for k in ["aircraft", "fgvc"]):
        return 2
    if any(k in sl for k in ["cars", "stanford"]):
        return 3
    # if any(k in sl for k in ["cub200", "cub_200", "cub-200", "cub2011", "cub_2011"]):
    #     return 4
    return None


def make_grid_diff_figure(grid_data, y_key, x_key, bands=True):
    """Create a 2×4 grid of diff plots.

    Parameters
    ----------
    grid_data : dict
        Keys are (row, col) tuples (row 0=DeiT-S, 1=ResNet-50;
        col 0-3 for CIFAR-100/TinyImageNet/Aircraft/Cars).
        Values are dicts with keys "diff_results" and "diff_results_fg".
    y_key : str
        "acc" for accuracy improvement, "score" for score improvement.
    x_key : str
        "label_err" to bin by mean label error,
        "obj_size" to bin by relative object size.
    bands : bool
        Whether to draw shaded std bands.
    """
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 3))
    colors = ["tab:blue", "tab:orange"]
    legend_added = False

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            cell = grid_data.get((row, col))

            if cell is None:
                ax.set_visible(False)
                continue

            if x_key == "label_err":
                diff_res = cell["diff_results"]
                xlabel = "Mean label error (abs)"
            else:
                diff_res = cell["diff_results_fg"]
                xlabel = "Relative object size"

            for i, (label, entry) in enumerate(diff_res.items()):
                c, acc_d, score_d, acc_std, score_std = entry
                values = acc_d if y_key == "acc" else score_d
                stds = acc_std if y_key == "acc" else score_std
                color = colors[i % len(colors)]
                # Only add legend labels once (top-left occupied cell)
                lbl = label if (not legend_added) else None
                ax.plot(c, values, marker="o", ms=3, label=lbl, color=color)
                if bands:
                    ax.fill_between(c, values - stds, values + stds, color=color, alpha=0.15)

            # ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # x-axis label only on bottom row
            if row == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            else:
                ax.tick_params(labelbottom=False)

            # y-axis label only on leftmost column
            if col == 0:
                ylabel = "Accuracy improvement" if y_key == "acc" else "Score improvement"
                ax.set_ylabel(ylabel, fontsize=8)

            # Column title on top row
            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=12)

            # Row label as rotated text to the left of the leftmost column
            if col == 0:
                ax.annotate(
                    ROW_LABELS[row],
                    xy=(0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-0.38, 0.5),
                    textcoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=12,
                    rotation=90,
                    annotation_clip=False,
                )

            if not legend_added and len(diff_res) > 1:
                ax.legend(fontsize=6, loc="best")
                legend_added = True

    fig.tight_layout(pad=0.0, w_pad=0.5, h_pad=0.5)
    return fig


def compute_both(base_ps, fga_ps, n_bins=20):
    """Align two models by sample index, bin by label error/fg_frac,
    return separate bin stats for base and fga."""
    base_idx_map = {idx: i for i, idx in enumerate(base_ps["indices"])}
    common_mask = np.array([idx in base_idx_map for idx in fga_ps["indices"]])
    fga_rows = np.where(common_mask)[0]
    base_rows = np.array([base_idx_map[fga_ps["indices"][i]] for i in fga_rows])

    base_correct = base_ps["is_correct"][base_rows]
    base_prob = base_ps["correct_prob"][base_rows]
    fga_correct = fga_ps["is_correct"][fga_rows]
    fga_prob = fga_ps["correct_prob"][fga_rows]

    base_results, fga_results = {}, {}
    for role, err_key in [
        ("Image as A", "err_a"),
        ("Image as B", "err_b"),
    ]:
        err = base_ps[err_key][base_rows]
        c, a, s, as_, ss = bin_stats(err, base_correct, base_prob, n_bins)
        base_results[role] = (c, a, s, as_, ss)
        c, a, s, as_, ss = bin_stats(err, fga_correct, fga_prob, n_bins)
        fga_results[role] = (c, a, s, as_, ss)

    fg = base_ps["fg_frac"][base_rows]
    c, a, s, as_, ss = bin_stats(fg, base_correct, base_prob, n_bins)
    base_results_fg = {"_All": (c, a, s, as_, ss)}
    c, a, s, as_, ss = bin_stats(fg, fga_correct, fga_prob, n_bins)
    fga_results_fg = {"_All": (c, a, s, as_, ss)}

    return base_results, fga_results, base_results_fg, fga_results_fg


def make_grid_both_figure(grid_data, y_key, x_key, bands=True):
    """Create a 2×4 grid plotting CutMix and OA-CutMix lines side-by-side.

    Legend is placed only in the top-right subplot.
    """
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 3))
    colors = {"CutMix": "tab:blue", "OA-CutMix": "tab:orange"}

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            cell = grid_data.get((row, col))

            if cell is None or "base_results" not in cell:
                ax.set_visible(False)
                continue

            if x_key == "label_err":
                base_res = cell["base_results"]
                fga_res = cell["fga_results"]
                xlabel = "Mean label error (abs)"
            else:
                base_res = cell["base_results_fg"]
                fga_res = cell["fga_results_fg"]
                xlabel = "Relative object size"

            show_legend = row == 0 and col == n_cols - 1

            for model_label, res in [
                ("CutMix", base_res),
                ("OA-CutMix", fga_res),
            ]:
                color = colors[model_label]
                for i, (role, entry) in enumerate(res.items()):
                    c, acc, score, acc_std, score_std = entry
                    values = acc if y_key == "acc" else score
                    stds = acc_std if y_key == "acc" else score_std
                    lbl = model_label if i == 0 else None
                    ax.plot(
                        c,
                        values,
                        marker="o",
                        ms=3,
                        label=lbl,
                        color=color,
                        alpha=1.0 if i == 0 else 0.5,
                        ls="-" if i == 0 else "--",
                    )
                    if bands:
                        ax.fill_between(
                            c,
                            values - stds,
                            values + stds,
                            color=color,
                            alpha=0.12,
                        )

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            if row == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            else:
                ax.tick_params(labelbottom=False)

            if col == 0:
                ylabel = "Accuracy (%)" if y_key == "acc" else "Mean correct-class prob. (%)"
                ax.set_ylabel(ylabel, fontsize=8)

            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=12)

            if col == 0:
                ax.annotate(
                    ROW_LABELS[row],
                    xy=(0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-0.38, 0.5),
                    textcoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=12,
                    rotation=90,
                    annotation_clip=False,
                )

            if show_legend:
                ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout(pad=0.0, w_pad=0.5, h_pad=0.5)
    return fig


def find_base_fga_pairs(pred_files):
    """Return list of (base_path, fga_path) pairs.

    Pairs a file without '_fga' with files that share its stem as a prefix
    and contain '_fga'.
    """
    pairs = []
    base_files = [p for p in pred_files if "_fga" not in p.stem]
    fga_files = [p for p in pred_files if "_fga" in p.stem]
    for base in base_files:
        for fga in fga_files:
            if fga.stem.startswith(base.stem):
                pairs.append((base, fga))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Plot label error vs accuracy/score.")
    parser.add_argument(
        "--pred_dir",
        default="work_dirs/predictions",
        help="Directory containing .npz files",
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--no-bands", action="store_true", help="Disable shaded error bands")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    label_err_files = sorted(pred_dir.glob("*_label_errors.npz"))
    if not label_err_files:
        print("No *_label_errors.npz files found.")
        return

    # Collect per-(row, col) diff data for the 2×4 grid figures
    grid_data = {}

    for le_path in label_err_files:
        dataset = le_path.stem.replace("_label_errors", "")
        pred_files = sorted(pred_dir.glob(f"*{dataset}*.npz"))
        pred_files = [p for p in pred_files if "_label_errors" not in p.name]

        if not pred_files:
            print(f"No prediction files found for dataset '{dataset}'")
            continue

        # Per-model plots
        per_sample_cache = {}
        for pred_path in pred_files:
            print(f"Processing: {pred_path.name}")
            try:
                results, results_fg, per_sample = process_pred(le_path, pred_path, args.n_bins)
            except Exception as e:
                print(f"  Error: {e}")
                continue

            per_sample_cache[pred_path] = per_sample

            bands = not args.no_bands
            for res, xlabel, suffix, legend in [
                # (results, "Mean label error (abs)", "", True),
                (results_fg, "Relative object size", "_vs_objsize", False),
            ]:
                fig_acc, fig_score = make_figure(res, pred_path.stem, bands=bands, xlabel=xlabel, legend=legend)
                out_acc = out_dir / f"plot_{pred_path.stem}_accuracy{suffix}.pdf"
                out_score = out_dir / f"plot_{pred_path.stem}_score{suffix}.pdf"
                fig_acc.savefig(out_acc, bbox_inches="tight")
                fig_score.savefig(out_score, bbox_inches="tight")
                print(f"  Saved: {out_acc}")
                print(f"  Saved: {out_score}")
                plt.close(fig_acc)
                plt.close(fig_score)

        # Diff plots for base/fga pairs
        pairs = find_base_fga_pairs(pred_files)
        for base_path, fga_path in pairs:
            if base_path not in per_sample_cache:
                continue
            if fga_path not in per_sample_cache:
                continue
            print(f"Diff: {fga_path.name} − {base_path.name}")
            diff_results, diff_results_fg = compute_diff(
                per_sample_cache[base_path],
                per_sample_cache[fga_path],
                args.n_bins,
            )
            stem = f"plot_diff_{fga_path.stem}_vs_{base_path.stem}"
            bands = not args.no_bands
            for diff_res, xlabel, suffix, legend in [
                # (diff_results, "Mean label error (abs)", "", True),
                (diff_results_fg, "Relative object size", "_vs_objsize", False),
            ]:
                fig_acc, fig_score = make_diff_figure(
                    diff_res, base_path.stem, fga_path.stem, bands=bands, xlabel=xlabel, legend=legend
                )
                out_acc = out_dir / f"{stem}_accuracy{suffix}.pdf"
                out_score = out_dir / f"{stem}_score{suffix}.pdf"
                fig_acc.savefig(out_acc, dpi=150, bbox_inches="tight")
                fig_score.savefig(out_score, dpi=150, bbox_inches="tight")
                print(f"  Saved: {out_acc}")
                print(f"  Saved: {out_score}")
                plt.close(fig_acc)
                plt.close(fig_score)

            # Accumulate data for the 2×4 grid
            row = detect_model_row(fga_path.stem)
            col = detect_dataset_col(fga_path.stem)
            if row is not None and col is not None:
                base_res, fga_res, base_res_fg, fga_res_fg = compute_both(
                    per_sample_cache[base_path],
                    per_sample_cache[fga_path],
                    args.n_bins,
                )
                grid_data[(row, col)] = {
                    "diff_results": diff_results,
                    "diff_results_fg": diff_results_fg,
                    "base_results": base_res,
                    "fga_results": fga_res,
                    "base_results_fg": base_res_fg,
                    "fga_results_fg": fga_res_fg,
                }

    # 2×4 grid figures (one per y/x combination)
    if grid_data:
        bands = not args.no_bands
        for y_key, y_label in [
            ("acc", "accuracy"),
            ("score", "score"),
        ]:
            for x_key, x_suffix in [
                # ("label_err", "vs_label_err"),
                ("obj_size", "vs_obj_size"),
            ]:
                fig = make_grid_diff_figure(grid_data, y_key=y_key, x_key=x_key, bands=bands)
                out = out_dir / f"grid_diff_{y_label}_{x_suffix}.pdf"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved grid: {out}")
                plt.close(fig)

                fig = make_grid_both_figure(grid_data, y_key=y_key, x_key=x_key, bands=bands)
                out = out_dir / f"grid_both_{y_label}_{x_suffix}.pdf"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved grid: {out}")
                plt.close(fig)


if __name__ == "__main__":
    main()
