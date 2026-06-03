#!/usr/bin/env python3
"""
Analyze a CutMix statistics CSV produced by cutmix_collect.py.

Outputs:
  - Console summary table (mean / std / min / max / median for key metrics)
  - plots/01_lambda_distributions.png   histograms of λ_area, λ_fg, Δλ
  - plots/02_lambda_scatter.png         scatter λ_area vs λ_fg
  - plots/03_fg_pixels.png             distribution of fg pixel counts
  - plots/04_fg_visible.png            fg pixels visible per source in mix
  - plots/05_lam_vs_fg_pixels.png      λ_area and λ_fg vs fg pixel counts
  - plots/06_per_class_fg.png          per-class mean fg pixel count (top-N classes)

Usage:
    python tools/cutmix_analyze.py cutmix_stats.csv [--out-dir plots] [--top-k 20]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load(path):
    df = pd.read_csv(path)
    df["delta_lam"] = (df["lam_area"] - df["lam_fg"]).abs()
    df["fg_frac_a"] = df["fg_pixels_a_total"] / df["total_area"]
    df["fg_frac_b"] = df["fg_pixels_b_total"] / df["total_area"]
    df["fg_vis_frac_a"] = df["fg_visible_a_in_mix"] / df["total_area"]
    df["fg_vis_frac_b"] = df["fg_visible_b_in_mix"] / df["total_area"]
    df["cut_frac"] = df["cut_area"] / df["total_area"]
    # fg pixels of A inside the bbox (cut away from A)
    df["fg_a_inside_bbox"] = df["fg_pixels_a_total"] - df["fg_visible_a_in_mix"]
    # fg pixels of B outside the bbox (not pasted into the mix)
    df["fg_b_outside_bbox"] = df["fg_pixels_b_total"] - df["fg_visible_b_in_mix"]
    # object size: mean fg pixels across both images in the pair
    df["object_size"] = (df["fg_pixels_a_total"] + df["fg_pixels_b_total"]) / 2.0
    # KL divergence between Bernoulli(λ_area) and Bernoulli(λ_fg)
    eps = 1e-7
    p = df["lam_area"].clip(eps, 1 - eps)
    q = df["lam_fg"].clip(eps, 1 - eps)
    df["kl_area_fg"] = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return df


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# 1. Console summary
# ---------------------------------------------------------------------------


def print_summary(df):
    cols = {
        "lam_area": "λ_area  (standard CutMix)",
        "lam_fg": "λ_fg    (foreground-based)",
        "delta_lam": "|λ_area − λ_fg|",
        "fg_pixels_a_total": "FG pixels orig A",
        "fg_pixels_b_total": "FG pixels orig B",
        "fg_visible_a_in_mix": "FG from A visible in mix",
        "fg_visible_b_in_mix": "FG from B visible in mix",
        "fg_frac_a": "FG fraction orig A",
        "fg_frac_b": "FG fraction orig B",
        "cut_frac": "Cut fraction of image",
    }
    print("\n" + "=" * 78)
    print(f"SUMMARY  (n={len(df):,} samples   img size={int(df['total_area'].iloc[0]**0.5)}²)")
    print("=" * 78)
    print(f"  {'Metric':<30}  {'mean':>8}  {'std':>8}  {'median':>8}  {'min':>8}  {'max':>8}")
    print("  " + "-" * 74)
    for col, label in cols.items():
        s = df[col]
        print(f"  {label:<30}  {s.mean():>8.4f}  {s.std():>8.4f}  {s.median():>8.4f}  {s.min():>8.4f}  {s.max():>8.4f}")

    corr = df["lam_area"].corr(df["lam_fg"])
    print(f"\n  Pearson correlation(λ_area, λ_fg) = {corr:.4f}")

    same_class = (df["label_a"] == df["label_b"]).mean()
    print(f"  Same-class pairs                  = {same_class:.3%}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# 2. Lambda distributions
# ---------------------------------------------------------------------------


def plot_lambda_distributions(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(8, 2))
    # fig.suptitle("CutMix Lambda Distributions", fontsize=13)

    for ax, col, label, color, func in [
        (axes[0], "lam_area", r"$1 - \lambda_\text{CutMix}$", "#4878cf", lambda x: 1 - x),
        (axes[1], "lam_fg", r"$1 - \lambda_\text{oa}$", "#6acc65", lambda x: 1 - x),
        (axes[2], "delta_lam", r"$|\lambda_\text{CutMix} − \lambda_\text{oa}|$", "#d65f5f", lambda x: x),
    ]:
        ax.hist(func(df[col]), bins=20, color=color, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(func(df[col]).mean(), color="black", linestyle="--", lw=1.5, label=f"mean={df[col].mean():.3f}")
        ax.axvline(
            func(df[col]).median(), color="orange", linestyle=":", lw=1.5, label=f"median={df[col].median():.3f}"
        )
        ax.set_xlabel(label)
        ax.set_xlim(0.0, 1.0)
        ax.get_yaxis().set_visible(False)
        # ax.set_yticks([])
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "01_lambda_distributions.pdf"))


# ---------------------------------------------------------------------------
# 3. Lambda scatter
# ---------------------------------------------------------------------------


def plot_lambda_scatter(df, out_dir):
    fig, ax = plt.subplots(figsize=(3.4, 2))

    hb = ax.hexbin(
        df["lam_area"],
        df["lam_fg"],
        gridsize=12,
        cmap="YlOrRd",
        mincnt=1,
        extent=[0, 1, 0, 1],
    )
    cbar = fig.colorbar(hb, ax=ax)
    # cbar.set_ticks([])
    cbar.set_label("Samples")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label=r"$\lambda_\text{oa} = \lambda_\text{CutMix}$")

    # Binned mean ± std of λ_fg for each λ_area bin
    bins = pd.cut(df["lam_area"], bins=10)
    centers = df.groupby(bins)["lam_area"].mean()
    means = df.groupby(bins)["lam_fg"].median()
    stds = df.groupby(bins)["lam_fg"].std()
    ax.plot(
        centers,
        means,
        "-",
        color="#4878cf",
        lw=1.5,
        ms=4,
        label=r"median$(\lambda_\text{oa} | \lambda_\text{CutMix})$",
    )
    # ax.fill_between(centers, means - stds, means + stds, color="#4878cf", alpha=0.25)

    corr = df["lam_area"].corr(df["lam_fg"])
    ax.set_xlabel(r"$\lambda_\text{CutMix}$", fontsize=11)
    ax.set_ylabel(r"$\lambda_\text{oa}$", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.03, 1.03)

    print(f"==============================\nmeans:\n{means}\n================================")

    fig.tight_layout(pad=0.1)

    savefig(fig, os.path.join(out_dir, "02_lambda_scatter.pdf"))


# ---------------------------------------------------------------------------
# 4. Foreground pixel distributions
# ---------------------------------------------------------------------------


def plot_fg_pixels(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Foreground Pixel Counts in Original Images", fontsize=12)

    for ax, col, label, color in [
        (axes[0], "fg_pixels_a_total", "Image A  (fg pixels)", "#4878cf"),
        (axes[1], "fg_pixels_b_total", "Image B  (fg pixels)", "#6acc65"),
    ]:
        ax.hist(df[col], bins=60, color=color, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(df[col].mean(), color="black", linestyle="--", lw=1.5, label=f"mean={df[col].mean():.0f}")
        ax.axvline(df[col].median(), color="orange", linestyle=":", lw=1.5, label=f"median={df[col].median():.0f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "03_fg_pixels.png"))


# ---------------------------------------------------------------------------
# 5. FG pixels visible in the mixed image
# ---------------------------------------------------------------------------


def plot_fg_visible(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Foreground Pixels Visible in Mixed Image", fontsize=12)

    for ax, col, label, color in [
        (axes[0], "fg_visible_a_in_mix", "From A  (outside cut box)", "#4878cf"),
        (axes[1], "fg_visible_b_in_mix", "From B  (inside cut box)", "#6acc65"),
    ]:
        ax.hist(df[col], bins=60, color=color, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(df[col].mean(), color="black", linestyle="--", lw=1.5, label=f"mean={df[col].mean():.0f}")
        ax.axvline(df[col].median(), color="orange", linestyle=":", lw=1.5, label=f"median={df[col].median():.0f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "04_fg_visible.png"))


# ---------------------------------------------------------------------------
# 5b. λ_area vs fg pixel count — hexbin (A and B side by side)
# ---------------------------------------------------------------------------


def plot_lam_vs_pixelcount_hexbin(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("λ_area vs foreground pixel count", fontsize=12)

    px_max = max(df["fg_visible_a_in_mix"].max(), df["fg_visible_b_in_mix"].max())

    for ax, px_col, title in [
        (axes[0], "fg_visible_a_in_mix", "Image A  — fg pixels outside bbox (kept)"),
        (axes[1], "fg_visible_b_in_mix", "Image B  — fg pixels inside bbox (pasted in)"),
    ]:
        hb = ax.hexbin(
            df["lam_area"],
            df[px_col],
            gridsize=30,
            cmap="YlOrRd",
            mincnt=1,
            extent=[0, 1, 0, px_max],
        )
        fig.colorbar(hb, ax=ax, label="Count")

        # Binned median pixel count per λ_area bin
        bins = pd.cut(df["lam_area"], bins=20)
        centers = df.groupby(bins)["lam_area"].mean()
        medians = df.groupby(bins)[px_col].median()
        ax.plot(centers, medians, "o-", color="#4878cf", lw=1.5, ms=4, label="median fg pixels")
        ax.legend(fontsize=8)

        ax.set_xlabel("λ_area")
        ax.set_ylabel("FG pixels in mixed image")
        ax.set_xlim(0, 1)
        ax.set_title(title)

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "05b_lam_vs_pixelcount_hexbin.png"))


# ---------------------------------------------------------------------------
# 6. Lambda vs FG pixel count
# ---------------------------------------------------------------------------


def plot_lam_vs_fg(df, out_dir, n_bins=20):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("λ vs Foreground Pixel Count  (binned means ± std)", fontsize=12)

    for ax, fg_col, title in [
        (axes[0], "fg_pixels_a_total", "Image A foreground pixels"),
        (axes[1], "fg_pixels_b_total", "Image B foreground pixels"),
    ]:
        bins = pd.cut(df[fg_col], bins=n_bins)
        bin_centers = df.groupby(bins)[fg_col].mean()

        for lam_col, label, color in [
            ("lam_area", "λ_area", "#4878cf"),
            ("lam_fg", "λ_fg", "#6acc65"),
        ]:
            means = df.groupby(bins)[lam_col].mean()
            stds = df.groupby(bins)[lam_col].std()
            ax.plot(bin_centers, means, "o-", color=color, lw=1.5, label=label)
            ax.fill_between(bin_centers, means - stds, means + stds, color=color, alpha=0.15)

        ax.set_xlabel(title)
        ax.set_ylabel("λ value")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(title)

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "05_lam_vs_fg_pixels.png"))


# ---------------------------------------------------------------------------
# 7. Per-class foreground pixel stats
# ---------------------------------------------------------------------------


def plot_per_class_fg(df, out_dir, top_k=30):
    per_class = (
        df.groupby("label_a")["fg_pixels_a_total"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(top_k)
    )

    fig, ax = plt.subplots(figsize=(max(10, top_k // 2), 5))
    x = np.arange(len(per_class))
    ax.bar(
        x,
        per_class["mean"],
        yerr=per_class["std"],
        capsize=3,
        color="#4878cf",
        alpha=0.8,
        error_kw=dict(elinewidth=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(per_class["label_a"], rotation=90, fontsize=7)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Mean FG pixels")
    ax.set_title(f"Per-class mean foreground pixels (top-{top_k} by mean)")
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "06_per_class_fg.png"))


# ---------------------------------------------------------------------------
# 8. Delta-lambda vs fg coverage
# ---------------------------------------------------------------------------


def plot_delta_vs_coverage(df, out_dir, n_bins=20):
    """How much does λ_fg deviate from λ_area as a function of fg coverage?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("|λ_area − λ_fg| vs foreground coverage", fontsize=12)

    for ax, frac_col, title in [
        (axes[0], "fg_frac_a", "FG fraction of image A"),
        (axes[1], "fg_frac_b", "FG fraction of image B"),
    ]:
        bins = pd.cut(df[frac_col], bins=n_bins)
        centers = df.groupby(bins)[frac_col].mean()
        means = df.groupby(bins)["delta_lam"].mean()
        stds = df.groupby(bins)["delta_lam"].std()

        ax.plot(centers, means, "o-", color="#d65f5f", lw=1.5)
        ax.fill_between(centers, means - stds, means + stds, color="#d65f5f", alpha=0.15)
        ax.set_xlabel(title)
        ax.set_ylabel("|λ_area − λ_fg|")
        ax.set_ylim(bottom=0)
        ax.set_title(title)

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "07_delta_vs_fg_coverage.png"))


# ---------------------------------------------------------------------------
# 9. Object size vs |λ_area − λ_fg|
# ---------------------------------------------------------------------------


def _objsize_vs_delta_single(ax, sizes, delta_lam, xlabel):
    px_max = sizes.max()
    hb = ax.hexbin(
        sizes,
        delta_lam,
        gridsize=20,
        cmap="YlOrRd",
        mincnt=1,
        extent=[0, px_max, 0, delta_lam.max()],
    )
    cbar = plt.gcf().colorbar(hb, ax=ax)
    # cbar.set_ticks([])
    cbar.set_label("Samples")

    bins = pd.cut(sizes, bins=25)
    tmp = pd.DataFrame({"size": sizes, "delta": delta_lam})
    centers = tmp.groupby(bins)["size"].mean()
    means = tmp.groupby(bins)["delta"].mean()
    ax.plot(centers, means, "-", color="#4878cf", lw=1.5, ms=4, label="mean")
    ax.legend(fontsize=9)
    ax.set_yticks([0.0, 0.4, 0.8])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r"$|\lambda_\text{CutMix} - \lambda_\text{oa}|$", fontsize=11)


def plot_objsize_vs_delta(df, out_dir):
    # Plot A: relative object size of image A (fg pixels / H*W)
    fig, ax = plt.subplots(figsize=(3.5, 2))
    _objsize_vs_delta_single(
        ax,
        df["fg_frac_a"].values,
        df["delta_lam"].values,
        "Relative object size (image A)",
    )
    plt.tight_layout(pad=0.0)
    savefig(fig, os.path.join(out_dir, "08a_objsize_a_vs_delta_lam.pdf"))

    # Plot B: relative object size of image B (fg pixels / H*W)
    fig, ax = plt.subplots(figsize=(3.5, 2))
    _objsize_vs_delta_single(
        ax,
        df["fg_frac_b"].values,
        df["delta_lam"].values,
        "Relative object size (image B)",
    )
    plt.tight_layout(pad=0.1)
    savefig(fig, os.path.join(out_dir, "08b_objsize_b_vs_delta_lam.pdf"))

    # Plot C: mean relative object size of A and B
    fig, ax = plt.subplots(figsize=(3.5, 2))
    _objsize_vs_delta_single(
        ax,
        ((df["fg_frac_a"] + df["fg_frac_b"]) / 2.0).values,
        df["delta_lam"].values,
        "Relative object size",
    )
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    savefig(fig, os.path.join(out_dir, "08_objsize_vs_delta_lam.pdf"))


# ---------------------------------------------------------------------------
# 10. KL divergence distribution
# ---------------------------------------------------------------------------


def plot_kl_distribution(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: histogram of KL(area || fg)
    ax = axes[0]
    ax.hist(df["kl_area_fg"], bins=60, color="#9b59b6", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(
        df["kl_area_fg"].mean(), color="black", linestyle="--", lw=1.5, label=f"mean={df['kl_area_fg'].mean():.4f}"
    )
    ax.axvline(
        df["kl_area_fg"].median(),
        color="orange",
        linestyle=":",
        lw=1.5,
        label=f"median={df['kl_area_fg'].median():.4f}",
    )
    ax.set_xlabel("KL( Bernoulli(λ_area) ‖ Bernoulli(λ_fg) )", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("KL divergence distribution")
    ax.legend(fontsize=8)

    # Right: KL vs object size (hexbin + median line)
    ax = axes[1]
    px_max = df["object_size"].max()
    hb = ax.hexbin(
        df["object_size"],
        df["kl_area_fg"],
        gridsize=40,
        cmap="YlOrRd",
        mincnt=1,
        extent=[0, px_max, 0, df["kl_area_fg"].quantile(0.99)],
    )
    fig.colorbar(hb, ax=ax, label="Count")

    bins = pd.cut(df["object_size"], bins=25)
    centers = df.groupby(bins)["object_size"].mean()
    medians = df.groupby(bins)["kl_area_fg"].median()
    ax.plot(centers, medians, "o-", color="#4878cf", lw=1.5, ms=4, label="median KL")
    ax.legend(fontsize=8)
    ax.set_xlabel("Object size  (mean fg pixels)", fontsize=10)
    ax.set_ylabel("KL divergence")
    ax.set_yscale("log")
    ax.set_title("KL divergence vs object size")

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "09_kl_divergence.png"))


# ---------------------------------------------------------------------------
# 11. Ghost labels
# ---------------------------------------------------------------------------


def plot_ghost_labels(df, out_dir):
    """
    Ghost label: a class receives non-zero label weight even though none of its
    object pixels appear in the mixed image.

      Ghost A: fg_visible_a_in_mix == 0  but  λ_area > 0  (A gets credit, no A pixels visible)
      Ghost B: fg_visible_b_in_mix == 0  but  (1 - λ_area) > 0  (B gets credit, no B pixels visible)

    Plots the distribution of the ghost label weight in each case.
    """
    ghost_a = df[df["fg_visible_a_in_mix"] <= 0.5].copy()  # no A pixels in mix
    ghost_b = df[df["fg_visible_b_in_mix"] <= 0.5].copy()  # no B pixels in mix

    frac_a = len(ghost_a) / len(df)
    frac_b = len(ghost_b) / len(df)

    print(f"\n  Ghost labels:")
    print(f"    Ghost A (no A pixels in mix): {len(ghost_a):,}  ({frac_a:.2%} of samples)")
    print(f"    Ghost B (no B pixels in mix): {len(ghost_b):,}  ({frac_b:.2%} of samples)")

    fig, axes = plt.subplots(1, 2, figsize=(6, 2))

    for ax, ghost_df, weight_col, label, color, frac, title in [
        (axes[0], ghost_a, "lam_area", r"$\lambda_\text{CutMix}$", "#4878cf", frac_a, "Ghost A"),
        (axes[1], ghost_b, "1 - lam_area", r"$1 - \lambda_\text{CutMix}$", "#6acc65", frac_b, "Ghost B"),
    ]:
        if len(ghost_df) == 0:
            ax.text(0.5, 0.5, "No ghost labels found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        weights = ghost_df[weight_col] if weight_col == "lam_area" else 1 - ghost_df["lam_area"]
        ax.hist(weights, bins=10, color=color, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(weights.mean(), color="black", linestyle="--", lw=1.5, label=f"mean={weights.mean():.3f}")
        ax.axvline(weights.median(), color="orange", linestyle=":", lw=1.5, label=f"median={weights.median():.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{title} ({frac*100:.1f}% of samples)")
        ax.get_yaxis().set_visible(False)
        ax.legend(fontsize=8)

    plt.tight_layout(pad=0.1)
    savefig(fig, os.path.join(out_dir, "10_ghost_labels.pdf"))


# ---------------------------------------------------------------------------
# 12. Ghost label rate vs object size
# ---------------------------------------------------------------------------


def print_ghost_by_objsize_percentiles(df):
    """Print ghost-label rate for samples with object_size <= each percentile."""
    is_ghost = (df["fg_visible_a_in_mix"] <= 0.5) | (df["fg_visible_b_in_mix"] <= 0.5)
    percentiles = [5, 10, 20, 25, 50]
    thresholds = np.percentile(df["object_size"], percentiles)

    print("\n  Ghost label rate by object-size percentile:")
    print(f"    {'Percentile':>10}  {'Size threshold (px)':>20}  {'Ghost rate':>12}  {'n samples':>10}")
    print("    " + "-" * 58)
    for pct, thr in zip(percentiles, thresholds):
        mask = df["object_size"] <= thr
        rate = is_ghost[mask].mean() if mask.sum() > 0 else float("nan")
        print(f"    {pct:>9}th  {thr:>20.1f}  {rate:>11.2%}  {mask.sum():>10,}")


def plot_ghost_vs_objsize(df, out_dir, n_bins=20):
    """Line plot: per-bin ghost-label rate vs object size percentile.

    Samples are sorted by object size and split into n_bins equal-size bins.
    Each point shows the ghost-label rate within that bin. X-axis shows the
    percentile of the bin centre (0–100).
    """
    is_ghost = (df["fg_visible_a_in_mix"] <= 0.5) | (df["fg_visible_b_in_mix"] <= 0.5)

    sorted_idx = df["object_size"].argsort().values
    sorted_ghost = is_ghost.iloc[sorted_idx].values

    bins = np.array_split(sorted_ghost, n_bins)
    bin_centers_pct = [(i + 0.5) / n_bins * 100 for i in range(n_bins)]
    bin_ghost_rates = [b.mean() * 100 for b in bins]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(bin_centers_pct, bin_ghost_rates, color="#d65f5f", lw=1.5, marker="o", ms=3)

    ax.set_xlabel("Object size percentile", fontsize=10)
    ax.set_ylabel("Ghost label rate [%]", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout(pad=0.2)
    savefig(fig, os.path.join(out_dir, "11_ghost_vs_objsize.pdf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze CutMix statistics CSV from cutmix_collect.py")
    parser.add_argument("csv", help="Path to CSV file produced by cutmix_collect.py")
    parser.add_argument("--out-dir", default="plots", help="Directory to write plot PNGs (default: plots/)")
    parser.add_argument("--top-k", type=int, default=30, help="Number of top classes to show in per-class plot")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = load(args.csv)
    print(f"  {len(df):,} rows loaded")

    print_summary(df)

    print(f"\nGenerating plots → {args.out_dir}/")
    plot_lambda_distributions(df, args.out_dir)
    plot_lambda_scatter(df, args.out_dir)
    plot_fg_pixels(df, args.out_dir)
    plot_fg_visible(df, args.out_dir)
    plot_lam_vs_fg(df, args.out_dir)
    plot_lam_vs_pixelcount_hexbin(df, args.out_dir)
    plot_per_class_fg(df, args.out_dir, top_k=args.top_k)
    plot_delta_vs_coverage(df, args.out_dir)
    plot_objsize_vs_delta(df, args.out_dir)
    plot_kl_distribution(df, args.out_dir)
    plot_ghost_labels(df, args.out_dir)
    print_ghost_by_objsize_percentiles(df)
    plot_ghost_vs_objsize(df, args.out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
