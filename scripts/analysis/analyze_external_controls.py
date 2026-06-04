#!/usr/bin/env python3
"""Analyze scrambled/negative control expression from external MPRA datasets.

Dataset: Inoue et al. 2017 (GSE83894) — lentiMPRA in HepG2
  - 2440 designed regulatory elements tested in both episomal and chromosomal context
  - 102 scrambled negative controls, 102 positive controls
  - MT = Mutant (non-integrating/episomal) RNA/DNA ratio
  - WT = Wild-type (integrating/chromosomal) RNA/DNA ratio

Reference: Inoue et al. Genome Research 27:38-52 (2017)
"""

import gzip
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("/tmp/GSE83894")
INOUE_FILE = DATA_DIR / "GSE83894_ActivityRatios.tsv.gz"
OUT_DIR = REPO_ROOT / "results" / "dataset_comparison"

# Download if not present
if not INOUE_FILE.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE83nnn/GSE83894/suppl/GSE83894_ActivityRatios.tsv.gz"
    print(f"Downloading {url} ...")
    import urllib.request

    urllib.request.urlretrieve(url, INOUE_FILE)
    print("Done.")


def load_inoue():
    """Load Inoue et al. 2017 activity ratios."""
    df = pd.read_csv(INOUE_FILE, sep="\t", compression="gzip")
    # Key columns: Region, category, categoryDetail, MT (episomal), WT (chromosomal)
    # MT1-MT3 are replicates; MT is the mean
    return df


def analyze_inoue(df):
    """Analyze negative/positive controls vs all elements."""
    neg = df[df["categoryDetail"] == "negative"]
    pos = df[df["categoryDetail"] == "positive"]
    genomic = df[~df["categoryDetail"].isin(["negative", "positive"])]

    print("=" * 70)
    print("Inoue et al. 2017 (GSE83894) — lentiMPRA in HepG2")
    print("=" * 70)
    print(f"Total elements: {len(df)}")
    print(f"Negative controls (scrambled): {len(neg)}")
    print(f"Positive controls (designed active): {len(pos)}")
    print(f"Genomic elements: {len(genomic)}")
    print()

    for label, subset in [
        ("All elements", df),
        ("Negative controls", neg),
        ("Positive controls", pos),
        ("Genomic elements", genomic),
    ]:
        for ctx, col in [("Episomal (MT)", "MT"), ("Chromosomal (WT)", "WT")]:
            vals = subset[col].dropna()
            log2_vals = np.log2(vals)
            print(f"  {label} — {ctx}:")
            print(f"    N = {len(vals)}")
            print(
                f"    RNA/DNA ratio: mean={vals.mean():.4f}, median={vals.median():.4f}, std={vals.std():.4f}"
            )
            print(
                f"    log2(RNA/DNA): mean={log2_vals.mean():.4f}, median={log2_vals.median():.4f}, std={log2_vals.std():.4f}"
            )
            print(f"    range: [{vals.min():.4f}, {vals.max():.4f}]")
            # Active = log2(RNA/DNA) > 0, i.e. ratio > 1
            pct_active = (vals > 1).sum() / len(vals) * 100
            print(f"    % active (ratio > 1): {pct_active:.1f}%")
            print()

    # Key comparison: episomal vs chromosomal for negative controls
    print("-" * 70)
    print("KEY: Negative control expression in episomal vs chromosomal context")
    print("-" * 70)
    neg_mt = neg["MT"].dropna()
    neg_wt = neg["WT"].dropna()
    all_mt = df["MT"].dropna()
    all_wt = df["WT"].dropna()

    print(f"  Episomal (MT) neg controls: log2 mean = {np.log2(neg_mt).mean():.4f}")
    print(f"  Chromosomal (WT) neg controls: log2 mean = {np.log2(neg_wt).mean():.4f}")
    print(f"  Episomal (MT) all elements: log2 mean = {np.log2(all_mt).mean():.4f}")
    print(f"  Chromosomal (WT) all elements: log2 mean = {np.log2(all_wt).mean():.4f}")
    print()

    # Test separation
    from scipy import stats

    t_mt, p_mt = stats.ttest_ind(np.log2(neg_mt), np.log2(pos["MT"].dropna()))
    t_wt, p_wt = stats.ttest_ind(np.log2(neg_wt), np.log2(pos["WT"].dropna()))
    print(f"  Neg vs Pos (episomal):     t={t_mt:.2f}, p={p_mt:.2e}")
    print(f"  Neg vs Pos (chromosomal):  t={t_wt:.2f}, p={p_wt:.2e}")
    print()

    # Effect size: how much more expression do positive controls have?
    print(f"  Positive/Negative ratio (episomal):     {pos['MT'].mean() / neg['MT'].mean():.2f}x")
    print(f"  Positive/Negative ratio (chromosomal):  {pos['WT'].mean() / neg['WT'].mean():.2f}x")
    print()

    return neg, pos, genomic


def plot_inoue(df, neg, pos, genomic):
    """Create multi-panel figure for Inoue et al. analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── Panel A: Histogram of log2(RNA/DNA) — Episomal ──
    ax = axes[0, 0]
    bins = np.linspace(-1.5, 3.0, 60)
    ax.hist(
        np.log2(genomic["MT"].dropna()),
        bins=bins,
        alpha=0.5,
        color="gray",
        label=f"Genomic (n={len(genomic)})",
        density=True,
    )
    ax.hist(
        np.log2(neg["MT"].dropna()),
        bins=bins,
        alpha=0.7,
        color="steelblue",
        label=f"Neg controls (n={len(neg)})",
        density=True,
    )
    ax.hist(
        np.log2(pos["MT"].dropna()),
        bins=bins,
        alpha=0.7,
        color="firebrick",
        label=f"Pos controls (n={len(pos)})",
        density=True,
    )
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("log2(RNA/DNA)")
    ax.set_ylabel("Density")
    ax.set_title("A. Episomal (non-integrating)")
    ax.legend(fontsize=8)

    # ── Panel B: Histogram of log2(RNA/DNA) — Chromosomal ──
    ax = axes[0, 1]
    ax.hist(
        np.log2(genomic["WT"].dropna()),
        bins=bins,
        alpha=0.5,
        color="gray",
        label=f"Genomic (n={len(genomic)})",
        density=True,
    )
    ax.hist(
        np.log2(neg["WT"].dropna()),
        bins=bins,
        alpha=0.7,
        color="steelblue",
        label=f"Neg controls (n={len(neg)})",
        density=True,
    )
    ax.hist(
        np.log2(pos["WT"].dropna()),
        bins=bins,
        alpha=0.7,
        color="firebrick",
        label=f"Pos controls (n={len(pos)})",
        density=True,
    )
    ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("log2(RNA/DNA)")
    ax.set_ylabel("Density")
    ax.set_title("B. Chromosomal (integrating)")
    ax.legend(fontsize=8)

    # ── Panel C: Episomal vs Chromosomal scatter ──
    ax = axes[1, 0]
    ax.scatter(
        np.log2(genomic["MT"].dropna()),
        np.log2(genomic["WT"].dropna()),
        alpha=0.15,
        s=8,
        color="gray",
        label="Genomic",
        rasterized=True,
    )
    ax.scatter(
        np.log2(neg["MT"].dropna()),
        np.log2(neg["WT"].dropna()),
        alpha=0.8,
        s=30,
        color="steelblue",
        label="Neg controls",
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    ax.scatter(
        np.log2(pos["MT"].dropna()),
        np.log2(pos["WT"].dropna()),
        alpha=0.8,
        s=30,
        color="firebrick",
        label="Pos controls",
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    lims = [-1.5, 3.0]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("log2(RNA/DNA) — Episomal")
    ax.set_ylabel("log2(RNA/DNA) — Chromosomal")
    ax.set_title("C. Episomal vs Chromosomal")
    ax.legend(fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    # ── Panel D: Box/strip plot comparing contexts ──
    ax = axes[1, 1]
    categories = []
    values = []
    colors = []

    for cat_label, subset, color in [
        ("Neg\nEpisomal", neg["MT"], "steelblue"),
        ("Neg\nChromosomal", neg["WT"], "cornflowerblue"),
        ("Pos\nEpisomal", pos["MT"], "firebrick"),
        ("Pos\nChromosomal", pos["WT"], "indianred"),
        ("All\nEpisomal", df["MT"], "gray"),
        ("All\nChromosomal", df["WT"], "silver"),
    ]:
        vals = np.log2(subset.dropna())
        categories.append(cat_label)
        values.append(vals.values)
        colors.append(color)

    bp = ax.boxplot(values, tick_labels=categories, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, (vals, color) in enumerate(zip(values, colors)):
        jitter = np.random.default_rng(42).normal(0, 0.08, len(vals))
        ax.scatter(
            np.ones(len(vals)) * (i + 1) + jitter,
            vals,
            alpha=0.3,
            s=6,
            color=color,
            zorder=3,
        )
    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("log2(RNA/DNA)")
    ax.set_title("D. Distribution comparison")
    ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Inoue et al. 2017 — Scrambled controls in episomal vs chromosomal lentiMPRA (HepG2)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "external_controls.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    print("Loading Inoue et al. 2017 data...")
    df = load_inoue()
    neg, pos, genomic = analyze_inoue(df)
    print("\nGenerating plots...")
    plot_inoue(df, neg, pos, genomic)

    # Summary table for quick reference
    print("\n" + "=" * 70)
    print("SUMMARY TABLE — log2(RNA/DNA)")
    print("=" * 70)
    rows = []
    for label, subset in [
        ("Neg controls (scrambled)", neg),
        ("Pos controls (designed)", pos),
        ("All genomic elements", genomic),
    ]:
        for ctx, col in [("Episomal", "MT"), ("Chromosomal", "WT")]:
            vals = np.log2(subset[col].dropna())
            rows.append(
                {
                    "Category": label,
                    "Context": ctx,
                    "N": len(vals),
                    "Mean": f"{vals.mean():.4f}",
                    "Median": f"{vals.median():.4f}",
                    "Std": f"{vals.std():.4f}",
                    "% > 0": f"{(vals > 0).sum() / len(vals) * 100:.1f}%",
                }
            )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
