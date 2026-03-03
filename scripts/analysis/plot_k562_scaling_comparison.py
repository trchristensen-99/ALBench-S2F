#!/usr/bin/env python3
"""
Plot K562 MPRA scaling curve: DREAM-RNN vs AlphaGenome.

Reads result.json files from:
  outputs/exp0_k562_scaling/                          (DREAM-RNN, hashFrag test sets)
  outputs/exp0_k562_scaling_alphagenome_cached_rcaug/ (AlphaGenome cached, hashFrag test sets)

Both use the same hashFrag test splits (in_dist, SNV, OOD) and are directly comparable.

Outputs PNG(s) to outputs/analysis/plots/.
Run from repo root:
  python scripts/analysis/plot_k562_scaling_comparison.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DREAM_DIR = REPO_ROOT / "outputs" / "exp0_k562_scaling"
AG_DIR = REPO_ROOT / "outputs" / "exp0_k562_scaling_alphagenome_cached_rcaug"
OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "plots"

# Standard experiment fractions (log-spaced design)
STANDARD_FRACS = {0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_records(results_dir: Path, model_name: str) -> pd.DataFrame:
    """Recursively find all result.json files and return a flat DataFrame."""
    records = []
    for path in sorted(results_dir.rglob("result.json")):
        with open(path) as f:
            d = json.load(f)
        tm = d.get("test_metrics", {})
        in_dist = tm.get("in_distribution", {})
        ood = tm.get("ood", {})
        snv_abs = tm.get("snv_abs", {})
        snv_delta = tm.get("snv_delta", {})
        records.append(
            {
                "model": model_name,
                "fraction": round(float(d["fraction"]), 6),
                "n_samples": d.get("n_samples"),
                "val_pearson": d.get("best_val_pearson_r") or d.get("best_val_pearson"),
                "in_dist_pearson": in_dist.get("pearson_r"),
                "ood_pearson": ood.get("pearson_r"),
                "snv_abs_pearson": snv_abs.get("pearson_r"),
                "snv_delta_pearson": snv_delta.get("pearson_r"),
                "path": str(path),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # Keep only standard fractions
    df = df[
        df["fraction"].apply(
            lambda f: min(STANDARD_FRACS, key=lambda s: abs(s - f)) == round(f, 6)
            or round(f, 2) in STANDARD_FRACS
        )
    ]
    # Snap fraction to nearest standard value to avoid float-key collisions
    df["fraction"] = df["fraction"].apply(lambda f: min(STANDARD_FRACS, key=lambda s: abs(s - f)))
    return df


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame(columns=["fraction", "mean", "std", "n", "pct"])
    valid = df.dropna(subset=[metric])
    agg = valid.groupby("fraction")[metric].agg(mean="mean", std="std", n="count").reset_index()
    agg["pct"] = agg["fraction"] * 100
    return agg.sort_values("fraction")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "DREAM-RNN": "#E05E4B",
    "AlphaGenome": "#4B7BE0",
}

MARKERS = {
    "DREAM-RNN": "o",
    "AlphaGenome": "s",
}


def plot_metric(
    dream_df: pd.DataFrame,
    ag_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    ax: plt.Axes,
    ag_label: str = "AlphaGenome",
):
    """Draw one scaling-curve panel onto ax."""
    dream_agg = aggregate(dream_df, metric)
    ag_agg = aggregate(ag_df, metric)

    for agg, label in [(dream_agg, "DREAM-RNN"), (ag_agg, ag_label)]:
        if agg.empty:
            continue
        color = COLORS.get(label.split()[0], "#555555")
        marker = MARKERS.get(label.split()[0], "^")
        yerr = agg["std"].fillna(0)
        ax.errorbar(
            agg["pct"],
            agg["mean"],
            yerr=yerr,
            fmt=f"-{marker}",
            color=color,
            label=label,
            capsize=4,
            linewidth=2,
            markersize=7,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}%"))
    ax.set_xlabel("Training data (% of total)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", ls="--", alpha=0.4, zorder=0)
    ax.tick_params(axis="both", labelsize=9)


def make_figure(dream_df: pd.DataFrame, ag_df: pd.DataFrame, out_path: Path):
    """Two-panel figure: in-distribution and OOD Pearson R."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Determine label suffix if AG data is incomplete
    n_ag_fracs = ag_df["fraction"].nunique() if not ag_df.empty else 0
    ag_label = "AlphaGenome" if n_ag_fracs == 7 else f"AlphaGenome (partial, {n_ag_fracs}/7 fracs)"

    plot_metric(
        dream_df, ag_df, "in_dist_pearson", "Test Pearson R (in-distribution)", axes[0], ag_label
    )
    axes[0].set_title("In-distribution test set", fontsize=12)

    plot_metric(dream_df, ag_df, "ood_pearson", "Test Pearson R (OOD)", axes[1], ag_label)
    axes[1].set_title("Out-of-distribution test set", fontsize=12)

    fig.suptitle("K562 MPRA: Data efficiency — DREAM-RNN vs AlphaGenome", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_four_panel_figure(dream_df: pd.DataFrame, ag_df: pd.DataFrame, out_path: Path):
    """Four-panel figure: in-dist, OOD, SNV abs, SNV delta Pearson R."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=False)
    axes = axes.flatten()

    n_ag_fracs = ag_df["fraction"].nunique() if not ag_df.empty else 0
    ag_label = "AlphaGenome" if n_ag_fracs == 7 else f"AlphaGenome (partial, {n_ag_fracs}/7 fracs)"

    panels = [
        ("in_dist_pearson", "Test Pearson R (in-distribution)", "In-distribution test"),
        ("ood_pearson", "Test Pearson R (OOD)", "Out-of-distribution test"),
        ("snv_abs_pearson", "Test Pearson R (SNV abs)", "SNV absolute expression"),
        ("snv_delta_pearson", "Test Pearson R (SNV Δ)", "SNV effect (Δlog2FC)"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        plot_metric(dream_df, ag_df, metric, ylabel, ax, ag_label)
        ax.set_title(title, fontsize=11)

    fig.suptitle("K562 MPRA: Data efficiency — DREAM-RNN vs AlphaGenome", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_single_panel(
    dream_df: pd.DataFrame, ag_df: pd.DataFrame, metric: str, ylabel: str, out_path: Path
):
    """Single-panel figure for one metric."""
    fig, ax = plt.subplots(figsize=(7, 5))
    n_ag_fracs = ag_df["fraction"].nunique() if not ag_df.empty else 0
    ag_label = "AlphaGenome" if n_ag_fracs == 7 else f"AlphaGenome (partial, {n_ag_fracs}/7 fracs)"
    plot_metric(dream_df, ag_df, metric, ylabel, ax, ag_label)
    ax.set_title("K562 MPRA: Data efficiency — DREAM-RNN vs AlphaGenome", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_model_summary(df: pd.DataFrame, name: str):
    metrics = [
        ("in_dist_pearson", "in-dist"),
        ("ood_pearson", "OOD"),
        ("snv_abs_pearson", "SNV-abs"),
        ("snv_delta_pearson", "SNV-delta"),
    ]
    print(f"\n=== {name} K562 scaling ===")
    if df.empty:
        print("  (no data)")
        return
    # Print in-dist as primary; show other metrics on same line if available
    agg_all = {m: aggregate(df, m) for m, _ in metrics}
    fracs = sorted(df["fraction"].dropna().unique())
    header = f"  {'frac':>7}  {'n':>2}  " + "  ".join(f"{lbl:>10}" for _, lbl in metrics)
    print(header)
    for frac in fracs:
        row_parts = [f"  {frac * 100:6.1f}%"]
        n_seeds = None
        for m, _ in metrics:
            agg = agg_all[m]
            row_frac = agg[agg["fraction"] == frac]
            if row_frac.empty or np.isnan(row_frac["mean"].values[0]):
                row_parts.append(f"{'—':>10}")
            else:
                mean = row_frac["mean"].values[0]
                std = row_frac["std"].values[0]
                n_seeds = int(row_frac["n"].values[0])
                std_str = f"±{std:.3f}" if not np.isnan(std) else "      "
                row_parts.append(f"{mean:.4f}{std_str:>7}")
        row_parts.insert(1, f"  {n_seeds or 0:>2}")
        print("".join(row_parts))
    print()


def print_summary(dream_df: pd.DataFrame, ag_df: pd.DataFrame):
    _print_model_summary(dream_df, "DREAM-RNN")
    _print_model_summary(ag_df, "AlphaGenome (cached, rcaug)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading DREAM-RNN results from: {DREAM_DIR}")
    dream_df = load_records(DREAM_DIR, "DREAM-RNN")
    print(
        f"  Found {len(dream_df)} records across {dream_df['fraction'].nunique() if not dream_df.empty else 0} fractions."
    )

    print(f"Loading AlphaGenome results from: {AG_DIR}")
    ag_df = load_records(AG_DIR, "AlphaGenome")
    print(
        f"  Found {len(ag_df)} records across {ag_df['fraction'].nunique() if not ag_df.empty else 0} fractions."
    )

    print_summary(dream_df, ag_df)

    # Save combined records CSV for downstream analysis
    all_df = pd.concat([dream_df, ag_df], ignore_index=True) if not ag_df.empty else dream_df
    all_df.to_csv(OUT_DIR / "k562_scaling_records.csv", index=False)
    print(f"Saved: {OUT_DIR / 'k562_scaling_records.csv'}")

    # Two-panel figure (in-dist + OOD)
    make_figure(dream_df, ag_df, OUT_DIR / "k562_scaling_comparison.png")

    # Four-panel figure (in-dist + OOD + SNV abs + SNV delta)
    make_four_panel_figure(dream_df, ag_df, OUT_DIR / "k562_scaling_comparison_4panel.png")

    # Single-panel in-distribution only (cleaner for presentations)
    make_single_panel(
        dream_df,
        ag_df,
        "in_dist_pearson",
        "Test Pearson R (in-distribution)",
        OUT_DIR / "k562_scaling_in_dist.png",
    )


if __name__ == "__main__":
    main()
