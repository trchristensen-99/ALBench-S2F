#!/usr/bin/env python3
"""
Plot K562 MPRA scaling curve: DREAM-RNN vs AlphaGenome.

Reads result.json files from:
  outputs/exp0_k562_scaling/              (DREAM-RNN)
  outputs/exp0_k562_scaling_alphagenome/  (AlphaGenome)

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
AG_DIR = REPO_ROOT / "outputs" / "exp0_k562_scaling_alphagenome"
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
        records.append(
            {
                "model": model_name,
                "fraction": round(float(d["fraction"]), 6),
                "n_samples": d.get("n_samples"),
                "val_pearson": d.get("best_val_pearson_r") or d.get("best_val_pearson"),
                "in_dist_pearson": in_dist.get("pearson_r"),
                "ood_pearson": ood.get("pearson_r"),
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


def print_summary(dream_df: pd.DataFrame, ag_df: pd.DataFrame):
    print("\n=== DREAM-RNN K562 scaling (in-dist Pearson R) ===")
    d_agg = aggregate(dream_df, "in_dist_pearson")
    for _, row in d_agg.iterrows():
        std_str = f" ± {row['std']:.4f}" if not np.isnan(row["std"]) else ""
        print(
            f"  {row['fraction'] * 100:5.1f}%  ({row['pct']:5.1f}%)  "
            f"mean={row['mean']:.4f}{std_str}  seeds={int(row['n'])}"
        )

    print("\n=== AlphaGenome K562 scaling (in-dist Pearson R) ===")
    if ag_df.empty:
        print("  (no data)")
        return
    a_agg = aggregate(ag_df, "in_dist_pearson")
    for _, row in a_agg.iterrows():
        std_str = f" ± {row['std']:.4f}" if not np.isnan(row["std"]) else ""
        print(
            f"  {row['fraction'] * 100:5.1f}%  ({row['pct']:5.1f}%)  "
            f"mean={row['mean']:.4f}{std_str}  seeds={int(row['n'])}"
        )
    print()


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

    # Two-panel figure (in-dist + OOD)
    make_figure(dream_df, ag_df, OUT_DIR / "k562_scaling_comparison.png")

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
