"""Parallel-coordinates plot for HP-search trials.

Reads all_trials.csv from aggregate_trials.py and generates a parallel
coordinates plot with one axis per HP (lr, batch_size, weight_decay,
dropout, width, depth) and a final axis for val_loss. Each line is one
trial, color-coded by val_loss (lower = better = greener).

Per the PI's request: this is the canonical figure for showing what
combinations of HPs work across the search.

Outputs (one per arch×D cell):
    results/preflight/hpsearch/parallel_coords_{arch}_d{D}.html (Plotly)
    results/preflight/hpsearch/parallel_coords_{arch}_d{D}.png  (matplotlib)

Optional --strategy filter to compare strategies within a cell.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "results/preflight/hpsearch"


def _matplotlib_parallel(df: pd.DataFrame, out_path: Path, title: str):
    """Matplotlib parallel coordinates (no plotly dep). Log-scale lr, wd."""
    dims = ["lr", "batch_size", "weight_decay", "dropout", "width", "depth", "val_loss"]
    df = df.dropna(subset=dims).copy()
    if len(df) == 0:
        print(f"  no trials with all dims present for {title}")
        return
    # Apply log10 transform to lr, weight_decay, val_loss
    log_dims = {"lr", "weight_decay", "val_loss"}
    plot_df = df[dims].copy()
    for col in dims:
        if col in log_dims:
            plot_df[col] = np.log10(plot_df[col].clip(lower=1e-10))

    # Normalize each axis 0-1 for display, store original ticks
    norms = {}
    for col in dims:
        v = plot_df[col].values
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-9:
            norms[col] = (lo, hi, np.full_like(v, 0.5))
        else:
            norms[col] = (lo, hi, (v - lo) / (hi - lo))

    fig, ax = plt.subplots(figsize=(14, 6))
    x_positions = list(range(len(dims)))
    val_loss = plot_df["val_loss"].values
    # Color by val_loss (lower = better)
    cmap = plt.cm.RdYlGn_r
    vmin, vmax = val_loss.min(), val_loss.max()
    norm = plt.Normalize(vmin, vmax)

    # Plot lines (low val_loss on top via reverse sort)
    order = np.argsort(-val_loss)  # high → bottom, low → top
    for idx in order:
        y = [norms[col][2][idx] for col in dims]
        ax.plot(x_positions, y, color=cmap(norm(val_loss[idx])), alpha=0.45, linewidth=1.0)

    # Axes
    for xi, col in enumerate(dims):
        lo, hi, _ = norms[col]
        ax.axvline(xi, color="black", linewidth=0.5, alpha=0.5)
        # 5 tick marks per axis
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = lo + frac * (hi - lo)
            disp = val
            if col in log_dims:
                disp = 10**val
                tick_label = f"{disp:.1e}"
            else:
                tick_label = f"{disp:.3g}"
            ax.text(xi - 0.05, frac, tick_label, ha="right", va="center", fontsize=7)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(dims, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, len(dims) - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
    cbar.set_label("val_loss (log scale)", fontsize=9)

    # Highlight top-3 in bold red
    top3 = df.nsmallest(3, "val_loss")
    for _, row in top3.iterrows():
        idx = df.index.get_loc(row.name) if row.name in df.index else None
        if idx is None:
            continue
        # Get row's index in plot_df
        try:
            plot_idx = plot_df.index.get_loc(row.name)
        except KeyError:
            continue
        y = [norms[col][2][plot_idx] for col in dims]
        ax.plot(x_positions, y, color="darkred", alpha=0.9, linewidth=2.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(ROOT / "all_trials.csv"),
        help="Path to aggregated trials CSV.",
    )
    ap.add_argument(
        "--by",
        default="arch_d",
        choices=["arch_d", "strategy", "all"],
        help="Generate plots grouped by (arch, d_train), by strategy, or one overall.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if len(df) == 0:
        print("No trials in CSV.")
        return
    out_root = ROOT / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.by == "arch_d":
        for (arch, d), g in df.groupby(["arch", "d_train"]):
            title = f"Parallel coords — {arch} D={d} ({len(g)} trials)"
            out = out_root / f"pcoords_{arch}_d{int(d)}.png"
            _matplotlib_parallel(g, out, title)
    elif args.by == "strategy":
        for strategy, g in df.groupby("strategy"):
            title = f"Parallel coords — strategy={strategy} ({len(g)} trials)"
            out = out_root / f"pcoords_strategy_{strategy}.png"
            _matplotlib_parallel(g, out, title)
    elif args.by == "all":
        title = f"Parallel coords — all {len(df)} trials"
        out = out_root / "pcoords_all.png"
        _matplotlib_parallel(df, out, title)


if __name__ == "__main__":
    main()
