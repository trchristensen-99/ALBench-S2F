"""Detailed multi-axis HP heatmap figure for LegNet @ D=2000.

Panel layout (6 panels):
  Top row: LR × BS  | LR × WD     | Dropout × LR
  Bot row: Dropout walk | Epochs walk | Shift × EvoAug aug grid

Augmentation panel: 2D heatmap of shift_magnitude × EvoAug
(RC is treated as standard, included in all cells).
Each cell shows BEST test_mse across HP variations at that aug combo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
OUT = REPO / "results/preflight/figures/meeting/12_legnet_d2k_detailed.png"


def load_all_legnet_d2k():
    rows = []
    df = pd.read_parquet(REPO / "results/preflight/all_hp_results.parquet")
    df = df[(df.arch == "legnet") & (df.d_train == 2000)].copy()
    df["source"] = "existing"
    df["max_shift"] = np.nan
    df["use_evoaug"] = False
    rows.append(df)

    # New high-dim sweep (legnet_d2k_highdim + v23 corner)
    for base_path in [REPO / "results/preflight/legnet_d2k_highdim",
                       REPO / "results/preflight/legnet_d2k_v23"]:
        if not base_path.exists():
            continue
        for rj in base_path.rglob("result.json"):
            try:
                r = json.loads(rj.read_text())
                hp = r.get("hp", {}) or {}
                aug = r.get("augmentations", "")
                # Parse aug into shift_magnitude + use_evoaug
                use_evoaug = "evoaug" in aug
                max_shift = hp.get("max_shift")
                if max_shift is None:
                    if aug in ("rev_complement", "none"):
                        max_shift = 0
                    elif "shift" in aug:
                        max_shift = 15  # default if not specified
                row = {
                    "source": base_path.name,
                    "arch": r["arch"],
                    "d_train": r.get("d_train"),
                    "seed": r.get("seed"),
                    "epochs": r.get("epochs"),
                    "aug": aug,
                    "max_shift": max_shift,
                    "use_evoaug": use_evoaug,
                    "test_mse": r.get("test_mse_at_best_val"),
                    "best_val": r.get("best_val_mse"),
                    "lr": hp.get("lr"),
                    "batch_size": hp.get("batch_size"),
                    "weight_decay": hp.get("weight_decay"),
                    "dropout": hp.get("dropout") or hp.get("dropout_rate"),
                }
                rows.append(pd.DataFrame([row]))
            except Exception:
                continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def heatmap_panel(ax, df, x_col, y_col, val_col, title):
    """Generic heatmap panel with min marked."""
    cell = df.dropna(subset=[x_col, y_col, val_col])
    if cell.empty:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_title(title)
        return
    piv = cell.groupby([y_col, x_col])[val_col].min().reset_index().pivot(
        index=y_col, columns=x_col, values=val_col
    )
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                cbar_kws={"label": "test_mse"}, annot_kws={"size": 8})
    flat = piv.stack()
    if not flat.empty:
        my, mx = flat.idxmin()
        r = list(piv.index).index(my)
        c = list(piv.columns).index(mx)
        ax.plot(c + 0.5, r + 0.5, "r*", markersize=22, markeredgecolor="white",
                markeredgewidth=1.5)
    ax.set_title(f"{title} (n={len(cell)})")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def walk_panel(ax, df, x_col, val_col, title):
    cell = df.dropna(subset=[x_col, val_col])
    if cell.empty:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_title(title)
        return
    agg = cell.groupby(x_col)[val_col].agg(["mean", "min", "max", "count"]).reset_index()
    ax.errorbar(agg[x_col], agg["mean"],
                yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
                marker="o", capsize=5, color="steelblue", markersize=8)
    for _, r in agg.iterrows():
        ax.annotate(f"n={int(r['count'])}", (r[x_col], r["mean"]),
                    fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(x_col)
    ax.set_ylabel("test_mse")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def aug_grid_panel(ax, df):
    """2D heatmap: shift_magnitude × use_evoaug. Each cell = BEST test_mse
    across HP variations at that aug combo. Corner (0, off) = RC only."""
    cell = df.dropna(subset=["max_shift", "test_mse"]).copy()
    if cell.empty:
        ax.text(0.5, 0.5, "aug grid data\nstill running",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Shift × EvoAug — best HP per combo")
        return
    # Quantize shift_magnitude to bins {0, 15, 50, 100}
    cell["shift_bin"] = cell["max_shift"].fillna(0).astype(int).clip(0, 100)
    cell["use_evoaug"] = cell["use_evoaug"].fillna(False)
    piv = cell.groupby(["use_evoaug", "shift_bin"])["test_mse"].min().reset_index().pivot(
        index="use_evoaug", columns="shift_bin", values="test_mse"
    )
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                cbar_kws={"label": "test_mse (best HP)"}, annot_kws={"size": 9})
    flat = piv.stack()
    if not flat.empty:
        my, mx = flat.idxmin()
        r = list(piv.index).index(my)
        c = list(piv.columns).index(mx)
        ax.plot(c + 0.5, r + 0.5, "r*", markersize=22, markeredgecolor="white",
                markeredgewidth=1.5)
    ax.set_title(f"Shift × EvoAug — best HP per combo (n={len(cell)})")
    ax.set_xlabel("max_shift (bp)")
    ax.set_ylabel("EvoAug on/off")


def main():
    df = load_all_legnet_d2k()
    df = df.dropna(subset=["test_mse"])
    n_existing = (df.source == "existing").sum()
    n_new = (df.source != "existing").sum()
    print(f"LegNet D=2000: {n_existing} existing + {n_new} new = {len(df)} total")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    heatmap_panel(axes[0, 0], df, "batch_size", "lr", "test_mse", "Learning Rate × Batch Size")
    heatmap_panel(axes[0, 1], df, "weight_decay", "lr", "test_mse", "Learning Rate × Weight Decay")
    heatmap_panel(axes[0, 2], df, "lr", "dropout", "test_mse", "Dropout × Learning Rate")

    walk_panel(axes[1, 0], df, "dropout", "test_mse", "Dropout 1D walk")
    walk_panel(axes[1, 1], df, "epochs", "test_mse", "Epochs 1D walk")
    aug_grid_panel(axes[1, 2], df)

    fig.suptitle(
        f"LegNet @ D=2000 high-dimensional HP optimization "
        f"({len(df)} runs across 6 HP dimensions)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
