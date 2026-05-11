"""Detailed multi-axis HP heatmap figure for LegNet @ D=2000.
Combines all existing HP coverage (LR×BS×WD from earlier sweeps) with
the new high-dim sweep (Dropout × Aug, Capacity, Epochs)."""

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
    """Combine pre-existing HP results + new high-dim sweep results."""
    rows = []
    # Existing audit-tracked results
    df = pd.read_parquet(REPO / "results/preflight/all_hp_results.parquet")
    df = df[(df.arch == "legnet") & (df.d_train == 2000)].copy()
    df["source"] = "existing_sweeps"
    rows.append(df)

    # New high-dim results
    base = REPO / "results/preflight/legnet_d2k_highdim"
    for rj in base.rglob("result.json"):
        try:
            r = json.loads(rj.read_text())
            hp = r.get("hp", {}) or {}
            row = {
                "source": "legnet_d2k_highdim",
                "arch": r["arch"],
                "d_train": r.get("d_train"),
                "seed": r.get("seed"),
                "epochs": r.get("epochs"),
                "aug": r.get("augmentations"),
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


def main():
    df = load_all_legnet_d2k()
    df = df.dropna(subset=["test_mse"])
    n_existing = (df.source == "existing_sweeps").sum()
    n_new = (df.source == "legnet_d2k_highdim").sum()
    print(f"LegNet D=2000: {n_existing} existing + {n_new} new = {len(df)} total cells")

    # 6-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: LR × BS heatmap (existing data)
    ax = axes[0, 0]
    cell = df.dropna(subset=["lr", "batch_size", "test_mse"])
    if not cell.empty:
        piv = cell.groupby(["lr", "batch_size"])["test_mse"].min().reset_index().pivot(
            index="lr", columns="batch_size", values="test_mse"
        )
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                    cbar_kws={"label": "test_mse"})
        # Mark min
        flat = piv.stack()
        if not flat.empty:
            mlr, mbs = flat.idxmin()
            r = list(piv.index).index(mlr); c = list(piv.columns).index(mbs)
            ax.plot(c + 0.5, r + 0.5, "r*", markersize=22, markeredgecolor="white",
                    markeredgewidth=1.5)
        ax.set_title(f"LR × BS (n={len(cell)})")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Learning Rate")

    # Panel 2: Dropout × Aug heatmap (NEW)
    ax = axes[0, 1]
    cell = df.dropna(subset=["dropout", "aug", "test_mse"])
    if not cell.empty:
        piv = cell.groupby(["dropout", "aug"])["test_mse"].mean().reset_index().pivot(
            index="dropout", columns="aug", values="test_mse"
        )
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                    cbar_kws={"label": "test_mse"})
        flat = piv.stack()
        if not flat.empty:
            mdrop, maug = flat.idxmin()
            r = list(piv.index).index(mdrop); c = list(piv.columns).index(maug)
            ax.plot(c + 0.5, r + 0.5, "r*", markersize=22, markeredgecolor="white",
                    markeredgewidth=1.5)
        ax.set_title(f"Dropout × Aug (n={len(cell)})")
        ax.set_xlabel("Augmentation")
        ax.set_ylabel("Dropout")
    else:
        ax.text(0.5, 0.5, "Dropout × Aug sweep\nstill running",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Dropout × Aug (NEW)")

    # Panel 3: LR × WD heatmap
    ax = axes[0, 2]
    cell = df.dropna(subset=["lr", "weight_decay", "test_mse"])
    if not cell.empty:
        piv = cell.groupby(["lr", "weight_decay"])["test_mse"].min().reset_index().pivot(
            index="lr", columns="weight_decay", values="test_mse"
        )
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                    cbar_kws={"label": "test_mse"})
        flat = piv.stack()
        if not flat.empty:
            mlr, mwd = flat.idxmin()
            r = list(piv.index).index(mlr); c = list(piv.columns).index(mwd)
            ax.plot(c + 0.5, r + 0.5, "r*", markersize=22, markeredgecolor="white",
                    markeredgewidth=1.5)
        ax.set_title(f"LR × Weight Decay (n={len(cell)})")
        ax.set_xlabel("Weight Decay")
        ax.set_ylabel("Learning Rate")

    # Panel 4: 1D Dropout walk
    ax = axes[1, 0]
    cell = df.dropna(subset=["dropout", "test_mse"])
    if not cell.empty:
        agg = cell.groupby("dropout")["test_mse"].agg(["mean", "min", "max", "count"]).reset_index()
        ax.errorbar(agg.dropout, agg["mean"],
                    yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
                    marker="o", capsize=5, color="steelblue", markersize=8)
        for _, r in agg.iterrows():
            ax.annotate(f"n={int(r['count'])}", (r.dropout, r["mean"]),
                        fontsize=8, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Dropout rate")
        ax.set_ylabel("Test MSE")
        ax.set_title("Dropout 1D walk (NEW)")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Dropout walk\nstill running",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    # Panel 5: 1D Aug walk
    ax = axes[1, 1]
    cell = df.dropna(subset=["aug", "test_mse"])
    if not cell.empty:
        agg = cell.groupby("aug")["test_mse"].agg(["mean", "min", "max", "count"]).reset_index()
        ax.bar(range(len(agg)), agg["mean"],
               yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
               capsize=5, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg.aug, rotation=20, ha="right", fontsize=9)
        for x, (_, r) in enumerate(agg.iterrows()):
            ax.text(x, r["mean"] + (agg["max"].max() - agg["min"].min()) * 0.02,
                    f"n={int(r['count'])}", ha="center", fontsize=8)
        ax.set_ylabel("Test MSE")
        ax.set_title("Augmentation 1D walk (NEW)")
        ax.grid(alpha=0.3, axis="y")

    # Panel 6: 1D Epochs walk
    ax = axes[1, 2]
    cell = df.dropna(subset=["epochs", "test_mse"])
    if not cell.empty and cell.epochs.nunique() > 1:
        agg = cell.groupby("epochs")["test_mse"].agg(["mean", "min", "max", "count"]).reset_index()
        ax.errorbar(agg.epochs, agg["mean"],
                    yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
                    marker="o", capsize=5, color="steelblue", markersize=8)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Test MSE")
        ax.set_title("Epochs 1D walk (NEW)")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Epochs walk\nstill running",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    fig.suptitle(
        f"LegNet @ D=2000 high-dimensional HP optimization "
        f"({len(df)} runs across 6 HP dimensions)\n"
        "Anchor: LR=0.0005, BS=128, WD=0.1 — exploring dropout, aug, capacity, epochs",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
