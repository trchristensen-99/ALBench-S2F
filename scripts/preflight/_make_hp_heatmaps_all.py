"""Generate comprehensive HP heatmaps from ALL preflight result.json files.

For each (arch, D) cell:
  - LR × BS heatmap with test_mse coloring
  - Marks local minimum with red star
  - Shows N configs tested per cell
  - Highlights edge optima

Loads results from any preflight subdir (no hardcoded list).

Outputs:
  results/preflight/figures/hp_heatmap_{arch}_d{D}.png
  results/preflight/figures/hp_heatmap_grid_{arch}.png  (multi-panel per arch)
  results/preflight/figures/hp_coverage_summary.png
  results/preflight/all_hp_results.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "results" / "preflight"
OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all() -> pd.DataFrame:
    rows = []
    for rj in RESULTS_DIR.rglob("result.json"):
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        if "arch" not in r or r.get("test_mse_at_best_val") is None:
            continue
        hp = r.get("hp", {}) or {}
        rows.append(
            {
                "task": rj.relative_to(RESULTS_DIR).parts[0],
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
                "dropout_cnn": hp.get("dropout_cnn"),
                "dropout_lstm": hp.get("dropout_lstm"),
                "dropout": hp.get("dropout") or hp.get("dropout_rate"),
                "in_channels": hp.get("in_channels"),
            }
        )
    return pd.DataFrame(rows)


def lr_bs_heatmap(df: pd.DataFrame, arch: str, d: int, out_path: Path) -> bool:
    """Heatmap of LR × BS for a single (arch, D) cell. Returns True if drew."""
    sub = df[(df.arch == arch) & (df.d_train == d)].copy()
    sub = sub.dropna(subset=["lr", "batch_size", "test_mse"])
    if len(sub) < 4:
        return False
    # take min across seeds at each (lr, bs)
    agg = sub.groupby(["lr", "batch_size"])["test_mse"].agg(["min", "mean", "count"]).reset_index()
    piv_min = agg.pivot(index="lr", columns="batch_size", values="min")
    piv_count = agg.pivot(index="lr", columns="batch_size", values="count")

    fig, ax = plt.subplots(
        figsize=(max(5, 0.6 * len(piv_min.columns) + 3), max(4, 0.6 * len(piv_min.index) + 2))
    )
    annot = (
        piv_min.round(3).astype(str) + "\n(n=" + piv_count.fillna(0).astype(int).astype(str) + ")"
    )
    annot[piv_min.isna()] = ""
    sns.heatmap(
        piv_min,
        annot=annot,
        fmt="",
        cmap="viridis_r",
        cbar_kws={"label": "test_mse (min)"},
        ax=ax,
        linewidths=0.3,
        linecolor="white",
    )
    # Mark the local min
    if not piv_min.dropna(how="all").empty:
        flat = piv_min.stack()
        if not flat.empty:
            min_lr, min_bs = flat.idxmin()
            row_idx = list(piv_min.index).index(min_lr)
            col_idx = list(piv_min.columns).index(min_bs)
            ax.plot(
                col_idx + 0.5,
                row_idx + 0.5,
                "r*",
                markersize=22,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
    n_total = len(sub)
    n_unique = len(agg)
    ax.set_title(f"{arch} | D={d:,} | {n_total} runs across {n_unique} (lr, BS) cells", fontsize=11)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Learning rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def grid_per_arch(df: pd.DataFrame, arch: str, out_path: Path):
    """Multi-panel: one LR × BS heatmap per D, all in one figure for the arch."""
    sub = df[df.arch == arch].dropna(subset=["lr", "batch_size", "test_mse"])
    if sub.empty:
        return
    ds = sorted(int(d) for d in sub.d_train.dropna().unique())
    if not ds:
        return
    n_cols = min(4, len(ds))
    n_rows = (len(ds) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for i, d in enumerate(ds):
        ax = axes[i // n_cols][i % n_cols]
        cell = sub[sub.d_train == d]
        agg = cell.groupby(["lr", "batch_size"])["test_mse"].min().reset_index()
        piv = agg.pivot(index="lr", columns="batch_size", values="test_mse")
        if piv.empty:
            ax.set_title(f"D={d}\n(no data)")
            ax.axis("off")
            continue
        sns.heatmap(
            piv,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",
            ax=ax,
            cbar=False,
            linewidths=0.2,
            linecolor="white",
            annot_kws={"size": 7},
        )
        # Mark min
        flat = piv.stack()
        if not flat.empty:
            mlr, mbs = flat.idxmin()
            r = list(piv.index).index(mlr)
            c = list(piv.columns).index(mbs)
            ax.plot(
                c + 0.5, r + 0.5, "r*", markersize=18, markeredgecolor="white", markeredgewidth=1.0
            )
        ax.set_title(f"D={d:,} (n={len(cell)})", fontsize=10)
        ax.set_xlabel("BS")
        ax.set_ylabel("LR")
    # Hide unused
    for i in range(len(ds), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")
    fig.suptitle(f"{arch}: LR × BS heatmap per D (red ⋆ = local min, lower=better)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


def coverage_summary(df: pd.DataFrame, out_path: Path):
    """Summary of how many configs per (arch, D)."""
    counts = df.groupby(["arch", "d_train"]).size().reset_index(name="n")
    archs = sorted(counts.arch.unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4), squeeze=False)
    for i, arch in enumerate(archs):
        ax = axes[0][i]
        sub = counts[counts.arch == arch].sort_values("d_train")
        ax.bar([str(int(d)) for d in sub.d_train], sub.n, color="steelblue")
        for x, y in zip(range(len(sub)), sub.n):
            ax.text(x, y + 1, str(y), ha="center", fontsize=8)
        ax.set_xlabel("D")
        ax.set_ylabel("# configs tested")
        ax.set_title(f"{arch}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("HP coverage: configs tested per (arch, D) — total runs", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


def best_per_d(df: pd.DataFrame, out_path: Path):
    """Best test_mse achieved per D for each arch — local minimum quality."""
    sub = df.dropna(subset=["test_mse", "d_train"])
    archs = sorted(sub.arch.unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"legnet": "#1f77b4", "dream_rnn": "#ff7f0e", "dream_attn": "#2ca02c"}
    for arch in archs:
        a = sub[sub.arch == arch]
        agg = a.groupby("d_train")["test_mse"].agg(["min", "mean", "count"]).reset_index()
        ax.plot(
            agg.d_train,
            agg["min"],
            "o-",
            label=f"{arch} (best)",
            color=colors.get(arch, "k"),
            markersize=8,
        )
        ax.fill_between(
            agg.d_train, agg["min"], agg["mean"], alpha=0.15, color=colors.get(arch, "k")
        )
    ax.set_xscale("log")
    ax.set_xlabel("D (log scale)")
    ax.set_ylabel("test_mse")
    ax.set_title("Best HP found per D (line = min; band = min→mean) — lower is better")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    df = load_all()
    print(f"Loaded {len(df):,} rows from results/preflight/**/result.json")
    print(f"  archs: {sorted(df.arch.unique())}")
    print(f"  D values: {sorted(int(d) for d in df.d_train.dropna().unique())}")
    print(f"  tasks: {df.task.value_counts().to_dict()}")
    df.to_parquet(RESULTS_DIR / "all_hp_results.parquet")

    print("\n=== Per-(arch, D) heatmaps ===")
    for arch in sorted(df.arch.unique()):
        sub = df[df.arch == arch]
        for d in sorted(int(x) for x in sub.d_train.dropna().unique()):
            out = OUT_DIR / f"hp_heatmap_{arch}_d{d}.png"
            ok = lr_bs_heatmap(df, arch, d, out)
            if ok:
                print(f"  Saved {out.name}")

    print("\n=== Grid heatmap per arch (all D in one figure) ===")
    for arch in sorted(df.arch.unique()):
        grid_per_arch(df, arch, OUT_DIR / f"hp_heatmap_grid_{arch}.png")

    print("\n=== Coverage + best-per-D summaries ===")
    coverage_summary(df, OUT_DIR / "hp_coverage_summary.png")
    best_per_d(df, OUT_DIR / "hp_best_per_d.png")

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
