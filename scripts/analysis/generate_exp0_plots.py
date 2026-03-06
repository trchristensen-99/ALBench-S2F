#!/usr/bin/env python3
"""Generate comprehensive Experiment 0 scaling plots.

Reads result.json files from all completed experiments and generates
multi-panel comparison plots for both K562 and Yeast datasets.

Run from repo root:
    python scripts/analysis/generate_exp0_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "exp0_plots"

# ── Color / marker scheme ─────────────────────────────────────────────────────
# K562 uses AlphaGenome as the oracle ensemble.
# Yeast uses DREAM-RNN as the oracle ensemble.
STYLE = {
    # K562
    "DREAM-RNN (real labels)": dict(color="#E05E4B", marker="o", ls="-"),
    "DREAM-RNN (AlphaGenome Ensemble labels)": dict(color="#E05E4B", marker="o", ls="--"),
    "AlphaGenome (real labels, frozen)": dict(color="#4B7BE0", marker="s", ls="-"),
    "AlphaGenome (AlphaGenome Ensemble labels, frozen)": dict(color="#4B7BE0", marker="s", ls="--"),
    # Yeast
    "DREAM-RNN (real labels) ": dict(
        color="#E05E4B", marker="o", ls="-"
    ),  # trailing space distinguishes
    "DREAM-RNN (DREAM-RNN Ensemble labels)": dict(color="#E05E4B", marker="o", ls="--"),
    "AlphaGenome frozen (partial)": dict(color="#4B7BE0", marker="s", ls="-"),
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_results(
    results_dir: Path,
    *,
    require_n_total: int | None = None,
) -> list[dict]:
    records = []
    for p in sorted(results_dir.rglob("result.json")):
        with open(p) as f:
            d = json.load(f)
        if require_n_total is not None and d.get("n_total") != require_n_total:
            continue
        d["_path"] = str(p)
        records.append(d)
    return records


def make_df(records: list[dict], model: str, metric_keys: dict[str, str]) -> pd.DataFrame:
    """Flatten result.json records into a DataFrame.

    metric_keys maps output column name -> (test_metrics sub-key, metric field).
    E.g. {"test_id": ("in_distribution", "pearson_r")}
    """
    rows = []
    for r in records:
        tm = r.get("test_metrics", {})
        row = {
            "model": model,
            "fraction": round(float(r["fraction"]), 6),
            "n_samples": r.get("n_samples"),
            "val_pearson": r.get("best_val_pearson_r"),
        }
        for col, (sub_key, field) in metric_keys.items():
            val = tm.get(sub_key, {}).get(field)
            row[col] = val
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = df.dropna(subset=[metric])
    if valid.empty:
        return pd.DataFrame(columns=["fraction", "mean", "std", "n", "n_samples_mean"])
    agg = (
        valid.groupby("fraction")
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            n=(metric, "count"),
            n_samples_mean=("n_samples", "mean"),
        )
        .reset_index()
        .sort_values("fraction")
    )
    return agg


# ── Oracle ensemble baselines ─────────────────────────────────────────────────
def load_k562_oracle_baselines() -> dict[str, float]:
    """Load the true ensemble prediction scores (not mean of individual folds)."""
    summary_path = REPO / "outputs" / "oracle_pseudolabels_k562_ag" / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        s = json.load(f)
    baselines = {}
    key_map = {
        "test_id": "test_in_distribution",
        "test_ood": "test_ood",
        "test_snv_abs": "test_snv_alt",
        "test_snv_delta": "test_snv_delta",
    }
    for col, summary_key in key_map.items():
        v = s.get(summary_key, {}).get("ensemble_pearson_r")
        if v is not None:
            baselines[col] = float(v)
    return baselines


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _format_n_samples(v: float, _pos) -> str:
    """Format absolute sample count for x-axis tick labels."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return f"{v:.0f}"


def plot_scaling_panel(
    ax: plt.Axes,
    dfs: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    title: str,
    oracle_baseline: float | None = None,
    oracle_baseline_label: str = "AlphaGenome Ensemble",
    ylim: tuple[float, float] | None = None,
):
    for label, df in dfs.items():
        agg = aggregate(df, metric)
        if agg.empty:
            continue
        style = STYLE.get(label, dict(color="gray", marker="^", ls="-"))
        x = agg["n_samples_mean"]
        ax.errorbar(
            x,
            agg["mean"],
            yerr=agg["std"].fillna(0),
            fmt=f"{style['ls']}{style['marker']}",
            color=style["color"],
            label=label,
            capsize=4,
            linewidth=2,
            markersize=6,
            zorder=3,
        )
    if oracle_baseline is not None:
        ax.axhline(
            oracle_baseline,
            color="#2CA02C",
            ls=":",
            lw=1.5,
            alpha=0.8,
            label=oracle_baseline_label,
            zorder=2,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_n_samples))
    ax.set_xlabel("Number of training sequences", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", ls="--", alpha=0.35, zorder=0)


# ── K562 plots ────────────────────────────────────────────────────────────────
K562_METRICS = {
    "test_id": ("in_distribution", "pearson_r"),
    "test_ood": ("ood", "pearson_r"),
    "test_snv_abs": ("snv_abs", "pearson_r"),
    "test_snv_delta": ("snv_delta", "pearson_r"),
}

K562_FRACS = {0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00}


def snap_k562_fracs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fraction"] = df["fraction"].apply(lambda f: min(K562_FRACS, key=lambda s: abs(s - f)))
    df = df[df["fraction"].isin(K562_FRACS)]
    return df


def generate_k562_plots():
    oracle_baselines = load_k562_oracle_baselines()
    print(
        f"  AlphaGenome Ensemble baselines: "
        f"{ {k: f'{v:.4f}' for k, v in oracle_baselines.items()} }"
    )

    dream_real = make_df(
        load_results(REPO / "outputs" / "exp0_k562_scaling"),
        "DREAM-RNN (real labels)",
        K562_METRICS,
    )
    dream_real = snap_k562_fracs(dream_real)

    ag_real = make_df(
        load_results(REPO / "outputs" / "exp0_k562_scaling_alphagenome_cached_rcaug"),
        "AlphaGenome (real labels, frozen)",
        K562_METRICS,
    )

    ag_oracle = make_df(
        load_results(REPO / "outputs" / "exp0_k562_scaling_oracle_labels_ag"),
        "AlphaGenome (AlphaGenome Ensemble labels, frozen)",
        K562_METRICS,
    )

    dream_oracle = make_df(
        load_results(REPO / "outputs" / "exp0_k562_scaling_oracle_labels"),
        "DREAM-RNN (AlphaGenome Ensemble labels)",
        K562_METRICS,
    )

    dfs = {}
    for label, df in [
        ("DREAM-RNN (real labels)", dream_real),
        ("AlphaGenome (real labels, frozen)", ag_real),
        ("DREAM-RNN (AlphaGenome Ensemble labels)", dream_oracle),
        ("AlphaGenome (AlphaGenome Ensemble labels, frozen)", ag_oracle),
    ]:
        if not df.empty:
            dfs[label] = df

    print(f"K562 data: {', '.join(f'{k}: {len(v)} records' for k, v in dfs.items())}")

    panels = [
        ("test_id", "Pearson R", "In-distribution"),
        ("test_ood", "Pearson R", "Out-of-distribution (designed)"),
        ("test_snv_abs", "Pearson R", "SNV absolute"),
        ("test_snv_delta", "Pearson R", "SNV effect (delta)"),
    ]

    # 4-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric, ylabel, title) in zip(axes.flatten(), panels):
        plot_scaling_panel(
            ax,
            dfs,
            metric,
            ylabel,
            title,
            oracle_baseline=oracle_baselines.get(metric),
            oracle_baseline_label="AlphaGenome Ensemble",
            ylim=(0, 1),
        )
    fig.suptitle("K562 MPRA — Exp 0 Scaling Curves (all conditions)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k562_all_conditions_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_all_conditions_4panel.png")

    # Real vs oracle comparison (in-dist + OOD, 2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel, title) in zip(axes, panels[:2]):
        plot_scaling_panel(
            ax,
            dfs,
            metric,
            ylabel,
            title,
            oracle_baseline=oracle_baselines.get(metric),
            oracle_baseline_label="AlphaGenome Ensemble",
            ylim=(0, 1),
        )
    fig.suptitle("K562 MPRA — Real vs AlphaGenome Ensemble Labels", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k562_real_vs_oracle_2panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_real_vs_oracle_2panel.png")

    # Single-panel in-dist
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_scaling_panel(
        ax,
        dfs,
        "test_id",
        "Pearson R",
        "K562 In-distribution Scaling",
        oracle_baseline=oracle_baselines.get("test_id"),
        oracle_baseline_label="AlphaGenome Ensemble",
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k562_in_dist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_in_dist.png")

    # Save combined CSV
    all_k562 = pd.concat(dfs.values(), ignore_index=True)
    all_k562.to_csv(OUT_DIR / "k562_all_results.csv", index=False)

    # Print summary table
    print("\n  K562 Summary (mean test in-dist Pearson R):")
    for label, df in dfs.items():
        agg = aggregate(df, "test_id")
        if agg.empty:
            continue
        print(f"    {label}:")
        for _, row in agg.iterrows():
            ns = int(row["n_samples_mean"])
            m = row["mean"]
            s = row["std"]
            n = int(row["n"])
            print(f"      n={ns:,}: {m:.4f} +/- {s:.4f} (seeds={n})")


# ── Yeast plots ───────────────────────────────────────────────────────────────
YEAST_METRICS = {
    "test_random": ("random", "pearson_r"),
    "test_genomic": ("genomic", "pearson_r"),
    "test_snv_abs": ("snv_abs", "pearson_r"),
    "test_snv": ("snv", "pearson_r"),
}

YEAST_FRACS = {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00}


def snap_yeast_fracs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fraction"] = df["fraction"].apply(lambda f: min(YEAST_FRACS, key=lambda s: abs(s - f)))
    df = df[df["fraction"].isin(YEAST_FRACS)]
    return df


def generate_yeast_plots():
    # Only include 6M runs (n_total=6065325); exclude old 100K runs
    dream_real = make_df(
        load_results(
            REPO / "outputs" / "exp0_yeast_scaling",
            require_n_total=6065325,
        ),
        "DREAM-RNN (real labels) ",  # trailing space distinguishes from K562 style key
        YEAST_METRICS,
    )
    dream_real = snap_yeast_fracs(dream_real)

    dream_oracle = make_df(
        load_results(
            REPO / "outputs" / "exp0_yeast_scaling_oracle_labels",
            require_n_total=6065325,
        ),
        "DREAM-RNN (DREAM-RNN Ensemble labels)",
        YEAST_METRICS,
    )
    dream_oracle = snap_yeast_fracs(dream_oracle)

    # AlphaGenome frozen-encoder (partial — only fractions up to 0.05, 1 seed each)
    ag_frozen = make_df(
        load_results(
            REPO / "outputs" / "exp0_yeast_scaling_alphagenome",
            require_n_total=6065325,
        ),
        "AlphaGenome frozen (partial)",
        YEAST_METRICS,
    )
    ag_frozen = snap_yeast_fracs(ag_frozen)

    dfs = {}
    for label, df in [
        ("DREAM-RNN (real labels) ", dream_real),
        ("DREAM-RNN (DREAM-RNN Ensemble labels)", dream_oracle),
        ("AlphaGenome frozen (partial)", ag_frozen),
    ]:
        if not df.empty:
            dfs[label] = df

    print(f"\nYeast data: {', '.join(f'{k}: {len(v)} records' for k, v in dfs.items())}")

    panels = [
        ("test_random", "Pearson R", "Random (in-distribution)"),
        ("test_genomic", "Pearson R", "Genomic (out-of-distribution)"),
        ("test_snv_abs", "Pearson R", "SNV absolute"),
        ("test_snv", "Pearson R", "SNV effect (delta)"),
    ]

    # 4-panel comparison: real vs DREAM-RNN Ensemble
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric, ylabel, title) in zip(axes.flatten(), panels):
        plot_scaling_panel(ax, dfs, metric, ylabel, title, ylim=(0, 1))
    fig.suptitle(
        "Yeast — Exp 0 Scaling Curves: Real vs DREAM-RNN Ensemble Labels",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yeast_real_vs_oracle_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: yeast_real_vs_oracle_4panel.png")

    # 2-panel: random (ID) + genomic (OOD)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel, title) in zip(axes, panels[:2]):
        plot_scaling_panel(ax, dfs, metric, ylabel, title, ylim=(0, 1))
    fig.suptitle(
        "Yeast — Real vs DREAM-RNN Ensemble Labels",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yeast_real_vs_oracle_2panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: yeast_real_vs_oracle_2panel.png")

    # Single-panel: val Pearson comparison
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, df in dfs.items():
        agg = aggregate(df, "val_pearson")
        if agg.empty:
            continue
        style = STYLE[label]
        x = agg["n_samples_mean"]
        ax.errorbar(
            x,
            agg["mean"],
            yerr=agg["std"].fillna(0),
            fmt=f"{style['ls']}{style['marker']}",
            color=style["color"],
            label=label,
            capsize=4,
            linewidth=2,
            markersize=6,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_n_samples))
    ax.set_xlabel("Number of training sequences", fontsize=10)
    ax.set_ylabel("Pearson R", fontsize=10)
    ax.set_title(
        "Yeast — Validation Pearson: Real vs DREAM-RNN Ensemble Labels",
        fontsize=11,
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", ls="--", alpha=0.35, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yeast_val_pearson_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: yeast_val_pearson_comparison.png")

    # Save combined CSV
    all_yeast = pd.concat(dfs.values(), ignore_index=True)
    all_yeast.to_csv(OUT_DIR / "yeast_all_results.csv", index=False)

    # Print summary
    print("\n  Yeast Summary (mean test random Pearson R):")
    for label, df in dfs.items():
        agg = aggregate(df, "test_random")
        if agg.empty:
            continue
        print(f"    {label}:")
        for _, row in agg.iterrows():
            ns = int(row["n_samples_mean"])
            m = row["mean"]
            s = row["std"]
            n = int(row["n"])
            print(f"      n={ns:,}: {m:.4f} +/- {s:.4f} (seeds={n})")

    print("\n  Yeast Summary (mean val Pearson R):")
    for label, df in dfs.items():
        agg = aggregate(df, "val_pearson")
        if agg.empty:
            continue
        print(f"    {label}:")
        for _, row in agg.iterrows():
            ns = int(row["n_samples_mean"])
            m = row["mean"]
            s = row["std"]
            n = int(row["n"])
            print(f"      n={ns:,}: {m:.4f} +/- {s:.4f} (seeds={n})")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")
    generate_k562_plots()
    generate_yeast_plots()
    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
