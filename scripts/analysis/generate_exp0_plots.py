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
    "DREAM-RNN (real)": dict(color="#E05E4B", marker="o", ls="-"),
    "DREAM-RNN (oracle)": dict(color="#E05E4B", marker="o", ls="--"),
    "AlphaGenome S1 (frozen)": dict(color="#4B7BE0", marker="s", ls="-"),
    "AlphaGenome S2 (fine-tuned)": dict(color="#2CA02C", marker="D", ls="-"),
    # Short names (used in real-label-only plots)
    "DREAM-RNN": dict(color="#E05E4B", marker="o", ls="-"),
    "AlphaGenome": dict(color="#4B7BE0", marker="s", ls="-"),
}

# Map verbose real-label keys to short legend labels
_SHORT_LABELS = {
    "DREAM-RNN (real labels)": "DREAM-RNN",
    "AlphaGenome (real labels, frozen)": "AlphaGenome",
    "DREAM-RNN (real)": "DREAM-RNN",
    "AlphaGenome S1 (frozen)": "AlphaGenome S1",
    "AlphaGenome S2 (fine-tuned)": "AlphaGenome S2",
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_results(
    results_dir: Path,
    *,
    require_n_total: int | None = None,
    require_test_keys: set[str] | None = None,
) -> list[dict]:
    records = []
    for p in sorted(results_dir.rglob("result.json")):
        with open(p) as f:
            d = json.load(f)
        if require_n_total is not None and d.get("n_total") != require_n_total:
            continue
        if require_test_keys is not None:
            tm = d.get("test_metrics", {})
            if not require_test_keys.issubset(tm.keys()):
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
    show_legend: bool = True,
):
    for label, df in dfs.items():
        agg = aggregate(df, metric)
        if agg.empty:
            continue
        style = STYLE.get(label, dict(color="gray", marker="^", ls="-"))
        x = agg["n_samples_mean"]
        yerr = agg["std"].fillna(0)
        # Clamp error bars so they don't extend below 0
        means = agg["mean"]
        yerr_lo = np.clip(yerr, 0, np.maximum(means, 0))
        ax.errorbar(
            x,
            means,
            yerr=[yerr_lo, yerr],
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
    else:
        # Default: start y-axis slightly below 0 so small negative values are visible
        ax.set_ylim(bottom=-0.1)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", ls="--", alpha=0.35, zorder=0)


def _add_shared_legend(fig, axes, *, ncol=None, loc="upper center", y_offset=-0.02):
    """Collect unique legend handles from all axes and add one figure legend."""
    handles, labels = [], []
    seen = set()
    for ax in axes.flatten() if hasattr(axes, "flatten") else [axes]:
        for handle, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen:
                handles.append(handle)
                labels.append(lbl)
                seen.add(lbl)
    if not handles:
        return
    if ncol is None:
        ncol = min(len(handles), 3)
    fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        fontsize=9,
        framealpha=0.9,
    )


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

    # Require all 4 test metric keys to exclude old runs that used a different
    # OOD test set (run_05/run_10/run_25 — missing snv_abs, spurious OOD values).
    _k562_test_keys = {"in_distribution", "ood", "snv_abs", "snv_delta"}
    # Use v2 results (batch_size=128) if available, fall back to v1 (batch_size=1024)
    _dream_real_dir = REPO / "outputs" / "exp0_k562_scaling_v2"
    if not _dream_real_dir.exists() or not list(_dream_real_dir.rglob("result.json")):
        _dream_real_dir = REPO / "outputs" / "exp0_k562_scaling"
    dream_real = make_df(
        load_results(
            _dream_real_dir,
            require_test_keys=_k562_test_keys,
        ),
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

    _dream_oracle_dir = REPO / "outputs" / "exp0_k562_scaling_oracle_labels_v2"
    if not _dream_oracle_dir.exists() or not list(_dream_oracle_dir.rglob("result.json")):
        _dream_oracle_dir = REPO / "outputs" / "exp0_k562_scaling_oracle_labels"
    dream_oracle = make_df(
        load_results(_dream_oracle_dir),
        "DREAM-RNN (AlphaGenome Ensemble labels)",
        K562_METRICS,
    )

    dfs_all = {}
    for label, df in [
        ("DREAM-RNN (real labels)", dream_real),
        ("AlphaGenome (real labels, frozen)", ag_real),
        ("DREAM-RNN (AlphaGenome Ensemble labels)", dream_oracle),
        ("AlphaGenome (AlphaGenome Ensemble labels, frozen)", ag_oracle),
    ]:
        if not df.empty:
            dfs_all[label] = df

    dfs_real = {k: v for k, v in dfs_all.items() if "Ensemble labels" not in k}

    print(f"K562 data: {', '.join(f'{k}: {len(v)} records' for k, v in dfs_all.items())}")

    panels = [
        ("test_id", "Pearson R", "In-distribution"),
        ("test_ood", "Pearson R", "Out-of-distribution (designed)"),
        ("test_snv_abs", "Pearson R", "SNV absolute"),
        ("test_snv_delta", "Pearson R", "SNV effect (delta)"),
    ]

    # 4-panel: ALL conditions (real + oracle)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric, ylabel, title) in zip(axes.flatten(), panels):
        plot_scaling_panel(
            ax,
            dfs_all,
            metric,
            ylabel,
            title,
            oracle_baseline=oracle_baselines.get(metric),
            oracle_baseline_label="AlphaGenome Ensemble",
            ylim=(-0.1, 1),
            show_legend=False,
        )
    fig.suptitle("K562 MPRA — Exp 0 Scaling Curves (all conditions)", fontsize=14, y=1.02)
    _add_shared_legend(fig, axes, ncol=3, y_offset=-0.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_DIR / "k562_all_conditions_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_all_conditions_4panel.png")

    # 4-panel: REAL LABELS ONLY
    dfs_real_short = {_SHORT_LABELS.get(k, k): v for k, v in dfs_real.items()}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric, ylabel, title) in zip(axes.flatten(), panels):
        plot_scaling_panel(
            ax,
            dfs_real_short,
            metric,
            ylabel,
            title,
            ylim=(-0.1, 1),
            show_legend=False,
        )
    fig.suptitle("K562 MPRA — Exp 0 Scaling Curves (real labels only)", fontsize=14, y=1.02)
    _add_shared_legend(fig, axes, y_offset=0.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / "k562_real_labels_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_real_labels_4panel.png")

    # 2-panel: real vs oracle comparison (in-dist + OOD)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel, title) in zip(axes, panels[:2]):
        plot_scaling_panel(
            ax,
            dfs_all,
            metric,
            ylabel,
            title,
            oracle_baseline=oracle_baselines.get(metric),
            oracle_baseline_label="AlphaGenome Ensemble",
            ylim=(-0.1, 1),
            show_legend=False,
        )
    fig.suptitle("K562 MPRA — Real vs AlphaGenome Ensemble Labels", fontsize=13, y=1.02)
    _add_shared_legend(fig, axes, ncol=3, y_offset=-0.06)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(OUT_DIR / "k562_real_vs_oracle_2panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_real_vs_oracle_2panel.png")

    # 2-panel: REAL LABELS ONLY (in-dist + OOD)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel, title) in zip(axes, panels[:2]):
        plot_scaling_panel(
            ax,
            dfs_real_short,
            metric,
            ylabel,
            title,
            ylim=(-0.1, 1),
            show_legend=False,
        )
    fig.suptitle("K562 MPRA — Scaling Curves (real labels)", fontsize=13, y=1.02)
    _add_shared_legend(fig, axes, y_offset=-0.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_DIR / "k562_real_labels_2panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_real_labels_2panel.png")

    # Single-panel in-dist (all conditions)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_scaling_panel(
        ax,
        dfs_all,
        "test_id",
        "Pearson R",
        "K562 In-distribution Scaling",
        oracle_baseline=oracle_baselines.get("test_id"),
        oracle_baseline_label="AlphaGenome Ensemble",
        ylim=(-0.1, 1),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k562_in_dist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_in_dist.png")

    # Save combined CSV
    all_k562 = pd.concat(dfs_all.values(), ignore_index=True)
    all_k562.to_csv(OUT_DIR / "k562_all_results.csv", index=False)

    # Print summary table
    print("\n  K562 Summary (mean test in-dist Pearson R):")
    for label, df in dfs_all.items():
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
    # DREAM-RNN real labels: prefer v2 (optimized HPs), fall back to original
    _dream_real_dir = REPO / "outputs" / "exp0_yeast_scaling_v2"
    if not _dream_real_dir.exists() or not list(_dream_real_dir.rglob("result.json")):
        _dream_real_dir = REPO / "outputs" / "exp0_yeast_scaling"
        print("  DREAM-RNN yeast: using original (pre-HP-optimization) results")
    else:
        print("  DREAM-RNN yeast: using v2 (optimized HP) results")
    dream_real = make_df(
        load_results(_dream_real_dir, require_n_total=6065325),
        "DREAM-RNN (real)",
        YEAST_METRICS,
    )
    dream_real = snap_yeast_fracs(dream_real)

    # DREAM-RNN oracle labels: prefer v2, fall back to original
    _dream_oracle_dir = REPO / "outputs" / "exp0_yeast_scaling_oracle_labels_v2"
    if not _dream_oracle_dir.exists() or not list(_dream_oracle_dir.rglob("result.json")):
        _dream_oracle_dir = REPO / "outputs" / "exp0_yeast_scaling_oracle_labels"
    dream_oracle = make_df(
        load_results(_dream_oracle_dir, require_n_total=6065325),
        "DREAM-RNN (oracle)",
        YEAST_METRICS,
    )
    dream_oracle = snap_yeast_fracs(dream_oracle)

    # AlphaGenome S1 frozen-encoder: prefer v2 (optimized head HPs), fall back
    _ag_s1_dir = REPO / "outputs" / "exp0_yeast_scaling_ag_v2"
    if not _ag_s1_dir.exists() or not list(_ag_s1_dir.rglob("result.json")):
        _ag_s1_dir = REPO / "outputs" / "exp0_yeast_scaling_alphagenome"
        print("  AG yeast S1: using original results")
    else:
        print("  AG yeast S1: using v2 (optimized head HP) results")
    ag_s1 = make_df(
        load_results(_ag_s1_dir, require_n_total=6065325),
        "AlphaGenome S1 (frozen)",
        YEAST_METRICS,
    )
    ag_s1 = snap_yeast_fracs(ag_s1)

    # AlphaGenome S2 fine-tuned encoder (if available)
    _ag_s2_dir = REPO / "outputs" / "exp0_yeast_scaling_ag_s2"
    ag_s2 = make_df(
        load_results(_ag_s2_dir, require_n_total=6065325) if _ag_s2_dir.exists() else [],
        "AlphaGenome S2 (fine-tuned)",
        YEAST_METRICS,
    )
    ag_s2 = snap_yeast_fracs(ag_s2)

    dfs_all = {}
    for label, df in [
        ("DREAM-RNN (real)", dream_real),
        ("DREAM-RNN (oracle)", dream_oracle),
        ("AlphaGenome S1 (frozen)", ag_s1),
        ("AlphaGenome S2 (fine-tuned)", ag_s2),
    ]:
        if not df.empty:
            dfs_all[label] = df

    dfs_real = {k: v for k, v in dfs_all.items() if "oracle" not in k.lower()}

    print(f"\nYeast data: {', '.join(f'{k}: {len(v)} records' for k, v in dfs_all.items())}")

    panels = [
        ("test_random", "Pearson R", "Random (in-distribution)"),
        ("test_genomic", "Pearson R", "Genomic (out-of-distribution)"),
        ("test_snv_abs", "Pearson R", "SNV absolute"),
        ("test_snv", "Pearson R", "SNV effect (delta)"),
    ]

    # 4-panel: ALL conditions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric, ylabel, title) in zip(axes.flatten(), panels):
        plot_scaling_panel(ax, dfs_all, metric, ylabel, title, ylim=(-0.1, 1), show_legend=False)
    fig.suptitle("Yeast — Exp 0 Scaling Curves (all conditions)", fontsize=14, y=1.02)
    _add_shared_legend(fig, axes, y_offset=-0.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_DIR / "yeast_all_conditions_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: yeast_all_conditions_4panel.png")

    # 4-panel: REAL LABELS ONLY
    dfs_real_short = {_SHORT_LABELS.get(k, k): v for k, v in dfs_real.items()}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric, ylabel, title) in zip(axes.flatten(), panels):
        plot_scaling_panel(
            ax, dfs_real_short, metric, ylabel, title, ylim=(-0.1, 1), show_legend=False
        )
    fig.suptitle("Yeast — Exp 0 Scaling Curves (real labels)", fontsize=14, y=1.02)
    _add_shared_legend(fig, axes, y_offset=0.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / "yeast_real_labels_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: yeast_real_labels_4panel.png")

    # 2-panel: ALL conditions (random + genomic)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel, title) in zip(axes, panels[:2]):
        plot_scaling_panel(ax, dfs_all, metric, ylabel, title, ylim=(-0.1, 1), show_legend=False)
    fig.suptitle("Yeast — All Conditions", fontsize=13, y=1.02)
    _add_shared_legend(fig, axes, y_offset=-0.06)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(OUT_DIR / "yeast_all_conditions_2panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: yeast_all_conditions_2panel.png")

    # 2-panel: REAL LABELS ONLY (random + genomic)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel, title) in zip(axes, panels[:2]):
        plot_scaling_panel(
            ax, dfs_real_short, metric, ylabel, title, ylim=(-0.1, 1), show_legend=False
        )
    fig.suptitle("Yeast — Scaling Curves (real labels)", fontsize=13, y=1.02)
    _add_shared_legend(fig, axes, y_offset=-0.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_DIR / "yeast_real_labels_2panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: yeast_real_labels_2panel.png")

    # Save combined CSV
    all_yeast = pd.concat(dfs_all.values(), ignore_index=True)
    all_yeast.to_csv(OUT_DIR / "yeast_all_results.csv", index=False)

    # Print summary
    print("\n  Yeast Summary (mean test random Pearson R):")
    for label, df in dfs_all.items():
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


# ── K562 bar plot (full-dataset 3-model comparison) ──────────────────────────

# Test set sizes (from hashfrag data)
_K562_TEST_SIZES = {
    "in_distribution": 40_718,
    "ood": 22_862,
    "snv_abs": 35_226,
    "snv_delta": 35_226,
}


def _load_bar_model_metrics(results_dir: Path, json_name: str = "result.json") -> list[dict]:
    """Load test_metrics from all result JSON files in a directory tree."""
    metrics_list = []
    for p in sorted(results_dir.rglob(json_name)):
        with open(p) as f:
            d = json.load(f)
        tm = d.get("test_metrics", d)  # test_metrics.json has metrics at top level
        if "in_distribution" in tm:
            metrics_list.append(tm)
    return metrics_list


def _extract_pearson(metrics_list: list[dict], test_key: str) -> list[float]:
    return [m[test_key]["pearson_r"] for m in metrics_list if test_key in m]


def generate_k562_bar_plot():
    """8-model bar plot comparing all methods on full K562 MPRA."""
    # ── Model definitions: (name, dir, json_name, color) ─────────────────────
    # Train-from-scratch in purple, foundation S1 in muted, S2 in vivid, AG green.
    models = [
        ("DREAM-RNN", "dream_rnn_k562_3seeds", "result.json", "#7B2D8E"),
        ("Malinois", "malinois_k562_basset_pretrained", "result.json", "#B07CC6"),
        (
            "Nucleotide Transformer (v3)",
            "ntv3_post_k562_stage2/sweep_elr1e-4_uf12",
            "result_eval.json",
            "#E8602C",
        ),
        ("Borzoi", "borzoi_k562_cached_v2", "result.json", "#DAA520"),
        ("Enformer", "enformer_k562_3seeds", "result.json", "#3A86C8"),
        ("AlphaGenome", "stage2_k562_full_train", "test_metrics.json", "#2CA02C"),
    ]

    all_metrics = {}
    for name, dirname, json_name, _ in models:
        d = REPO / "outputs" / dirname
        all_metrics[name] = _load_bar_model_metrics(d, json_name)

    # Fallback: if Borzoi v2 not ready, use original S1 results
    if not all_metrics.get("Borzoi"):
        fallback = REPO / "outputs" / "borzoi_k562_3seeds"
        fb_data = _load_bar_model_metrics(fallback, "result.json")
        if fb_data:
            all_metrics["Borzoi"] = fb_data
            print("  Borzoi: using original S1 results as fallback")

    # Fallback: if Enformer 3-seed not ready, use grid search best (single seed)
    if not all_metrics.get("Enformer"):
        gs_dir = REPO / "outputs" / "foundation_grid_search" / "enformer"
        fb_data = _load_bar_model_metrics(gs_dir, "result.json")
        if fb_data:
            # Pick only the best by val Pearson
            best = max(fb_data, key=lambda m: m.get("in_distribution", {}).get("pearson_r", 0))
            all_metrics["Enformer"] = [best]
            print("  Enformer: using grid search best (1 seed) as fallback")

    counts = {name: len(m) for name, m in all_metrics.items()}
    print(f"  Bar plot data: {counts}")

    if all(n == 0 for n in counts.values()):
        print("  Skipping bar plot: no results yet")
        return

    # ── Build per-metric arrays ───────────────────────────────────────────────
    test_keys = [
        ("in_distribution", "Reference"),
        ("snv_abs", "SNV"),
        ("snv_delta", "SNV effect (delta)"),
        ("ood", "Synthetic design"),
    ]

    labels = []
    model_means = {name: [] for name, *_ in models}
    model_stds = {name: [] for name, *_ in models}

    for key, display in test_keys:
        n_seqs = _K562_TEST_SIZES.get(key, "?")
        labels.append(f"{display}\n(n={n_seqs:,})")

        for name, *_ in models:
            vals = _extract_pearson(all_metrics[name], key)
            model_means[name].append(np.mean(vals) if vals else 0)
            model_stds[name].append(np.std(vals) if len(vals) > 1 else 0)

    n_models = sum(1 for name, *_ in models if any(v > 0 for v in model_means[name]))
    if n_models == 0:
        print("  Skipping bar plot: no non-zero results")
        return

    x = np.arange(len(labels))
    width = 0.8 / max(n_models, 1)
    offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    bar_groups = []
    offset_idx = 0
    for name, _, _, color in models:
        means = model_means[name]
        if not any(v > 0 for v in means):
            continue
        bars = ax.bar(
            x + offsets[offset_idx],
            means,
            width,
            label=name,
            color=color,
            zorder=3,
        )
        bar_groups.append((bars, means))
        offset_idx += 1

    for bars, means in bar_groups:
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    rotation=30,
                )

    ax.set_ylabel("Pearson R", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("K562 MPRA (Gosai et al. 2024)", fontsize=15)
    ax.legend(fontsize=10, loc="upper right", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "k562_full_dataset_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: k562_full_dataset_bar.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")
    generate_k562_plots()
    generate_k562_bar_plot()
    generate_yeast_plots()
    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
