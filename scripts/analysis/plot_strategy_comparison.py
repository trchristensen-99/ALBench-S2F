#!/usr/bin/env python3
"""Generate multi-panel plots comparing data generation strategies for Exp 1.1.

Reads results from outputs/exp1_1_strategy_comparison/ and creates:
1. Pearson R bar plot (in-dist, SNV, SNV delta, OOD)
2. MSE bar plot (in-dist, SNV, SNV delta, OOD)
3. Summary table printed to stdout

Usage:
    python scripts/analysis/plot_strategy_comparison.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "strategy_plots"

STRATEGY_DISPLAY = {
    "random": "Random",
    "dinuc_shuffle": "Dinuc\nShuffle",
    "genomic": "Genomic",
    "gc_matched": "GC\nMatched",
    "prm_5pct": "PRM\n5%",
    "prm_10pct": "PRM\n10%",
    "prm_20pct": "PRM\n20%",
    "activity_stratified_oracle": "Activity\nStratified",
    "motif_clustering": "Motif\nClustering",
    "motif_clustering_mutant": "Motif Clust\n+ Mutant",
    "evoaug_structural": "EvoAug\nStructural",
    "snv": "SNV\n(0.5%)",
}

# Ordered test conditions with display names
TEST_CONDITIONS = [
    ("in_distribution", "Reference"),
    ("snv_abs", "SNV"),
    ("snv_delta", "SNV delta"),
    ("ood", "OOD (Synthetic)"),
]

TEST_COLORS = {
    "in_distribution": "#3A86C8",
    "snv_abs": "#E8602C",
    "snv_delta": "#B07CC6",
    "ood": "#66BB6A",
}


def load_strategy_results(task: str = "k562", student: str = "dream_cnn") -> dict[str, list[dict]]:
    base = REPO / "outputs" / "exp1_1_strategy_comparison" / task / student
    results: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(base.rglob("result.json")):
        try:
            d = json.loads(p.read_text())
            strat = d.get("reservoir", "?")
            tm = d.get("test_metrics", {})
            results[strat].append(tm)
        except Exception:
            continue
    return dict(results)


def _get_metric(tm: dict, test_key: str, metric: str = "pearson_r") -> float | None:
    """Extract metric from test_metrics with key normalization."""
    for k in [test_key, test_key.replace("in_distribution", "in_dist")]:
        if k in tm and isinstance(tm[k], dict) and metric in tm[k]:
            return tm[k][metric]
    return None


def _get_ordered_strategies(results: dict[str, list[dict]]) -> list[str]:
    """Order strategies by in-dist Pearson R (descending)."""
    strat_means = {}
    for strat, metrics_list in results.items():
        vals = [_get_metric(tm, "in_distribution") for tm in metrics_list]
        vals = [v for v in vals if v is not None]
        if vals:
            strat_means[strat] = np.mean(vals)
    return sorted(strat_means.keys(), key=lambda s: -strat_means[s])


def _make_grouped_bar(
    ax: plt.Axes,
    results: dict[str, list[dict]],
    ordered: list[str],
    metric: str,
    test_conditions: list[tuple[str, str]],
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    label_top_bar: bool = True,
    higher_is_better: bool = True,
):
    """Draw a grouped bar chart on the given axes."""
    n_strategies = len(ordered)
    n_tests = len(test_conditions)
    x = np.arange(n_strategies)
    width = 0.8 / n_tests
    offsets = np.linspace(-(n_tests - 1) / 2 * width, (n_tests - 1) / 2 * width, n_tests)

    for ti, (test_key, test_label) in enumerate(test_conditions):
        means, stds = [], []
        for strat in ordered:
            vals = [_get_metric(tm, test_key, metric) for tm in results[strat]]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)

        color = TEST_COLORS.get(test_key, f"C{ti}")
        bars = ax.bar(
            x + offsets[ti],
            means,
            width,
            yerr=stds if any(s > 0 for s in stds) else None,
            label=test_label,
            color=color,
            alpha=0.85,
            zorder=3,
            capsize=2,
        )

        if label_top_bar and ti == 0:
            for bar, val in zip(bars, means):
                if val > 0:
                    fmt = f"{val:.3f}" if metric == "pearson_r" else f"{val:.2f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        fmt,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                        rotation=30,
                    )

    labels = [STRATEGY_DISPLAY.get(s, s) for s in ordered]
    n_runs = [len(results[s]) for s in ordered]
    labels = [f"{lab}\n(n={nr})" for lab, nr in zip(labels, n_runs)]

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right" if higher_is_better else "upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0)


def plot_strategy_multipanel(
    results: dict[str, list[dict]],
    task: str = "k562",
    student: str = "dream_cnn",
    n_train: int = 31974,
):
    """Generate 2-panel figure: Pearson R (top) and MSE (bottom)."""
    ordered = _get_ordered_strategies(results)
    if not ordered:
        print("  No strategy results found")
        return

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(ordered) * 1.3), 10))

    suptitle = (
        f"Data Generation Strategy Comparison — "
        f"{task.upper()} {student.replace('_', '-').upper()}, N={n_train:,}"
    )
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Pearson R
    _make_grouped_bar(
        axes[0],
        results,
        ordered,
        metric="pearson_r",
        test_conditions=TEST_CONDITIONS,
        ylabel="Pearson R",
        title="Pearson Correlation",
        ylim=(0, 1.0),
        label_top_bar=True,
        higher_is_better=True,
    )

    # Panel 2: MSE
    _make_grouped_bar(
        axes[1],
        results,
        ordered,
        metric="mse",
        test_conditions=TEST_CONDITIONS,
        ylabel="MSE",
        title="Mean Squared Error",
        ylim=None,
        label_top_bar=True,
        higher_is_better=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"strategy_comparison_{task}_{student}"
    fig.savefig(OUT_DIR / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png / .pdf")


def print_summary_table(results: dict[str, list[dict]]):
    """Print a full summary table with all metrics."""
    ordered = _get_ordered_strategies(results)
    if not ordered:
        return

    header_metrics = [
        ("in_distribution", "pearson_r", "ID R"),
        ("in_distribution", "mse", "ID MSE"),
        ("snv_abs", "pearson_r", "SNV R"),
        ("snv_delta", "pearson_r", "SNVd R"),
        ("snv_delta", "mse", "SNVd MSE"),
        ("ood", "pearson_r", "OOD R"),
        ("ood", "mse", "OOD MSE"),
    ]

    header = f"{'Strategy':<25} {'Runs':>4}"
    for _, _, short in header_metrics:
        header += f" {short:>9}"
    print(header)
    print("-" * len(header))

    for strat in ordered:
        row = f"{strat:<25} {len(results[strat]):>4}"
        for test_key, metric, _ in header_metrics:
            vals = [_get_metric(tm, test_key, metric) for tm in results[strat]]
            vals = [v for v in vals if v is not None]
            if vals:
                mean = np.mean(vals)
                fmt = f"{mean:.4f}" if metric == "pearson_r" else f"{mean:.3f}"
                row += f" {fmt:>9}"
            else:
                row += f" {'—':>9}"
        print(row)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading strategy comparison results...")

    results = load_strategy_results("k562", "dream_cnn")

    print("\n=== Full Summary Table ===\n")
    print_summary_table(results)

    print("\nGenerating multi-panel strategy comparison plot...")
    plot_strategy_multipanel(results)
    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
