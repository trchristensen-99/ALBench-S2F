#!/usr/bin/env python3
"""Generate bar plot comparing data generation strategies for Exp 1.1.

Reads results from outputs/exp1_1_strategy_comparison/ and creates
a grouped bar plot showing in-distribution, SNV, and OOD Pearson R
for each strategy.

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

STRATEGY_COLORS = {
    "random": "#9E9E9E",
    "dinuc_shuffle": "#78909C",
    "genomic": "#3A86C8",
    "gc_matched": "#26A69A",
    "prm_5pct": "#E8602C",
    "prm_10pct": "#D84315",
    "prm_20pct": "#BF360C",
    "activity_stratified_oracle": "#7B2D8E",
    "motif_clustering": "#66BB6A",
    "motif_clustering_mutant": "#1B5E20",
    "evoaug_structural": "#DAA520",
    "snv": "#F06292",
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


def plot_strategy_bars(
    results: dict[str, list[dict]],
    task: str = "k562",
    student: str = "dream_cnn",
    n_train: int = 31974,
):
    test_keys = [
        ("in_distribution", "In-Distribution"),
        ("snv_abs", "SNV"),
        ("ood", "OOD (Synthetic)"),
    ]

    # Order strategies by in-dist performance
    strat_means = {}
    for strat, metrics_list in results.items():
        vals = [_get_metric(tm, "in_distribution") for tm in metrics_list]
        vals = [v for v in vals if v is not None]
        if vals:
            strat_means[strat] = np.mean(vals)
    ordered = sorted(strat_means.keys(), key=lambda s: -strat_means[s])

    if not ordered:
        print("  No strategy results found")
        return

    n_strategies = len(ordered)
    n_tests = len(test_keys)
    x = np.arange(n_strategies)
    width = 0.8 / n_tests
    offsets = np.linspace(-(n_tests - 1) / 2 * width, (n_tests - 1) / 2 * width, n_tests)

    test_colors = ["#3A86C8", "#E8602C", "#66BB6A"]

    fig, ax = plt.subplots(figsize=(max(10, n_strategies * 1.2), 5.5))

    for ti, (test_key, test_label) in enumerate(test_keys):
        means = []
        stds = []
        for strat in ordered:
            vals = [_get_metric(tm, test_key) for tm in results[strat]]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)

        bars = ax.bar(
            x + offsets[ti],
            means,
            width,
            yerr=stds if any(s > 0 for s in stds) else None,
            label=test_label,
            color=test_colors[ti],
            alpha=0.85,
            zorder=3,
            capsize=2,
        )

        for bar, val in zip(bars, means):
            if val > 0 and ti == 0:  # Only label in-dist
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    rotation=30,
                )

    labels = [STRATEGY_DISPLAY.get(s, s) for s in ordered]
    n_runs = [len(results[s]) for s in ordered]
    labels = [f"{lab}\n(n={n})" for lab, n in zip(labels, n_runs)]

    ax.set_ylabel("Pearson R", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Data Generation Strategy Comparison\n"
        f"{task.upper()} {student.replace('_', '-').upper()}, N={n_train:,}",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fname = f"strategy_comparison_{task}_{student}"
    fig.savefig(OUT_DIR / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.png / .pdf")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading strategy comparison results...")

    results = load_strategy_results("k562", "dream_cnn")
    for strat, metrics in sorted(results.items()):
        vals = [_get_metric(tm, "in_distribution") for tm in metrics]
        vals = [v for v in vals if v is not None]
        mean_r = np.mean(vals) if vals else float("nan")
        print(f"  {strat:30s}: {len(metrics)} runs, in_dist={mean_r:.4f}")

    print("\nGenerating strategy comparison plot...")
    plot_strategy_bars(results)
    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
