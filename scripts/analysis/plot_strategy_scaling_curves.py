#!/usr/bin/env python3
"""Strategy scaling curves: performance vs training size for each reservoir strategy.

Generates the key Exp 1.1 figure: multiple lines showing how different
reservoir sampling strategies scale with training data size.

Reads from outputs/exp1_1/{task}/{student_oracle}/{strategy}/n*/hp*/seed*/result.json

Usage:
    python scripts/analysis/plot_strategy_scaling_curves.py
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
EXP1_DIR = REPO / "outputs" / "exp1_1"
OUT_DIR = REPO / "results" / "strategy_scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy display names and colors
STRATEGY_ORDER = [
    "random",
    "genomic",
    "dinuc_shuffle",
    "gc_matched",
    "prm_1pct",
    "prm_5pct",
    "prm_10pct",
    "prm_20pct",
    "prm_50pct",
    "prm_uniform_1_10",
    "recombination_uniform",
    "recombination_2pt",
    "evoaug_structural",
    "evoaug_heavy",
    "motif_planted",
    "motif_grammar",
    "motif_grammar_tight",
    "ise_maximize",
    "ise_diverse_targets",
    "ise_target_high",
    "snv",
]

STRATEGY_COLORS = {
    "random": "#888888",
    "genomic": "#1f77b4",
    "dinuc_shuffle": "#ff7f0e",
    "gc_matched": "#2ca02c",
    "prm_1pct": "#d62728",
    "prm_5pct": "#e377c2",
    "prm_10pct": "#9467bd",
    "prm_20pct": "#8c564b",
    "prm_50pct": "#bcbd22",
    "prm_uniform_1_10": "#17becf",
    "recombination_uniform": "#aec7e8",
    "recombination_2pt": "#ffbb78",
    "evoaug_structural": "#98df8a",
    "evoaug_heavy": "#ff9896",
    "motif_planted": "#c5b0d5",
    "motif_grammar": "#c49c94",
    "motif_grammar_tight": "#f7b6d2",
    "ise_maximize": "#dbdb8d",
    "ise_diverse_targets": "#9edae5",
    "ise_target_high": "#393b79",
    "snv": "#637939",
}

STRATEGY_LABELS = {
    "random": "Random",
    "genomic": "Genomic",
    "dinuc_shuffle": "Dinuc Shuffle",
    "gc_matched": "GC-matched",
    "prm_1pct": "PRM 1%",
    "prm_5pct": "PRM 5%",
    "prm_10pct": "PRM 10%",
    "prm_20pct": "PRM 20%",
    "prm_50pct": "PRM 50%",
    "prm_uniform_1_10": "PRM Unif 1-10%",
    "recombination_uniform": "Recomb. Uniform",
    "recombination_2pt": "Recomb. 2-point",
    "evoaug_structural": "EvoAug Structural",
    "evoaug_heavy": "EvoAug Heavy",
    "motif_planted": "Motif Planted",
    "motif_grammar": "Motif Grammar",
    "motif_grammar_tight": "Motif Grammar (tight)",
    "ise_maximize": "ISE Maximize",
    "ise_diverse_targets": "ISE Diverse",
    "ise_target_high": "ISE Target High",
    "snv": "SNV",
}

# Category grouping for simplified plots
STRATEGY_CATEGORIES = {
    "Baseline": ["random", "genomic", "dinuc_shuffle", "gc_matched"],
    "Mutagenesis (PRM)": ["prm_1pct", "prm_5pct", "prm_10pct", "prm_20pct", "prm_50pct"],
    "Recombination": ["recombination_uniform", "recombination_2pt"],
    "Augmentation": ["evoaug_structural", "evoaug_heavy"],
    "Motif": ["motif_planted", "motif_grammar", "motif_grammar_tight"],
    "ISE (Oracle-guided)": ["ise_maximize", "ise_diverse_targets", "ise_target_high"],
}


def load_strategy_data(task: str, config_dir: str) -> dict:
    """Load results for all strategies under a student_oracle config.

    Returns: {strategy: {n_train: [(val_r, test_metrics)]}}
    """
    base = EXP1_DIR / task / config_dir
    if not base.exists():
        return {}

    # Group by (strategy, n_train, hp_config)
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rj in base.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        parts = str(rj.relative_to(base)).split("/")
        strategy = parts[0]
        n = d.get("n_train", 0)
        hp = json.dumps(d.get("hp_config", {}), sort_keys=True)
        val_r = d.get("val_pearson_r", 0)
        test = d.get("test_metrics", {})
        raw[strategy][n][hp].append((val_r, test))

    # For each (strategy, n_train), pick best HP by mean val_r
    result = {}
    for strategy in raw:
        result[strategy] = {}
        for n, hp_map in raw[strategy].items():
            best_hp = max(hp_map, key=lambda k: np.mean([v[0] for v in hp_map[k]]))
            result[strategy][n] = hp_map[best_hp]
    return result


def get_metric(test_metrics, key="in_dist", field="pearson_r"):
    """Extract metric with key fallbacks."""
    for k in [key, "in_distribution", "random"]:
        if k in test_metrics and field in test_metrics[k]:
            return test_metrics[k][field]
    return None


def plot_scaling_curves(
    data: dict,
    metric_key: str,
    metric_field: str,
    title: str,
    ylabel: str,
    out_path: Path,
    strategies: list[str] | None = None,
    highlight_random: bool = True,
    figsize: tuple = (12, 7),
):
    """Plot scaling curves for all strategies."""
    if strategies is None:
        strategies = [s for s in STRATEGY_ORDER if s in data]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for strat in strategies:
        if strat not in data:
            continue
        sizes = sorted(data[strat].keys())
        means, stds = [], []
        valid_sizes = []
        for n in sizes:
            vals = [get_metric(t, metric_key, metric_field) for _, t in data[strat][n]]
            vals = [v for v in vals if v is not None]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_sizes.append(n)

        if not valid_sizes:
            continue

        means = np.array(means)
        stds = np.array(stds)
        color = STRATEGY_COLORS.get(strat, "#666666")
        label = STRATEGY_LABELS.get(strat, strat)
        lw = 2.5 if strat == "random" and highlight_random else 1.2
        ls = "--" if strat == "random" and highlight_random else "-"
        alpha = 1.0 if strat == "random" and highlight_random else 0.8
        zorder = 10 if strat == "random" else 1

        ax.plot(
            valid_sizes,
            means,
            color=color,
            label=label,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
            zorder=zorder,
            marker="o",
            markersize=3,
        )
        ax.fill_between(valid_sizes, means - stds, means + stds, alpha=0.1, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("N training examples", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=150)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_category_scaling(
    data: dict,
    metric_key: str,
    metric_field: str,
    title: str,
    ylabel: str,
    out_path: Path,
):
    """Plot scaling curves grouped by category (one subplot per category)."""
    cats = {k: v for k, v in STRATEGY_CATEGORIES.items() if any(s in data for s in v)}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flat

    for idx, (cat_name, strats) in enumerate(cats.items()):
        ax = axes[idx]
        # Always plot random as reference
        if "random" in data:
            sizes = sorted(data["random"].keys())
            means = []
            valid_sizes = []
            for n in sizes:
                vals = [get_metric(t, metric_key, metric_field) for _, t in data["random"][n]]
                vals = [v for v in vals if v is not None]
                if vals:
                    means.append(np.mean(vals))
                    valid_sizes.append(n)
            if valid_sizes:
                ax.plot(
                    valid_sizes,
                    means,
                    color="#888888",
                    linestyle="--",
                    linewidth=2,
                    label="Random",
                    zorder=10,
                )

        for strat in strats:
            if strat not in data or strat == "random":
                continue
            sizes = sorted(data[strat].keys())
            means, stds = [], []
            valid_sizes = []
            for n in sizes:
                vals = [get_metric(t, metric_key, metric_field) for _, t in data[strat][n]]
                vals = [v for v in vals if v is not None]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    valid_sizes.append(n)
            if not valid_sizes:
                continue
            means = np.array(means)
            stds = np.array(stds)
            color = STRATEGY_COLORS.get(strat, "#666666")
            label = STRATEGY_LABELS.get(strat, strat)
            ax.plot(
                valid_sizes,
                means,
                color=color,
                label=label,
                linewidth=1.5,
                marker="o",
                markersize=3,
            )
            ax.fill_between(valid_sizes, means - stds, means + stds, alpha=0.1, color=color)

        ax.set_xscale("log")
        ax.set_title(cat_name, fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    # Hide extra axes
    for idx in range(len(cats), len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel("N training examples", fontsize=12)
    fig.supylabel(ylabel, fontsize=12)
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=150)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    configs = {
        "k562": [
            ("alphagenome_k562_s1_ag", "AG S1 (AG oracle)"),
            ("dream_rnn_dream_rnn", "DREAM-RNN (DREAM oracle)"),
            ("alphagenome_k562_s1_dream_rnn", "AG S1 (DREAM oracle)"),
        ],
        "yeast": [
            ("alphagenome_yeast_s1_ag", "AG S1 (AG oracle)"),
            ("dream_rnn_dream_rnn", "DREAM-RNN (DREAM oracle)"),
        ],
    }

    for task, task_configs in configs.items():
        print(f"\n=== {task.upper()} ===")
        for config_dir, config_label in task_configs:
            data = load_strategy_data(task, config_dir)
            if not data:
                print(f"  {config_label}: no data")
                continue

            n_strats = len(data)
            print(f"  {config_label}: {n_strats} strategies loaded")

            safe_name = config_dir.replace("/", "_")

            # All strategies, in_dist
            plot_scaling_curves(
                data,
                "in_dist",
                "pearson_r",
                f"{task.upper()} — {config_label}\nIn-Distribution Pearson R vs Training Size",
                "In-dist Pearson R",
                OUT_DIR / f"{task}_{safe_name}_in_dist.png",
            )

            # All strategies, OOD
            plot_scaling_curves(
                data,
                "ood",
                "pearson_r",
                f"{task.upper()} — {config_label}\nOOD Pearson R vs Training Size",
                "OOD Pearson R",
                OUT_DIR / f"{task}_{safe_name}_ood.png",
            )

            # Category panels, in_dist
            plot_category_scaling(
                data,
                "in_dist",
                "pearson_r",
                f"{task.upper()} — {config_label}: Strategy Categories (In-Dist)",
                "In-dist Pearson R",
                OUT_DIR / f"{task}_{safe_name}_categories_in_dist.png",
            )

            # Category panels, OOD
            plot_category_scaling(
                data,
                "ood",
                "pearson_r",
                f"{task.upper()} — {config_label}: Strategy Categories (OOD)",
                "OOD Pearson R",
                OUT_DIR / f"{task}_{safe_name}_categories_ood.png",
            )

    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
