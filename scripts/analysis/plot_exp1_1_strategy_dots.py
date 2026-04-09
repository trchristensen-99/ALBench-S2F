#!/usr/bin/env python3
"""Strategy scaling with HP dots: performance vs training size for each strategy.

Each (strategy, training_size) shows ALL HP configs as individual dots,
with the best HP config highlighted. This is the key Exp 1.1 figure
showing data informativeness independent of model configuration.

Usage:
    python scripts/analysis/plot_exp1_1_strategy_dots.py
    python scripts/analysis/plot_exp1_1_strategy_dots.py --config legnet_ag_s2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP1_DIR = REPO / "outputs" / "exp1_1"
OUT_DIR = REPO / "results" / "strategy_scaling_dots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy display
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
    "motif_clustering_mutant": "#f7b6d2",
    "activity_stratified_oracle": "#dbdb8d",
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
    "motif_clustering_mutant": "Motif Cluster+Mut",
    "activity_stratified_oracle": "Activity Strat.",
    "snv": "SNV",
}

# Default configs to plot
CONFIGS = {
    "legnet_ag_s2": ("k562", "legnet_ag_s2", "LegNet student, AG S2 oracle"),
    "ag_s1_ag": ("k562", "alphagenome_k562_s1_ag", "AG S1 student, AG oracle"),
    "drnn_drnn": ("k562", "dream_rnn_dream_rnn", "DREAM-RNN student, DREAM oracle"),
}


def load_all_results(task: str, config_dir: str):
    """Load ALL result.json files, grouped by (strategy, n_train, hp_idx).

    Returns: {strategy: {n_train: {hp_idx: [(val_r, test_metrics, hp_config)]}}}
    """
    base = EXP1_DIR / task / config_dir
    if not base.exists():
        return {}

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rj in base.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        parts = str(rj.relative_to(base)).split("/")
        if len(parts) < 5:
            continue
        strategy = parts[0]
        n = d.get("n_train", 0)
        hp_config = d.get("hp_config", {})
        hp_idx = parts[2] if len(parts) > 2 else "hp0"  # e.g., "hp0", "hp1"
        val_r = d.get("val_pearson_r", 0)
        test = d.get("test_metrics", {})
        data[strategy][n][hp_idx].append((val_r, test, hp_config))

    return data


def get_metric(test_metrics, key="in_dist", field="pearson_r"):
    for k in [key, "in_distribution", "random"]:
        if k in test_metrics and field in test_metrics[k]:
            return test_metrics[k][field]
    return None


def plot_strategy_dots(
    data: dict,
    metric_key: str,
    metric_field: str,
    title: str,
    ylabel: str,
    out_path: Path,
    strategies: list[str] | None = None,
):
    """Plot with HP config dots at each training size."""
    if strategies is None:
        strategies = [s for s in STRATEGY_COLORS if s in data]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    for strat in strategies:
        if strat not in data:
            continue
        color = STRATEGY_COLORS.get(strat, "#666666")
        label = STRATEGY_LABELS.get(strat, strat)

        sizes = sorted(data[strat].keys())
        best_means = []
        best_sizes = []

        for n in sizes:
            hp_map = data[strat][n]

            # Plot each HP config as a dot (mean across seeds)
            best_hp_mean = -999
            for hp_idx, runs in hp_map.items():
                vals = [get_metric(t, metric_key, metric_field) for _, t, _ in runs]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                hp_mean = np.mean(vals)

                # Jitter x position slightly for visibility
                jitter = np.random.default_rng(abs(hash(f"{strat}_{n}_{hp_idx}"))).uniform(
                    -0.05, 0.05
                )
                x = n * (10**jitter)

                # Individual HP dots (small, semi-transparent)
                ax.scatter(
                    [x] * len(vals),
                    vals,
                    color=color,
                    alpha=0.15,
                    s=8,
                    zorder=1,
                    edgecolors="none",
                )

                # HP mean dot (medium)
                ax.scatter(
                    x,
                    hp_mean,
                    color=color,
                    alpha=0.4,
                    s=25,
                    zorder=2,
                    edgecolors="none",
                )

                if hp_mean > best_hp_mean:
                    best_hp_mean = hp_mean

            if best_hp_mean > -999:
                best_means.append(best_hp_mean)
                best_sizes.append(n)

        # Connect best HPs with a bold line
        if best_sizes:
            ax.plot(
                best_sizes,
                best_means,
                color=color,
                linewidth=2.5 if strat == "random" else 1.5,
                linestyle="--" if strat == "random" else "-",
                label=label,
                zorder=5 if strat == "random" else 3,
                marker="D" if strat == "random" else "o",
                markersize=6,
            )

    ax.set_xscale("log")
    ax.set_xlabel("N training examples", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(str(out_path).replace(".png", ".pdf"), dpi=150)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Config key from CONFIGS dict")
    args = parser.parse_args()

    configs = CONFIGS if args.config is None else {args.config: CONFIGS[args.config]}

    for config_key, (task, config_dir, config_label) in configs.items():
        print(f"\n=== {config_label} ===")
        data = load_all_results(task, config_dir)
        if not data:
            print(f"  No data for {config_dir}")
            continue

        n_strats = len(data)
        n_total = sum(
            sum(len(runs) for runs in hp_map.values())
            for strat_data in data.values()
            for hp_map in strat_data.values()
        )
        print(f"  {n_strats} strategies, {n_total} total results")

        safe = config_key.replace("/", "_")

        # In-dist with dots
        plot_strategy_dots(
            data,
            "in_dist",
            "pearson_r",
            f"{config_label}\nIn-Distribution Pearson R (dots = HP configs)",
            "In-dist Pearson R",
            OUT_DIR / f"{safe}_in_dist_dots.png",
        )

        # OOD with dots
        plot_strategy_dots(
            data,
            "ood",
            "pearson_r",
            f"{config_label}\nOOD Pearson R (dots = HP configs)",
            "OOD Pearson R",
            OUT_DIR / f"{safe}_ood_dots.png",
        )

    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
