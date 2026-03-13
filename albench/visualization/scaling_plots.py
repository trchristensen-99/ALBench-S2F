"""Scaling curve comparison plots for Experiment 1.

Loads result JSONs from multiple reservoir strategies and plots
scaling curves (training size vs. metric) with error bands.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Consistent colors and markers per reservoir strategy
RESERVOIR_STYLES = {
    "random": {"color": "#1f77b4", "marker": "o", "label": "Random"},
    "genomic": {"color": "#ff7f0e", "marker": "s", "label": "Genomic Pool"},
    "prm_1pct": {"color": "#2ca02c", "marker": "^", "label": "PRM 1%"},
    "prm_5pct": {"color": "#d62728", "marker": "v", "label": "PRM 5%"},
    "prm_10pct": {"color": "#9467bd", "marker": "D", "label": "PRM 10%"},
    "prm_uniform_1_10": {"color": "#8c564b", "marker": "P", "label": "PRM U(1-10%)"},
}


def load_results(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all result.json files grouped by reservoir strategy.

    Args:
        results_dir: Base output directory (e.g. outputs/exp1_1/k562/dream_rnn).

    Returns:
        Dict mapping reservoir name to list of result dicts.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result_path in sorted(results_dir.rglob("result.json")):
        data = json.loads(result_path.read_text())
        reservoir = data.get("reservoir", "unknown")
        grouped.setdefault(reservoir, []).append(data)
    return grouped


def _aggregate_by_size(
    results: list[dict[str, Any]],
    metric_path: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate results by training size, picking best HP per size.

    For each n_train, finds the HP config with highest mean val_pearson_r,
    then reports mean and std of the target metric across replicates with
    that best HP.

    Returns:
        (sizes, means, stds) arrays sorted by size.
    """
    # Group by (n_train, hp_config_key)
    by_size_hp: dict[tuple[int, str], list[dict]] = {}
    for r in results:
        n = r["n_train"]
        hp_key = json.dumps(r["hp_config"], sort_keys=True)
        by_size_hp.setdefault((n, hp_key), []).append(r)

    # For each size, find best HP by mean val_pearson_r
    sizes_set: set[int] = {n for (n, _) in by_size_hp}
    size_metrics: dict[int, list[float]] = {}

    for n in sorted(sizes_set):
        best_hp_key = None
        best_val = -1.0
        for (sz, hp_key), runs in by_size_hp.items():
            if sz != n:
                continue
            mean_val = float(np.mean([r["val_pearson_r"] for r in runs]))
            if mean_val > best_val:
                best_val = mean_val
                best_hp_key = hp_key
        if best_hp_key is None:
            continue

        # Extract target metric for best HP runs
        vals = []
        for r in by_size_hp[(n, best_hp_key)]:
            v = r
            for key in metric_path:
                if isinstance(v, dict) and key in v:
                    v = v[key]
                else:
                    v = None
                    break
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))

        if vals:
            size_metrics[n] = vals

    sizes = np.array(sorted(size_metrics.keys()))
    means = np.array([float(np.mean(size_metrics[n])) for n in sizes])
    stds = np.array([float(np.std(size_metrics[n])) for n in sizes])
    return sizes, means, stds


def plot_reservoir_scaling_comparison(
    results_dir: Path,
    metric_path: list[str] | None = None,
    metric_label: str = "Pearson r",
    title: str | None = None,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Plot scaling curves comparing reservoir strategies.

    Args:
        results_dir: Base output directory containing reservoir subdirs.
        metric_path: Path to metric in test_metrics dict.
            Default: ["in_dist", "pearson_r"].
        metric_label: Y-axis label.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if metric_path is None:
        metric_path = ["test_metrics", "in_dist", "pearson_r"]

    grouped = load_results(results_dir)
    if not grouped:
        logger.warning(f"No results found in {results_dir}")
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for reservoir, results in sorted(grouped.items()):
        style = RESERVOIR_STYLES.get(
            reservoir,
            {"color": "#333333", "marker": "x", "label": reservoir},
        )
        sizes, means, stds = _aggregate_by_size(results, metric_path)
        if len(sizes) == 0:
            continue

        ax.plot(
            sizes,
            means,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2,
            markersize=7,
        )
        ax.fill_between(
            sizes,
            means - stds,
            means + stds,
            color=style["color"],
            alpha=0.15,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Set Size", fontsize=13)
    ax.set_ylabel(metric_label, fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_multi_panel_scaling(
    results_dir: Path,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (18, 10),
) -> Any:
    """Plot 4-panel scaling comparison (in_dist, OOD, SNV abs, SNV delta).

    Args:
        results_dir: Base output directory.
        save_path: If provided, save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    panels = [
        {
            "metric_path": ["test_metrics", "in_dist", "pearson_r"],
            "label": "In-Distribution Pearson r",
            "title": "In-Distribution",
        },
        {
            "metric_path": ["test_metrics", "ood", "pearson_r"],
            "label": "OOD Pearson r",
            "title": "Out-of-Distribution",
        },
        {
            "metric_path": ["test_metrics", "snv_abs", "pearson_r"],
            "label": "SNV Absolute Pearson r",
            "title": "SNV Absolute",
        },
        {
            "metric_path": ["test_metrics", "snv_delta", "pearson_r"],
            "label": "SNV Delta Pearson r",
            "title": "SNV Delta",
        },
    ]

    grouped = load_results(results_dir)
    if not grouped:
        logger.warning(f"No results found in {results_dir}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()

    for ax, panel in zip(axes_flat, panels):
        for reservoir, results in sorted(grouped.items()):
            style = RESERVOIR_STYLES.get(
                reservoir,
                {"color": "#333333", "marker": "x", "label": reservoir},
            )
            sizes, means, stds = _aggregate_by_size(results, panel["metric_path"])
            if len(sizes) == 0:
                continue

            ax.plot(
                sizes,
                means,
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(
                sizes,
                means - stds,
                means + stds,
                color=style["color"],
                alpha=0.15,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training Set Size", fontsize=11)
        ax.set_ylabel(panel["label"], fontsize=11)
        ax.set_title(panel["title"], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    # Single legend for all panels
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved multi-panel plot to {save_path}")

    return fig
