#!/usr/bin/env python
"""Plot scaling curves from Experiment 1.1 results.

Generates three types of publication-quality figures:
  1. Per-reservoir scaling curves (all HP configs + best-HP line)
  2. Multi-reservoir comparison (best-HP only, all reservoirs on one plot)
  3. Heatmap of best Pearson R values (reservoir x n_train)

Usage::

    # Full analysis with all defaults
    uv run python experiments/plot_exp1_1_scaling.py \
        --results-dir outputs/exp1_1/k562/alphagenome_k562_s1_ag

    # Filter to specific reservoirs and test sets
    uv run python experiments/plot_exp1_1_scaling.py \
        --results-dir outputs/exp1_1/k562/dream_rnn \
        --reservoirs random genomic \
        --test-sets in_dist ood

    # Plot MSE instead of Pearson R
    uv run python experiments/plot_exp1_1_scaling.py \
        --results-dir outputs/exp1_1/k562/dream_rnn \
        --metric mse
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# Expanded palette: 21 unique color+marker combinations for all reservoirs.
# Uses tab20 + extra colors to ensure no duplicates.
RESERVOIR_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#e7969c",
]

RESERVOIR_MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "D",
    "P",
    "X",
    "h",
    "<",
    ">",
    "p",
    "*",
    "H",
    "d",
    "8",
    "+",
    "x",
    "1",
    "2",
    "3",
    "4",
]

# Fixed mapping: each reservoir always gets the same style
RESERVOIR_STYLE_MAP: dict[str, dict[str, str]] = {}

TEST_SET_LABELS = {
    "in_dist": "In-Distribution",
    "ood": "Out-of-Distribution",
    "snv_abs": "SNV Absolute",
    "snv_delta": "SNV Delta",
}


def _get_reservoir_style(reservoir: str, idx: int) -> dict[str, str]:
    """Return a unique color/marker for a reservoir (no duplicates up to 21)."""
    if reservoir not in RESERVOIR_STYLE_MAP:
        i = len(RESERVOIR_STYLE_MAP)
        RESERVOIR_STYLE_MAP[reservoir] = {
            "color": RESERVOIR_COLORS[i % len(RESERVOIR_COLORS)],
            "marker": RESERVOIR_MARKERS[i % len(RESERVOIR_MARKERS)],
            "label": reservoir,
        }
    return RESERVOIR_STYLE_MAP[reservoir]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(
    results_dir: Path,
    reservoirs: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Recursively find result.json files and group by reservoir.

    Parses both the directory structure
    ``{reservoir}/n{N}/hp{idx}/seed{S}/result.json`` and the ``reservoir``
    field inside the JSON (the JSON field takes precedence).

    Args:
        results_dir: Root directory to search.
        reservoirs: If given, only include these reservoirs.

    Returns:
        Dict mapping reservoir name to list of result dicts.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    n_loaded = 0

    for result_path in sorted(results_dir.rglob("result.json")):
        try:
            data = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", result_path, e)
            continue

        # Determine reservoir name: prefer JSON field, fall back to dir name
        reservoir = data.get("reservoir")
        if reservoir is None:
            # Try to parse from path: .../reservoir_name/n.../hp.../seed.../result.json
            parts = result_path.relative_to(results_dir).parts
            if len(parts) >= 4:
                reservoir = parts[0]
            else:
                reservoir = "unknown"

        if reservoirs and reservoir not in reservoirs:
            continue

        grouped.setdefault(reservoir, []).append(data)
        n_loaded += 1

    logger.info(
        "Loaded %d results across %d reservoirs from %s",
        n_loaded,
        len(grouped),
        results_dir,
    )
    return grouped


def _extract_metric(result: dict[str, Any], test_set: str, metric: str) -> float | None:
    """Extract a metric value from a result dict.

    Navigates ``result["test_metrics"][test_set][metric]``.
    Returns None if the path is missing or value is non-finite.
    """
    try:
        val = float(result["test_metrics"][test_set][metric])
        if np.isfinite(val):
            return val
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _select_best_hp(
    results: list[dict[str, Any]],
    test_set: str,
    metric: str,
) -> dict[str, Any]:
    """For each n_train, select the HP config with best mean val Pearson.

    Returns a dict with structure::

        {
            n_train: {
                "best_hp_key": str,
                "metric_values": list[float],   # metric from best-HP replicates
                "all_hp_values": list[tuple[str, list[float]]],  # all HP results
            }
        }
    """
    # Group by (n_train, hp_config serialized)
    by_size_hp: dict[tuple[int, str], list[dict]] = {}
    for r in results:
        n = r["n_train"]
        hp_key = json.dumps(r.get("hp_config", {}), sort_keys=True)
        by_size_hp.setdefault((n, hp_key), []).append(r)

    sizes = sorted({n for (n, _) in by_size_hp})
    output: dict[int, dict[str, Any]] = {}

    for n in sizes:
        best_hp_key: str | None = None
        best_val_mean = -np.inf

        # Collect all HP results for scatter points
        all_hp_values: list[tuple[str, list[float]]] = []

        for (sz, hp_key), runs in by_size_hp.items():
            if sz != n:
                continue

            # Mean val Pearson R for HP selection
            val_rs = [r.get("val_pearson_r", -np.inf) for r in runs]
            mean_val = float(np.mean(val_rs))

            # Extract target metric
            vals = [v for r in runs if (v := _extract_metric(r, test_set, metric)) is not None]
            all_hp_values.append((hp_key, vals))

            if mean_val > best_val_mean:
                best_val_mean = mean_val
                best_hp_key = hp_key

        if best_hp_key is None:
            continue

        # Get metric values for best HP
        best_runs = by_size_hp[(n, best_hp_key)]
        best_vals = [
            v for r in best_runs if (v := _extract_metric(r, test_set, metric)) is not None
        ]

        if best_vals:
            output[n] = {
                "best_hp_key": best_hp_key,
                "metric_values": best_vals,
                "all_hp_values": all_hp_values,
            }

    return output


# ---------------------------------------------------------------------------
# Plot 1: Per-reservoir scaling curves
# ---------------------------------------------------------------------------


def plot_per_reservoir(
    grouped: dict[str, list[dict[str, Any]]],
    test_sets: list[str],
    metric: str,
    output_dir: Path,
) -> list[Path]:
    """Generate per-reservoir scaling curve figures.

    Each reservoir gets a figure with subplots for each test set. Semi-transparent
    scatter points show all HP configs; a solid line highlights the best HP.

    Returns list of saved file paths.
    """
    saved: list[Path] = []
    metric_label = "Pearson r" if metric == "pearson_r" else "MSE"

    for res_idx, (reservoir, results) in enumerate(sorted(grouped.items())):
        n_panels = len(test_sets)
        ncols = min(n_panels, 2)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        style = _get_reservoir_style(reservoir, res_idx)

        for panel_idx, ts in enumerate(test_sets):
            ax = axes_flat[panel_idx]
            analysis = _select_best_hp(results, ts, metric)

            if not analysis:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#999",
                )
                ax.set_title(TEST_SET_LABELS.get(ts, ts), fontsize=12, fontweight="bold")
                continue

            sizes_sorted = sorted(analysis.keys())

            # Scatter all HP configs (semi-transparent)
            for n in sizes_sorted:
                for _hp_key, vals in analysis[n]["all_hp_values"]:
                    for v in vals:
                        ax.scatter(
                            n,
                            v,
                            color=style["color"],
                            alpha=0.2,
                            s=20,
                            edgecolors="none",
                            zorder=1,
                        )

            # Best HP line
            best_sizes = np.array(sizes_sorted)
            best_means = np.array(
                [float(np.mean(analysis[n]["metric_values"])) for n in sizes_sorted]
            )
            best_stds = np.array(
                [float(np.std(analysis[n]["metric_values"])) for n in sizes_sorted]
            )

            ax.errorbar(
                best_sizes,
                best_means,
                yerr=best_stds,
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                markersize=7,
                capsize=3,
                capthick=1.5,
                label="Best HP",
                zorder=3,
            )

            ax.set_xscale("log")
            ax.set_xlabel("Training Set Size", fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(TEST_SET_LABELS.get(ts, ts), fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)

        # Hide unused axes
        for j in range(n_panels, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"Reservoir: {reservoir}", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()

        save_path = output_dir / f"scaling_per_reservoir_{reservoir}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(save_path)
        logger.info("Saved per-reservoir plot: %s", save_path)

    return saved


# ---------------------------------------------------------------------------
# Plot 2: Multi-reservoir comparison
# ---------------------------------------------------------------------------


def plot_multi_reservoir_comparison(
    grouped: dict[str, list[dict[str, Any]]],
    test_sets: list[str],
    metric: str,
    output_dir: Path,
) -> Path | None:
    """Generate a single figure comparing all reservoirs.

    4 subplots (one per test set), each with best-HP lines for all reservoirs.
    """
    metric_label = "Pearson r" if metric == "pearson_r" else "MSE"

    n_panels = len(test_sets)
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for panel_idx, ts in enumerate(test_sets):
        ax = axes_flat[panel_idx]

        for res_idx, (reservoir, results) in enumerate(sorted(grouped.items())):
            style = _get_reservoir_style(reservoir, res_idx)
            analysis = _select_best_hp(results, ts, metric)

            if not analysis:
                continue

            sizes_sorted = sorted(analysis.keys())
            means = np.array([float(np.mean(analysis[n]["metric_values"])) for n in sizes_sorted])
            stds = np.array([float(np.std(analysis[n]["metric_values"])) for n in sizes_sorted])

            ax.plot(
                sizes_sorted,
                means,
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=2,
                markersize=7,
                zorder=3,
            )
            ax.fill_between(
                sizes_sorted,
                means - stds,
                means + stds,
                color=style["color"],
                alpha=0.15,
                zorder=2,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training Set Size", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(TEST_SET_LABELS.get(ts, ts), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    # Hide unused axes
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 6),
            fontsize=11,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = output_dir / "scaling_comparison.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved multi-reservoir comparison: %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Plot 3: Heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    grouped: dict[str, list[dict[str, Any]]],
    test_sets: list[str],
    metric: str,
    output_dir: Path,
) -> Path | None:
    """Generate heatmap(s) of best Pearson R: reservoir x n_train.

    One heatmap subplot per test set, annotated with values.
    """
    metric_label = "Pearson r" if metric == "pearson_r" else "MSE"
    reservoirs_sorted = sorted(grouped.keys())

    # Collect all n_train values across reservoirs
    all_sizes: set[int] = set()
    analysis_cache: dict[tuple[str, str], dict] = {}
    for reservoir, results in grouped.items():
        for ts in test_sets:
            a = _select_best_hp(results, ts, metric)
            analysis_cache[(reservoir, ts)] = a
            all_sizes.update(a.keys())

    if not all_sizes:
        logger.warning("No data for heatmap")
        return None

    sizes_sorted = sorted(all_sizes)

    n_panels = len(test_sets)
    n_res = len(reservoirs_sorted)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5 * n_panels + 1, max(4, 0.45 * n_res + 2)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for panel_idx, ts in enumerate(test_sets):
        ax = axes_flat[panel_idx]

        # Build matrix (reservoirs x sizes)
        matrix = np.full((len(reservoirs_sorted), len(sizes_sorted)), np.nan)
        for i, reservoir in enumerate(reservoirs_sorted):
            a = analysis_cache.get((reservoir, ts), {})
            for j, n in enumerate(sizes_sorted):
                if n in a and a[n]["metric_values"]:
                    matrix[i, j] = float(np.mean(a[n]["metric_values"]))

        # Determine color range across all valid values
        valid = matrix[~np.isnan(matrix)]
        if len(valid) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(TEST_SET_LABELS.get(ts, ts), fontsize=12, fontweight="bold")
            continue

        vmin, vmax = float(valid.min()), float(valid.max())
        # Add small padding so colors are not clipped
        margin = max((vmax - vmin) * 0.05, 1e-4)

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="YlOrRd" if metric == "mse" else "YlGnBu",
            vmin=vmin - margin,
            vmax=vmax + margin,
        )

        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                # Adaptive font size based on number of cells
                cell_fontsize = 7 if n_res > 15 else 8 if n_res > 10 else 9
                if np.isnan(val):
                    ax.text(
                        j, i, "--", ha="center", va="center", fontsize=cell_fontsize, color="#999"
                    )
                else:
                    mid = (vmin + vmax) / 2
                    text_color = "white" if val > mid else "black"
                    # Shorter format for crowded heatmaps
                    fmt = f"{val:.2f}" if n_res > 15 else f"{val:.3f}"
                    ax.text(
                        j,
                        i,
                        fmt,
                        ha="center",
                        va="center",
                        fontsize=cell_fontsize,
                        color=text_color,
                    )

        ax.set_xticks(range(len(sizes_sorted)))
        ax.set_xticklabels([str(s) for s in sizes_sorted], fontsize=9, rotation=45, ha="right")
        ax.set_yticks(range(len(reservoirs_sorted)))
        ytick_fontsize = 7 if n_res > 15 else 8 if n_res > 10 else 10
        ax.set_yticklabels(reservoirs_sorted, fontsize=ytick_fontsize)
        ax.set_xlabel("Training Set Size", fontsize=11)
        ax.set_title(TEST_SET_LABELS.get(ts, ts), fontsize=12, fontweight="bold")

        fig.colorbar(im, ax=ax, label=metric_label, shrink=0.8)

    fig.suptitle(
        f"Best {metric_label} by Reservoir and Training Size", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = output_dir / "scaling_heatmap.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_TEST_SETS = ["in_dist", "ood", "snv_abs", "snv_delta"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Experiment 1.1 scaling curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root results directory (e.g. outputs/exp1_1/k562/dream_rnn)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output figures. Default: {results-dir}/figures",
    )
    parser.add_argument(
        "--reservoirs",
        nargs="*",
        default=None,
        help="Filter to specific reservoirs (default: all found)",
    )
    parser.add_argument(
        "--metric",
        choices=["pearson_r", "mse"],
        default="pearson_r",
        help="Which metric to plot (default: pearson_r)",
    )
    parser.add_argument(
        "--test-sets",
        nargs="+",
        default=["all"],
        choices=ALL_TEST_SETS + ["all"],
        help="Test sets to include (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        logger.error("Results directory does not exist: %s", results_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_sets = ALL_TEST_SETS if "all" in args.test_sets else args.test_sets

    # Load data
    grouped = load_results(results_dir, reservoirs=args.reservoirs)
    if not grouped:
        logger.error("No results found in %s", results_dir)
        sys.exit(1)

    logger.info(
        "Reservoirs: %s | Test sets: %s | Metric: %s",
        list(grouped.keys()),
        test_sets,
        args.metric,
    )

    # Generate all plots
    plot_per_reservoir(grouped, test_sets, args.metric, output_dir)
    plot_multi_reservoir_comparison(grouped, test_sets, args.metric, output_dir)
    plot_heatmap(grouped, test_sets, args.metric, output_dir)

    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
