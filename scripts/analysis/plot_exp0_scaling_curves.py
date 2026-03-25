#!/usr/bin/env python3
"""Generate comprehensive Exp 0 scaling curve plots (multi-panel).

Reads results from outputs/exp0_oracle_scaling_v4/{task}/{student}/random/n*/hp*/seed*/result.json
and produces:
  1. K562 2x2 panel (in_dist Pearson, OOD Pearson, SNV delta Pearson, in_dist MSE)
  2. Yeast 2x2 panel (same layout, with random/genomic key fallbacks)
  3. Combined K562 overview (single in-dist Pearson panel)

Saves to results/exp0_scaling_plots/
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
BASE = REPO / "outputs" / "exp0_oracle_scaling_v4"
OUT = REPO / "results" / "exp0_scaling_plots"

# --- Model definitions -----------------------------------------------------------

K562_STUDENTS = ["dream_cnn", "dream_rnn", "alphagenome_k562_s1"]
YEAST_STUDENTS = ["dream_cnn", "dream_rnn", "alphagenome_yeast_s2"]

COLORS = {
    "dream_cnn": "#E8602C",
    "dream_rnn": "#7B2D8E",
    "alphagenome_k562_s1": "#66BB6A",
    "alphagenome_yeast_s2": "#66BB6A",
}

LABELS = {
    "dream_cnn": "DREAM-CNN",
    "dream_rnn": "DREAM-RNN",
    "alphagenome_k562_s1": "AG S1",
    "alphagenome_yeast_s2": "AG S2",
}

MARKERS = {
    "dream_cnn": "o",
    "dream_rnn": "s",
    "alphagenome_k562_s1": "D",
    "alphagenome_yeast_s2": "D",
}


# --- Metric extraction helpers ---------------------------------------------------


def _get_metric(test_metrics: dict, category: str, field: str) -> float | None:
    """Extract a metric value with key fallbacks.

    category mapping:
      "in_dist"   -> try "in_dist", "in_distribution", "random" (yeast)
      "ood"       -> try "ood", "genomic"
      "snv_delta" -> try "snv_delta"
    field: "pearson_r" or "mse"
    """
    if category == "in_dist":
        for key in ("in_dist", "in_distribution", "random"):
            if key in test_metrics and field in test_metrics[key]:
                return test_metrics[key][field]
    elif category == "ood":
        for key in ("ood", "genomic"):
            if key in test_metrics and field in test_metrics[key]:
                return test_metrics[key][field]
    elif category == "snv_delta":
        if "snv_delta" in test_metrics and field in test_metrics["snv_delta"]:
            return test_metrics["snv_delta"][field]
    return None


# --- Data loading -----------------------------------------------------------------


def load_scaling_data(
    task: str, students: list[str]
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Load all result.json files for a task.

    Returns:
        {student: {metric_key: {n_train: [values_across_seeds]}}}
        where metric_key is one of:
          "in_dist_pearson", "ood_pearson", "snv_delta_pearson", "in_dist_mse"
    """
    task_dir = BASE / task
    if not task_dir.exists():
        print(f"  WARNING: {task_dir} does not exist")
        return {}

    # Collect raw data grouped by (student, n_train, hp_config) -> list of metric dicts
    raw: dict[tuple, list[dict]] = defaultdict(list)

    for rj in task_dir.rglob("result.json"):
        r = json.loads(rj.read_text())
        parts = str(rj.relative_to(task_dir)).split("/")
        student = parts[0]
        if student not in students:
            continue
        n = r["n_train"]
        hp = json.dumps(r.get("hp_config", {}), sort_keys=True)
        raw[(student, n, hp)].append(r)

    # For each (student, n_train), pick the hp config with best mean val_pearson_r
    best: dict[tuple, list[dict]] = {}
    grouped: dict[tuple, dict[str, list[dict]]] = defaultdict(dict)
    for (student, n, hp), results in raw.items():
        grouped[(student, n)][hp] = results

    for (student, n), hp_map in grouped.items():
        best_hp = max(hp_map.keys(), key=lambda k: np.mean([r["val_pearson_r"] for r in hp_map[k]]))
        best[(student, n)] = hp_map[best_hp]

    # Build metric curves
    metric_keys = ["in_dist_pearson", "ood_pearson", "snv_delta_pearson", "in_dist_mse"]
    data: dict[str, dict[str, dict[int, list[float]]]] = {
        s: {m: {} for m in metric_keys} for s in students
    }

    for (student, n), results in sorted(best.items()):
        for r in results:
            tm = r.get("test_metrics", {})
            vals = {
                "in_dist_pearson": _get_metric(tm, "in_dist", "pearson_r"),
                "ood_pearson": _get_metric(tm, "ood", "pearson_r"),
                "snv_delta_pearson": _get_metric(tm, "snv_delta", "pearson_r"),
                "in_dist_mse": _get_metric(tm, "in_dist", "mse"),
            }
            for mk in metric_keys:
                if vals[mk] is not None:
                    data[student][mk].setdefault(n, []).append(vals[mk])

    return data


def _curve_arrays(
    n_to_vals: dict[int, list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert {n_train: [values]} to sorted (ns, means, stds) arrays."""
    if not n_to_vals:
        return np.array([]), np.array([]), np.array([])
    ns = sorted(n_to_vals.keys())
    means = np.array([np.mean(n_to_vals[n]) for n in ns])
    stds = np.array([np.std(n_to_vals[n]) if len(n_to_vals[n]) > 1 else 0.0 for n in ns])
    return np.array(ns), means, stds


# --- Plotting helpers -------------------------------------------------------------


def _setup_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )


def _plot_scaling_panel(
    ax: plt.Axes,
    data: dict[str, dict[int, list[float]]],
    students: list[str],
    ylabel: str,
    title: str,
    invert: bool = False,
):
    """Plot one metric panel with lines, markers, and error bands."""
    for student in students:
        n_to_vals = data.get(student, {})
        ns, means, stds = _curve_arrays(n_to_vals)
        if len(ns) == 0:
            continue
        color = COLORS[student]
        label = LABELS[student]
        marker = MARKERS[student]

        ax.plot(
            ns,
            means,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            ns,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("N training examples")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="semibold")
    if invert:
        # For MSE, lower is better — keep default (not inverted) but note it
        pass


def _format_n_ticks(ax, ns_union: list[int]):
    """Set x-ticks to actual N values with human-readable labels."""
    if not ns_union:
        return

    def _fmt(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(n)

    ax.set_xticks(ns_union)
    ax.set_xticklabels([_fmt(n) for n in ns_union], rotation=45, ha="right", fontsize=9)
    ax.minorticks_off()


# --- Main plot functions ----------------------------------------------------------


def plot_2x2_panel(
    task: str,
    task_label: str,
    students: list[str],
    data: dict,
    out_stem: str,
):
    """Create a 2x2 panel figure for the given task."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panels = [
        ("in_dist_pearson", "In-distribution Pearson R", False),
        ("ood_pearson", "OOD Pearson R", False),
        ("snv_delta_pearson", "SNV-delta Pearson R", False),
        ("in_dist_mse", "In-distribution MSE", True),
    ]

    # Collect union of all N values for consistent ticks
    all_ns: set[int] = set()
    for student in students:
        for mk in data.get(student, {}):
            all_ns.update(data[student][mk].keys())
    ns_union = sorted(all_ns)

    for idx, (metric_key, ylabel, invert) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        panel_data = {s: data.get(s, {}).get(metric_key, {}) for s in students}
        _plot_scaling_panel(
            ax,
            panel_data,
            students,
            ylabel,
            f"{task_label} — {ylabel}",
            invert=invert,
        )
        _format_n_ticks(ax, ns_union)

    # Single legend on first panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(students),
            fontsize=11,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(
        f"{task_label} Scaling Curves (Exp 0, Oracle Labels)",
        fontsize=14,
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        path = OUT / f"{out_stem}{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def plot_overview(
    students: list[str],
    data: dict,
    out_stem: str,
):
    """Create a single-panel 'money plot' for K562 in-dist Pearson R."""
    fig, ax = plt.subplots(figsize=(7, 5))

    all_ns: set[int] = set()
    for s in students:
        all_ns.update(data.get(s, {}).get("in_dist_pearson", {}).keys())
    ns_union = sorted(all_ns)

    panel_data = {s: data.get(s, {}).get("in_dist_pearson", {}) for s in students}
    _plot_scaling_panel(
        ax,
        panel_data,
        students,
        "In-distribution Pearson R",
        "K562 Scaling — In-distribution Pearson R",
    )
    _format_n_ticks(ax, ns_union)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor="0.8")

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        path = OUT / f"{out_stem}{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# --- Main -------------------------------------------------------------------------


def main():
    _setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- K562 ---
    print("Loading K562 data...")
    k562_data = load_scaling_data("k562", K562_STUDENTS)
    if k562_data:
        # Print summary
        for s in K562_STUDENTS:
            ns = sorted(k562_data.get(s, {}).get("in_dist_pearson", {}).keys())
            if ns:
                vals = [f"n={n}: {np.mean(k562_data[s]['in_dist_pearson'][n]):.4f}" for n in ns]
                print(f"  {LABELS.get(s, s)}: {', '.join(vals)}")

        print("Plotting K562 2x2 panel...")
        plot_2x2_panel("k562", "K562", K562_STUDENTS, k562_data, "k562_scaling_2x2")
        print("Plotting K562 overview...")
        plot_overview(K562_STUDENTS, k562_data, "k562_scaling_overview")
    else:
        print("  No K562 data found, skipping.")

    # --- Yeast ---
    print("Loading Yeast data...")
    yeast_data = load_scaling_data("yeast", YEAST_STUDENTS)
    if yeast_data:
        for s in YEAST_STUDENTS:
            ns = sorted(yeast_data.get(s, {}).get("in_dist_pearson", {}).keys())
            if ns:
                vals = [f"n={n}: {np.mean(yeast_data[s]['in_dist_pearson'][n]):.4f}" for n in ns]
                print(f"  {LABELS.get(s, s)}: {', '.join(vals)}")

        print("Plotting Yeast 2x2 panel...")
        plot_2x2_panel("yeast", "Yeast", YEAST_STUDENTS, yeast_data, "yeast_scaling_2x2")
    else:
        print("  No Yeast data found, skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
