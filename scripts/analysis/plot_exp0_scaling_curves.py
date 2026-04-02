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

K562_STUDENTS = ["legnet", "dream_cnn", "dream_rnn", "alphagenome_k562_s1", "alphagenome_k562_s2"]
YEAST_STUDENTS = [
    "legnet",
    "dream_cnn",
    "dream_rnn",
    "alphagenome_yeast_s1",
    "alphagenome_yeast_s2",
]
# AG S2 yeast: softmax bug fixed; _hlr variant merged in load_scaling_data

# Extra directories to merge into main student results (same student, different HP)
YEAST_MERGE_DIRS = {
    "alphagenome_yeast_s2": ["alphagenome_yeast_s2_hlr"],
}

COLORS = {
    "legnet": "#D4A017",
    "dream_cnn": "#E8602C",
    "dream_rnn": "#7B2D8E",
    "alphagenome_k562_s1": "#66BB6A",
    "alphagenome_k562_s2": "#1B5E20",
    "alphagenome_yeast_s1": "#66BB6A",
    "alphagenome_yeast_s2": "#1B5E20",
}

LABELS = {
    "legnet": "LegNet",
    "dream_cnn": "DREAM-CNN",
    "dream_rnn": "DREAM-RNN",
    "alphagenome_k562_s1": "AG S1",
    "alphagenome_k562_s2": "AG S2",
    "alphagenome_yeast_s1": "AG S1",
    "alphagenome_yeast_s2": "AG S2",
}

MARKERS = {
    "legnet": "P",
    "dream_cnn": "o",
    "dream_rnn": "s",
    "alphagenome_k562_s1": "D",
    "alphagenome_k562_s2": "^",
    "alphagenome_yeast_s1": "D",
    "alphagenome_yeast_s2": "^",
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
    elif category == "in_dist_real":
        if "in_dist_real" in test_metrics and field in test_metrics["in_dist_real"]:
            return test_metrics["in_dist_real"][field]
    elif category == "ood":
        for key in ("ood", "genomic"):
            if key in test_metrics and field in test_metrics[key]:
                return test_metrics[key][field]
    elif category == "ood_real":
        if "ood_real" in test_metrics and field in test_metrics["ood_real"]:
            return test_metrics["ood_real"][field]
    elif category == "snv_delta":
        if "snv_delta" in test_metrics and field in test_metrics["snv_delta"]:
            return test_metrics["snv_delta"][field]
    elif category == "snv_delta_real":
        if "snv_delta_real" in test_metrics and field in test_metrics["snv_delta_real"]:
            return test_metrics["snv_delta_real"][field]
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

    # Build mapping of merge dirs → target student
    merge_map: dict[str, str] = {}
    if task == "yeast":
        for target, extras in YEAST_MERGE_DIRS.items():
            for extra in extras:
                merge_map[extra] = target

    for rj in task_dir.rglob("result.json"):
        r = json.loads(rj.read_text())
        parts = str(rj.relative_to(task_dir)).split("/")
        student = parts[0]
        # Merge extra dirs into their target student
        student = merge_map.get(student, student)
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

    # Build metric curves (oracle + real label versions)
    metric_keys = [
        "in_dist_pearson",
        "ood_pearson",
        "snv_delta_pearson",
        "in_dist_mse",
        # Real-label counterparts (from _real keys in test_metrics)
        "in_dist_real_pearson",
        "ood_real_pearson",
        "snv_delta_real_pearson",
    ]
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
                "in_dist_real_pearson": _get_metric(tm, "in_dist_real", "pearson_r"),
                "ood_real_pearson": _get_metric(tm, "ood_real", "pearson_r"),
                "snv_delta_real_pearson": _get_metric(tm, "snv_delta_real", "pearson_r"),
            }
            for mk in metric_keys:
                if vals[mk] is not None:
                    data[student][mk].setdefault(n, []).append(vals[mk])

    return data


def _curve_arrays(
    n_to_vals: dict[int, list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert {n_train: [values]} to sorted (ns, means, stds) arrays.

    Filters out data points where the best HP config's mean is effectively zero
    (broken runs that haven't been replaced yet).
    """
    if not n_to_vals:
        return np.array([]), np.array([]), np.array([])
    ns_out, means_out, stds_out = [], [], []
    for n in sorted(n_to_vals.keys()):
        vals = n_to_vals[n]
        # Filter out near-zero values (broken runs)
        good = [v for v in vals if abs(v) > 0.01]
        if not good:
            continue  # skip this N entirely if all values are ~0
        m = np.mean(good)
        s = np.std(good) if len(good) > 1 else 0.0
        ns_out.append(n)
        means_out.append(m)
        stds_out.append(s)
    return np.array(ns_out), np.array(means_out), np.array(stds_out)


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
    annotate: bool = False,
    skip_annotate: set | None = None,
):
    """Plot one metric panel with lines, markers, and error bands.

    Args:
        annotate: If True, add value labels (3 decimal places) above each point.
        skip_annotate: Set of student keys to skip annotation for (e.g., AG S1
            on K562 where values are nearly identical to AG S2).
    """
    if skip_annotate is None:
        skip_annotate = set()

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

        # Add value annotations
        if annotate and student not in skip_annotate:
            for n_val, m_val in zip(ns, means):
                ax.annotate(
                    f"{m_val:.3f}",
                    (n_val, m_val),
                    textcoords="offset points",
                    xytext=(0, 7),
                    fontsize=5.5,
                    ha="center",
                    color=color,
                    fontweight="semibold",
                    zorder=5,
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
            m = n / 1_000_000
            return f"{m:.1f}M" if m != int(m) else f"{int(m)}M"
        if n >= 1_000:
            k = round(n / 1_000)
            return f"{k}k"
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
    skip_annotate: set | None = None,
):
    """Create a 2x2 panel figure for the given task."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
            annotate=True,
            skip_annotate=skip_annotate,
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
    skip_annotate: set | None = None,
):
    """Create a single-panel 'money plot' for K562 in-dist Pearson R."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

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
        annotate=True,
        skip_annotate=skip_annotate,
    )
    _format_n_ticks(ax, ns_union)
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor="0.8")

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        path = OUT / f"{out_stem}{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# --- Real-label data loading (old format) -----------------------------------------


def load_real_label_data(task: str) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Load real-label scaling results from older experiment directories.

    Returns same format as load_scaling_data.
    """
    data: dict[str, dict[str, dict[int, list[float]]]] = {}

    if task == "k562":
        # DREAM-RNN real labels
        base = REPO / "outputs" / "exp0_k562_scaling"
        if base.exists():
            student = "dream_rnn"
            mk_data: dict[str, dict[int, list[float]]] = {
                "in_dist_pearson": {},
                "ood_pearson": {},
                "snv_delta_pearson": {},
                "in_dist_mse": {},
            }
            for rj in base.rglob("result.json"):
                r = json.loads(rj.read_text())
                n = r.get("n_samples", 0)
                if n <= 0:
                    continue
                tm = r.get("test_metrics", {})
                for cat, mk in [
                    ("in_dist", "in_dist_pearson"),
                    ("ood", "ood_pearson"),
                    ("snv_delta", "snv_delta_pearson"),
                    ("in_dist", "in_dist_mse"),
                ]:
                    field = "mse" if "mse" in mk else "pearson_r"
                    val = _get_metric(tm, cat, field)
                    if val is not None:
                        mk_data[mk].setdefault(n, []).append(val)
            data[student] = mk_data

        # AG S1 real labels
        base = REPO / "outputs" / "exp0_k562_scaling_alphagenome_cached_rcaug"
        if base.exists():
            student = "alphagenome_k562_s1"
            mk_data = {
                "in_dist_pearson": {},
                "ood_pearson": {},
                "snv_delta_pearson": {},
                "in_dist_mse": {},
            }
            for rj in base.rglob("result.json"):
                r = json.loads(rj.read_text())
                n = r.get("n_samples", r.get("n_train", 0))
                if n <= 0:
                    continue
                tm = r.get("test_metrics", {})
                for cat, mk in [
                    ("in_dist", "in_dist_pearson"),
                    ("ood", "ood_pearson"),
                    ("snv_delta", "snv_delta_pearson"),
                    ("in_dist", "in_dist_mse"),
                ]:
                    field = "mse" if "mse" in mk else "pearson_r"
                    val = _get_metric(tm, cat, field)
                    if val is not None:
                        mk_data[mk].setdefault(n, []).append(val)
            data[student] = mk_data

    elif task == "yeast":
        # DREAM-RNN real labels
        base = REPO / "outputs" / "exp0_yeast_scaling_v2"
        if base.exists():
            student = "dream_rnn"
            mk_data = {
                "in_dist_pearson": {},
                "ood_pearson": {},
                "snv_delta_pearson": {},
                "in_dist_mse": {},
            }
            for rj in base.rglob("result.json"):
                r = json.loads(rj.read_text())
                n = r.get("n_samples", 0)
                if n <= 0:
                    continue
                tm = r.get("test_metrics", {})
                for cat, mk in [
                    ("in_dist", "in_dist_pearson"),
                    ("ood", "ood_pearson"),
                    ("snv_delta", "snv_delta_pearson"),
                    ("in_dist", "in_dist_mse"),
                ]:
                    field = "mse" if "mse" in mk else "pearson_r"
                    val = _get_metric(tm, cat, field)
                    if val is not None:
                        mk_data[mk].setdefault(n, []).append(val)
            data[student] = mk_data

        # AG S1 real labels
        base = REPO / "outputs" / "exp0_yeast_scaling_ag_v2"
        if base.exists():
            student = "alphagenome_yeast_s1"
            mk_data = {
                "in_dist_pearson": {},
                "ood_pearson": {},
                "snv_delta_pearson": {},
                "in_dist_mse": {},
            }
            for rj in base.rglob("result.json"):
                r = json.loads(rj.read_text())
                n = r.get("n_samples", r.get("n_train", 0))
                if n <= 0:
                    continue
                tm = r.get("test_metrics", {})
                for cat, mk in [
                    ("in_dist", "in_dist_pearson"),
                    ("ood", "ood_pearson"),
                    ("snv_delta", "snv_delta_pearson"),
                    ("in_dist", "in_dist_mse"),
                ]:
                    field = "mse" if "mse" in mk else "pearson_r"
                    val = _get_metric(tm, cat, field)
                    if val is not None:
                        mk_data[mk].setdefault(n, []).append(val)
            data[student] = mk_data

        # AG S2 real labels (old format: summary.json, fraction-based dirs)
        base = REPO / "outputs" / "exp0_yeast_scaling_ag_s2"
        if base.exists():
            student = "alphagenome_yeast_s2"
            mk_data = {
                "in_dist_pearson": {},
                "ood_pearson": {},
                "snv_delta_pearson": {},
                "in_dist_mse": {},
            }
            yeast_total = 6_065_324
            for sj in base.rglob("summary.json"):
                r = json.loads(sj.read_text())
                # Derive n from fraction directory name
                frac_dir = sj.parent.parent.name  # e.g., "fraction_0.05"
                try:
                    frac = float(frac_dir.split("_")[1])
                    n = round(yeast_total * frac)
                except (IndexError, ValueError):
                    continue
                if n <= 0:
                    continue
                tm = r.get("test_metrics", {})
                for cat, mk in [
                    ("in_dist", "in_dist_pearson"),
                    ("ood", "ood_pearson"),
                    ("snv_delta", "snv_delta_pearson"),
                    ("in_dist", "in_dist_mse"),
                ]:
                    field = "mse" if "mse" in mk else "pearson_r"
                    # summary.json uses "random" for in-dist, "genomic" for ood
                    val = _get_metric(tm, cat, field)
                    if val is not None:
                        mk_data[mk].setdefault(n, []).append(val)
            data[student] = mk_data

    return data


# --- Cross-oracle data loading ---------------------------------------------------

# Cross-oracle student names
K562_CROSS_ORACLE = [
    "dream_cnn_oracle_dream_rnn",
    "dream_rnn_oracle_dream_rnn",
    "alphagenome_k562_s1_oracle_dream_rnn",
]
YEAST_CROSS_ORACLE = [
    "dream_cnn_oracle_ag",
    "dream_rnn_oracle_ag",
    "alphagenome_yeast_s1_oracle_ag",
]

CROSS_ORACLE_COLORS = {
    "dream_cnn_oracle_dream_rnn": "#E8602C",
    "dream_rnn_oracle_dream_rnn": "#7B2D8E",
    "alphagenome_k562_s1_oracle_dream_rnn": "#66BB6A",
    "dream_cnn_oracle_ag": "#E8602C",
    "dream_rnn_oracle_ag": "#7B2D8E",
    "alphagenome_yeast_s1_oracle_ag": "#66BB6A",
}

CROSS_ORACLE_LABELS = {
    "dream_cnn_oracle_dream_rnn": "DREAM-CNN",
    "dream_rnn_oracle_dream_rnn": "DREAM-RNN",
    "alphagenome_k562_s1_oracle_dream_rnn": "AG S1",
    "dream_cnn_oracle_ag": "DREAM-CNN",
    "dream_rnn_oracle_ag": "DREAM-RNN",
    "alphagenome_yeast_s1_oracle_ag": "AG S1",
}

CROSS_ORACLE_MARKERS = {
    "dream_cnn_oracle_dream_rnn": "o",
    "dream_rnn_oracle_dream_rnn": "s",
    "alphagenome_k562_s1_oracle_dream_rnn": "D",
    "dream_cnn_oracle_ag": "o",
    "dream_rnn_oracle_ag": "s",
    "alphagenome_yeast_s1_oracle_ag": "D",
}


def plot_oracle_vs_real(
    task: str,
    task_label: str,
    students: list[str],
    oracle_data: dict,
    real_data: dict,
    out_stem: str,
):
    """Plot oracle vs real label comparison for in-dist Pearson R."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    all_ns: set[int] = set()

    # Plot oracle labels (solid lines)
    for student in students:
        n_to_vals = oracle_data.get(student, {}).get("in_dist_pearson", {})
        ns, means, stds = _curve_arrays(n_to_vals)
        if len(ns) == 0:
            continue
        all_ns.update(int(n) for n in ns)
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
            label=f"{label} (oracle)",
            zorder=3,
        )
        ax.fill_between(ns, means - stds, means + stds, color=color, alpha=0.12)

    # Plot real labels (dashed lines)
    for student in students:
        n_to_vals = real_data.get(student, {}).get("in_dist_pearson", {})
        ns, means, stds = _curve_arrays(n_to_vals)
        if len(ns) == 0:
            continue
        all_ns.update(int(n) for n in ns)
        color = COLORS[student]
        label = LABELS[student]
        marker = MARKERS[student]
        ax.plot(
            ns,
            means,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label=f"{label} (real)",
            zorder=3,
        )
        ax.fill_between(ns, means - stds, means + stds, color=color, alpha=0.08)

    ax.set_xscale("log")
    ax.set_xlabel("N training examples")
    ax.set_ylabel("In-distribution Pearson R")
    ax.set_title(
        f"{task_label} — Oracle vs Real Labels",
        fontsize=13,
        fontweight="semibold",
    )
    _format_n_ticks(ax, sorted(all_ns))
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.8", ncol=2)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        path = OUT / f"{out_stem}{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def plot_cross_oracle(
    task: str,
    task_label: str,
    default_students: list[str],
    cross_students: list[str],
    data: dict,
    default_oracle_name: str,
    cross_oracle_name: str,
    out_stem: str,
):
    """Plot default oracle vs cross oracle for in-dist Pearson R."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    all_ns: set[int] = set()

    # Default oracle (solid lines)
    for student in default_students:
        n_to_vals = data.get(student, {}).get("in_dist_pearson", {})
        ns, means, stds = _curve_arrays(n_to_vals)
        if len(ns) == 0:
            continue
        all_ns.update(int(n) for n in ns)
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
            label=f"{label} ({default_oracle_name})",
            zorder=3,
        )
        ax.fill_between(ns, means - stds, means + stds, color=color, alpha=0.12)

    # Cross oracle (dashed lines)
    for student in cross_students:
        n_to_vals = data.get(student, {}).get("in_dist_pearson", {})
        ns, means, stds = _curve_arrays(n_to_vals)
        if len(ns) == 0:
            continue
        all_ns.update(int(n) for n in ns)
        color = CROSS_ORACLE_COLORS[student]
        label = CROSS_ORACLE_LABELS[student]
        marker = CROSS_ORACLE_MARKERS[student]
        ax.plot(
            ns,
            means,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label=f"{label} ({cross_oracle_name})",
            zorder=3,
        )
        ax.fill_between(ns, means - stds, means + stds, color=color, alpha=0.08)

    ax.set_xscale("log")
    ax.set_xlabel("N training examples")
    ax.set_ylabel("In-distribution Pearson R")
    ax.set_title(
        f"{task_label} — {default_oracle_name} vs {cross_oracle_name} Oracle Labels",
        fontsize=13,
        fontweight="semibold",
    )
    _format_n_ticks(ax, sorted(all_ns))
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.8", ncol=2)

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
    all_k562_students = K562_STUDENTS + K562_CROSS_ORACLE
    k562_data = load_scaling_data("k562", all_k562_students)
    if k562_data:
        for s in K562_STUDENTS:
            ns = sorted(k562_data.get(s, {}).get("in_dist_pearson", {}).keys())
            if ns:
                vals = [f"n={n}: {np.mean(k562_data[s]['in_dist_pearson'][n]):.4f}" for n in ns]
                print(f"  {LABELS.get(s, s)}: {', '.join(vals)}")

        print("Plotting K562 2x2 panel...")
        # Skip annotating AG S1 on K562 (values nearly identical to AG S2 except MSE)
        k562_skip = {"alphagenome_k562_s1"}
        plot_2x2_panel(
            "k562",
            "K562",
            K562_STUDENTS,
            k562_data,
            "k562_scaling_2x2",
            skip_annotate=k562_skip,
        )
        print("Plotting K562 overview...")
        plot_overview(K562_STUDENTS, k562_data, "k562_scaling_overview", skip_annotate=k562_skip)

        # Cross-oracle comparison
        print("Plotting K562 cross-oracle comparison...")
        plot_cross_oracle(
            "k562",
            "K562",
            K562_STUDENTS[:3],  # no AG S2 for cross-oracle
            K562_CROSS_ORACLE,
            k562_data,
            "AG oracle",
            "DREAM-RNN oracle",
            "k562_cross_oracle",
        )
    else:
        print("  No K562 data found, skipping.")

    # Oracle vs real labels for K562
    print("Loading K562 real-label data...")
    k562_real = load_real_label_data("k562")
    if k562_real and k562_data:
        print("Plotting K562 oracle vs real comparison...")
        plot_oracle_vs_real(
            "k562",
            "K562",
            ["dream_rnn", "alphagenome_k562_s1"],
            k562_data,
            k562_real,
            "k562_oracle_vs_real",
        )

    # --- Yeast ---
    print("Loading Yeast data...")
    all_yeast_students = YEAST_STUDENTS + YEAST_CROSS_ORACLE
    yeast_data = load_scaling_data("yeast", all_yeast_students)
    if yeast_data:
        for s in YEAST_STUDENTS:
            ns = sorted(yeast_data.get(s, {}).get("in_dist_pearson", {}).keys())
            if ns:
                vals = [f"n={n}: {np.mean(yeast_data[s]['in_dist_pearson'][n]):.4f}" for n in ns]
                print(f"  {LABELS.get(s, s)}: {', '.join(vals)}")

        print("Plotting Yeast 2x2 panel...")
        plot_2x2_panel("yeast", "Yeast", YEAST_STUDENTS, yeast_data, "yeast_scaling_2x2")

        # Cross-oracle comparison
        print("Plotting Yeast cross-oracle comparison...")
        plot_cross_oracle(
            "yeast",
            "Yeast",
            YEAST_STUDENTS,
            YEAST_CROSS_ORACLE,
            yeast_data,
            "DREAM-RNN oracle",
            "AG oracle",
            "yeast_cross_oracle",
        )
    else:
        print("  No Yeast data found, skipping.")

    # Oracle vs real labels for Yeast
    print("Loading Yeast real-label data...")
    yeast_real = load_real_label_data("yeast")
    if yeast_real and yeast_data:
        print("Plotting Yeast oracle vs real comparison...")
        plot_oracle_vs_real(
            "yeast",
            "Yeast",
            ["dream_rnn", "alphagenome_yeast_s1", "alphagenome_yeast_s2"],
            yeast_data,
            yeast_real,
            "yeast_oracle_vs_real",
        )

    print("Done.")


if __name__ == "__main__":
    main()
