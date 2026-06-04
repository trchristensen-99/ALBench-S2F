#!/usr/bin/env python3
"""Exp0 scaling-law plot: Pearson R + log-log MSE for LegNet & AlphaGenome trained on real and oracle labels.

Uses the consistent v1 dirs (more replicates) and a common N range across all 4 curves.
Styled to match scripts/build_poster_plots_v2.py for visual consistency on the poster.

Curves (4):
  - LegNet (real labels)        legnet_ground_truth   (genomic reservoir, 12+ seeds)
  - LegNet (oracle labels)      legnet_oracle_ag_s2   (genomic reservoir, 3 seeds)
  - AG S1 (real labels)         alphagenome_k562_s1_ground_truth  (genomic reservoir, 3 seeds)
  - AG S1 (oracle labels)       alphagenome_k562_s1   (random reservoir, 12+ seeds)

Common N range: 3197 ... 159871 (6 values matched across all 4 curves).

Run:
  python scripts/analysis/plot_exp0_pearson_vs_mse.py
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
BASE = REPO / "outputs" / "exp0_oracle_scaling_v4" / "k562"
OUT = REPO / "outputs" / "exp0_scaling_pearson_vs_mse"

# Common N values where all 4 curves have data (genomic for first 3, random for AG oracle).
# 300k bin pools N=296382 (genomic max) and N=319742 (random max) at x=300000.
COMMON_NS = [3197, 6395, 15987, 31974, 63949, 159871, 300000]
N_300K_BIN = {296382, 319742}  # alias both to x=300000

CURVES = [
    # (key, dir_name, label, color, marker, linestyle, reservoir_filter)
    {
        "key": "legnet_real",
        "dir": "legnet_ground_truth",
        "label": "LegNet\n(real labels)",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "--",
        "reservoir": "genomic",
    },
    {
        "key": "legnet_orac",
        "dir": "legnet_oracle_ag_s2",
        "label": "LegNet\n(oracle labels)",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "reservoir": "genomic",
    },
    {
        "key": "ag_real",
        "dir": "alphagenome_k562_s1_ground_truth",
        "label": "AG\n(real labels)",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "--",
        "reservoir": "genomic",
    },
    {
        "key": "ag_orac",
        "dir": "alphagenome_k562_s1",
        "label": "AG\n(oracle labels)",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "-",
        "reservoir": "random",
    },
]

# Panel order matches build_poster_plots_v2.py: Genomic → SNV Δ → OOD
PANELS = [
    ("in_dist", "Genomic Reference (held-out chromosomes)"),
    ("snv_delta", "SNV Effect (Δ log2FC)"),
    ("ood", "High-Activity Designed"),
]


def load_curve(student_dir: str, reservoir: str) -> dict[int, list[dict]]:
    root = BASE / student_dir
    by_n: dict[int, list[dict]] = defaultdict(list)
    for f in root.glob("*/n*/hp*/seed*/result.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("reservoir") != reservoir:
            continue
        N = d.get("n_train")
        # Alias both ~300k values to a single x=300000 position
        if N in N_300K_BIN:
            N = 300_000
        if N in COMMON_NS:
            by_n[N].append(d)
    return dict(by_n)


def aggregate(results, panel, metric):
    vals = []
    for d in results:
        if metric == "one_minus_r2":
            pr = d.get("test_metrics", {}).get(panel, {}).get("pearson_r")
            if pr is None:
                continue
            v = 1.0 - pr * pr
        else:
            v = d.get("test_metrics", {}).get(panel, {}).get(metric)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        vals.append(v)
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    median = float(np.median(arr))
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return median, sem, len(arr)


def plot_panel(ax, curves_data, panel, panel_label, metric, show_legend=False):
    for c in CURVES:
        data_by_n = curves_data.get(c["key"], {})
        Ns = sorted(data_by_n.keys())
        xs, ys, errs = [], [], []
        for N in Ns:
            agg = aggregate(data_by_n[N], panel, metric)
            if agg is None:
                continue
            xs.append(N)
            ys.append(agg[0])
            errs.append(agg[1])
        if not xs:
            continue
        xs = np.array(xs)
        ys = np.array(ys)
        errs = np.array(errs)
        # Light shaded band (mean ± 1 SD)
        ax.fill_between(xs, ys - errs, ys + errs, color=c["color"], alpha=0.15, edgecolor="none")
        ax.plot(
            xs,
            ys,
            color=c["color"],
            marker=c["marker"],
            label=c["label"],
            markersize=8,
            linewidth=2,
            alpha=0.92,
            linestyle=c["linestyle"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (log scale)", fontsize=15)
    if metric == "mse":
        ax.set_yscale("log")
        ax.set_ylabel("MSE (log scale)", fontsize=15)
    elif metric == "one_minus_r2":
        ax.set_yscale("log")
        ax.set_ylabel("1 − R² (log scale)", fontsize=15)
    else:
        ax.set_ylabel("Pearson R", fontsize=15)
    ax.set_title(panel_label, fontsize=16, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    # Round-number tick positions
    ax.set_xticks([3_000, 10_000, 30_000, 100_000, 300_000])
    ax.set_xticklabels(
        ["3,000", "10,000", "30,000", "100,000", "300,000"], rotation=30, ha="right", fontsize=12
    )
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.tick_params(axis="y", which="minor", labelsize=12)
    # Legend handled at figure level outside of panels, not per-axes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves_data = {}
    for c in CURVES:
        curves_data[c["key"]] = load_curve(c["dir"], c["reservoir"])
        n_total = sum(len(v) for v in curves_data[c["key"]].values())
        ns_avail = sorted(curves_data[c["key"]].keys())
        print(f"  {c['label']:<32}  {len(ns_avail)} N values, {n_total} total runs   ({ns_avail})")

    for metric in ["pearson_r", "mse", "one_minus_r2"]:
        fig = plt.figure(figsize=(21, 6))
        gs = fig.add_gridspec(1, 3, wspace=0.12)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        fig.subplots_adjust(left=0.055, right=0.90, bottom=0.18, top=0.90)
        if metric == "pearson_r":
            for a in axes[1:]:
                a.sharey(axes[0])
        for i, (panel, label) in enumerate(PANELS):
            plot_panel(axes[i], curves_data, panel, label, metric, show_legend=False)
        if metric == "pearson_r":
            axes[0].set_ylim(0, 1.0)
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_,
            loc="center left",
            bbox_to_anchor=(0.905, 0.5),
            fontsize=10.5,
            framealpha=0.92,
            handlelength=3.2,
            labelspacing=1.0,
            borderpad=0.6,
            handletextpad=0.5,
        )
        suffix_map = {"pearson_r": "", "mse": "_mse", "one_minus_r2": "_1minusR2"}
        suffix = suffix_map[metric]
        fig.savefig(OUT / f"main{suffix}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"main{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved main{suffix}")


if __name__ == "__main__":
    main()
