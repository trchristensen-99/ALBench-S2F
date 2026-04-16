#!/usr/bin/env python3
"""Plot HP search method comparison: performance + consistency.

Reads exhaustive_hp_search results and generates a 2-panel figure:
  Panel A: Cross-validated performance per method (bar + individual strategy dots)
  Panel B: Consistency metrics (val_std, cross_std, lr_ratio)

Usage:
    python scripts/analysis/plot_hp_comparison.py
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
RESULTS_DIR = REPO / "outputs" / "exhaustive_hp_search"
OUT_DIR = REPO / "results" / "hp_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "tpe_20_multi_warm": "TPE Multi\n(20t, warm)",
    "random_20_warm": "Random\n(20t, warm)",
    "gp_20_warm": "GP\n(20t, warm)",
    "ensemble_3x20": "Ensemble\n(3×20t)",
    "tpe_10_cold": "TPE\n(10t, cold)",
    "cma_20_warm": "CMA-ES\n(20t, warm)",
    "cma_30_warm": "CMA-ES\n(30t, warm)",
    "tpe_20_warm_narrow": "TPE Narrow\n(20t, warm)",
}

STRAT_MARKERS = {"random": "o", "genomic": "s", "motif_grammar": "D"}
STRAT_COLORS = {"random": "#888888", "genomic": "#1f77b4", "motif_grammar": "#2ca02c"}


def load_results():
    """Load all exhaustive HP search results."""
    by_method = defaultdict(list)
    for f in RESULTS_DIR.rglob("*.json"):
        d = json.loads(f.read_text())
        m, s = d["method"], d["strategy"]
        if m == "ensemble_3x20":
            ens = d["ensemble"]
            by_method[m].append(
                {
                    "strategy": s,
                    "cross_val": ens["consensus_mean_val"],
                    "cross_std": ens["consensus_std_val"],
                    "val_std": ens["consensus_std_val"],
                    "lr_ratio": 0,
                }
            )
        else:
            c = d["consistency"]
            ce = d.get("cross_evaluation", [])
            best_ce = max(ce, key=lambda x: x["cross_eval_mean"]) if ce else {}
            by_method[m].append(
                {
                    "strategy": s,
                    "cross_val": d["recommended_hp_cross_val"],
                    "cross_std": best_ce.get("cross_eval_std", 0) if best_ce else 0,
                    "val_std": c["val_std"],
                    "lr_ratio": c["lr_ratio"],
                }
            )
    return by_method


def main():
    data = load_results()
    if not data:
        print("No results found")
        return

    # Sort methods by mean cross-val performance
    method_order = sorted(
        data.keys(),
        key=lambda m: np.mean([r["cross_val"] for r in data[m]]),
        reverse=True,
    )

    # ── Figure: 2-panel comparison ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Performance (cross-validated val Pearson R)
    x = np.arange(len(method_order))
    bar_means = []
    bar_stds = []
    for m in method_order:
        vals = [r["cross_val"] for r in data[m]]
        bar_means.append(np.mean(vals))
        bar_stds.append(np.std(vals))

    colors = ["#2ecc71" if i == 0 else "#bdc3c7" for i in range(len(method_order))]
    ax1.bar(
        x,
        bar_means,
        yerr=bar_stds,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )

    # Overlay individual strategy dots
    for m_idx, m in enumerate(method_order):
        for r in data[m]:
            marker = STRAT_MARKERS.get(r["strategy"], "o")
            color = STRAT_COLORS.get(r["strategy"], "#333")
            ax1.scatter(
                m_idx,
                r["cross_val"],
                marker=marker,
                c=color,
                s=40,
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )

    # Legend for strategy markers
    for strat, marker in STRAT_MARKERS.items():
        ax1.scatter(
            [],
            [],
            marker=marker,
            c=STRAT_COLORS[strat],
            label=strat.replace("_", " ").title(),
            s=40,
        )
    ax1.legend(fontsize=8, loc="lower left", frameon=True)

    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_LABELS.get(m, m) for m in method_order], fontsize=8, ha="center")
    ax1.set_ylabel("Cross-Validated Val Pearson R", fontsize=11)
    ax1.set_title("A. HP Search Performance (50K)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Set y-axis to show meaningful range
    ymin = min(bar_means) - 0.015
    ymax = max(bar_means) + 0.015
    ax1.set_ylim(ymin, ymax)

    # Panel B: Consistency metrics
    metrics = {
        "Val Std (↓ better)": ("val_std", "#3498db"),
        "Cross Std (↓ better)": ("cross_std", "#e74c3c"),
    }

    bar_width = 0.35
    for i, (label, (key, color)) in enumerate(metrics.items()):
        vals = []
        for m in method_order:
            entries = data[m]
            v = np.mean([r[key] for r in entries])
            vals.append(v)
        offset = (i - 0.5) * bar_width
        ax2.bar(
            x + offset,
            vals,
            bar_width,
            label=label,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([METHOD_LABELS.get(m, m) for m in method_order], fontsize=8, ha="center")
    ax2.set_ylabel("Standard Deviation", fontsize=11)
    ax2.set_title("B. HP Search Consistency (50K)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right", frameon=True)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT_DIR / "hp_method_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "hp_method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'hp_method_comparison.png'}")

    # ── Single summary panel for talks ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter: x = cross_std (consistency), y = cross_val (performance)
    for m in method_order:
        entries = data[m]
        cv = np.mean([r["cross_val"] for r in entries])
        cs = np.mean([r["cross_std"] for r in entries])
        n_strats = len(entries)

        color = "#2ecc71" if m == method_order[0] else "#3498db"
        size = 120 + n_strats * 30
        ax.scatter(cs, cv, s=size, c=color, zorder=3, edgecolors="black", linewidths=0.8)
        # Label
        label = METHOD_LABELS.get(m, m).replace("\n", " ")
        ax.annotate(label, (cs, cv), fontsize=7.5, xytext=(8, -4), textcoords="offset points")

    ax.set_xlabel("Cross-Eval Std (lower = more consistent)", fontsize=11)
    ax.set_ylabel("Cross-Validated Performance", fontsize=11)
    ax.set_title(
        "HP Search Methods: Performance vs Consistency (50K)", fontsize=12, fontweight="bold"
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Arrow pointing to ideal corner
    ax.annotate(
        "← Better",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=9,
        color="green",
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "hp_performance_vs_consistency.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "hp_performance_vs_consistency.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'hp_performance_vs_consistency.png'}")


if __name__ == "__main__":
    main()
