#!/usr/bin/env python3
"""Oracle ensemble bar plot: performance on all 3 test sets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "poster_stowers"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    vals = {"id": [], "ood": [], "snv_delta": []}
    for f in sorted(
        (REPO / "outputs" / "ag_hashfrag_oracle_cached").glob("oracle_*/test_metrics.json")
    ):
        d = json.loads(f.read_text())
        tm = d["test_metrics"]
        for k_src, k_dst in [
            ("in_distribution", "id"),
            ("in_dist", "id"),
            ("ood", "ood"),
            ("snv_delta", "snv_delta"),
        ]:
            if k_src in tm and k_dst not in [
                k for k, v in vals.items() if len(v) == len(vals["id"])
            ]:
                vals[k_dst].append(tm[k_src]["pearson_r"])

    metrics = {
        "In-Distribution\n(Chr 7/13 Holdout)": ("id", "#2980B9"),
        "SNV Effect (Δ)\n(Variant Pairs)": ("snv_delta", "#8E44AD"),
        "Designed CREs\n(OOD)": ("ood", "#E74C3C"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(metrics))
    means = []
    stds = []
    colors = []
    labels = []
    for label, (key, color) in metrics.items():
        v = vals[key]
        means.append(np.mean(v))
        stds.append(np.std(v))
        colors.append(color)
        labels.append(label)

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="white",
        linewidth=1,
        width=0.6,
    )

    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Pearson R (vs Real MPRA Labels)", fontsize=12)
    ax.set_title(
        "AlphaGenome S2 Oracle Performance (5-Fold Ensemble)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "panel_oracle_barplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "panel_oracle_barplot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: panel_oracle_barplot.png")


if __name__ == "__main__":
    main()
