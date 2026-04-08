#!/usr/bin/env python3
"""Generate presentation-quality figures for Peter's NYGC/MSKCC talks.

Panel A: Exp0 scaling curves — performance vs training data size
         Shows data regimes (small/medium/large) with annotations
Panel B: Exp1.1 strategy comparison — which reservoir strategies beat random
         Shows scaling behavior of different data generation strategies

Key messages:
- Foundation models (AG) have flat scaling (pretrained knowledge dominates)
- From-scratch models show steep scaling (data-hungry)
- Strategic data selection can beat random at every scale
- Identifies the data regimes where AL can have the most impact
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "talk_figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_exp0_scaling():
    """Load Exp0 K562 scaling curve data (AG oracle, in_dist Pearson R)."""
    sizes = [3197, 6395, 15987, 31974, 63949, 159871, 319742]
    models = {
        "AG S1 (Probing)": ("alphagenome_k562_s1", "#80A0C7"),
        "DREAM-CNN": ("dream_cnn", "#9B59B6"),
        "LegNet": ("legnet", "#D4A017"),
        "DREAM-RNN": ("dream_rnn", "#8B9DAF"),
    }

    data = {}
    for display_name, (model_name, color) in models.items():
        means, stds = [], []
        for n in sizes:
            vals = []
            for f in glob(
                str(
                    REPO
                    / f"outputs/exp0_oracle_scaling_v4/k562/{model_name}/random/n{n}/hp*/seed*/result.json"
                )
            ):
                d = json.load(open(f))
                p = d.get("test_metrics", {}).get("in_dist", {}).get("pearson_r")
                if p:
                    vals.append(p)
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
            else:
                means.append(np.nan)
                stds.append(0)
        data[display_name] = {
            "sizes": sizes,
            "means": np.array(means),
            "stds": np.array(stds),
            "color": color,
        }
    return data


def load_strategy_comparison():
    """Load Exp1.1 strategy comparison data.

    Uses DREAM-RNN student with DREAM-RNN oracle (shows most variation).
    Data is at exp1_1/k562/dream_rnn_dream_rnn/{strategy}/n*/hp*/seed*/result.json
    """
    base = REPO / "outputs" / "exp1_1" / "k562" / "dream_rnn_dream_rnn"

    strat_map = [
        ("random", "Random (Baseline)", "#E74C3C"),
        ("genomic", "Genomic", "#2ECC71"),
        ("dinuc_shuffle", "Dinuc. Shuffle", "#3498DB"),
        ("evoaug_heavy", "EvoAug", "#9B59B6"),
        ("recombination_uniform", "Recombination", "#E67E22"),
        ("gc_matched", "GC-Matched", "#1ABC9C"),
        ("motif_grammar", "Motif Grammar", "#F39C12"),
    ]

    # Find all available sizes
    sizes_found = set()
    for strat_dir, _, _ in strat_map:
        for d in (base / strat_dir).glob("n*"):
            try:
                sizes_found.add(int(d.name[1:]))
            except ValueError:
                pass
    sizes = sorted(sizes_found)
    if not sizes:
        return {}, []

    strategies = {}
    for strat_dir, display, color in strat_map:
        means, stds, valid_sizes = [], [], []
        for n in sizes:
            vals = []
            for f in glob(str(base / strat_dir / f"n{n}" / "hp*/seed*/result.json")):
                d = json.load(open(f))
                p = d.get("test_metrics", {}).get("in_dist", {}).get("pearson_r")
                if p:
                    vals.append(p)
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
                valid_sizes.append(n)

        if valid_sizes:
            strategies[display] = {
                "sizes": valid_sizes,
                "means": np.array(means),
                "stds": np.array(stds),
                "color": color,
            }

    return strategies, sizes


def plot_talk_figure():
    """Create the 2-panel talk figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Panel A: Exp0 Scaling Curves ──────────────────────────────────
    data = load_exp0_scaling()

    for name, d in data.items():
        valid = ~np.isnan(d["means"])
        sizes = np.array(d["sizes"])[valid]
        means = d["means"][valid]
        stds = d["stds"][valid]

        ax1.plot(
            sizes, means, "o-", color=d["color"], label=name, linewidth=2, markersize=6, zorder=5
        )
        if stds.any():
            ax1.fill_between(sizes, means - stds, means + stds, alpha=0.15, color=d["color"])

    # Annotate data regimes
    ax1.axvspan(1000, 10000, alpha=0.05, color="red", zorder=0)
    ax1.axvspan(10000, 100000, alpha=0.05, color="orange", zorder=0)
    ax1.axvspan(100000, 400000, alpha=0.05, color="green", zorder=0)
    ax1.text(5000, 0.52, "Small\ndata", ha="center", fontsize=9, color="#C0392B", fontweight="bold")
    ax1.text(
        35000, 0.52, "Medium\ndata", ha="center", fontsize=9, color="#E67E22", fontweight="bold"
    )
    ax1.text(
        200000, 0.52, "Large\ndata", ha="center", fontsize=9, color="#27AE60", fontweight="bold"
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("N training sequences", fontsize=12)
    ax1.set_ylabel("In-Distribution Pearson R", fontsize=12)
    ax1.set_title("A. Data Scaling Behavior (K562, AG Oracle)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0.45, 1.0)
    ax1.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white", edgecolor="gray")
    ax1.grid(alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Strategy Comparison ──────────────────────────────────
    strategies, all_sizes = load_strategy_comparison()

    if strategies:
        for name, d in strategies.items():
            linestyle = "--" if name == "Random (Baseline)" else "-"
            linewidth = 2.5 if name == "Random (Baseline)" else 1.8
            ax2.plot(
                d["sizes"],
                d["means"],
                "o" + linestyle,
                color=d["color"],
                label=name,
                linewidth=linewidth,
                markersize=5,
                zorder=5,
            )
            if d["stds"].any():
                ax2.fill_between(
                    d["sizes"],
                    d["means"] - d["stds"],
                    d["means"] + d["stds"],
                    alpha=0.1,
                    color=d["color"],
                )

        ax2.set_xscale("log")
        ax2.set_xlabel("N training sequences", fontsize=12)
        ax2.set_ylabel("In-Distribution Pearson R", fontsize=12)
        ax2.set_title(
            "B. Reservoir Strategy Comparison (K562, DREAM-RNN)", fontsize=13, fontweight="bold"
        )
        ax2.legend(
            fontsize=9, loc="lower right", frameon=True, facecolor="white", edgecolor="gray", ncol=1
        )
        ax2.grid(alpha=0.3, zorder=0)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    else:
        ax2.text(
            0.5,
            0.5,
            "Strategy comparison data\nnot available in expected format",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("B. Reservoir Strategy Comparison", fontsize=13, fontweight="bold")

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "scaling_and_strategy_2panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "scaling_and_strategy_2panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'scaling_and_strategy_2panel.png'}")

    # Also save Panel A alone (cleaner for single-slide use)
    fig_a, ax_a = plt.subplots(figsize=(8, 6))
    for name, d in data.items():
        valid = ~np.isnan(d["means"])
        sizes = np.array(d["sizes"])[valid]
        means = d["means"][valid]
        stds = d["stds"][valid]
        ax_a.plot(
            sizes, means, "o-", color=d["color"], label=name, linewidth=2.5, markersize=8, zorder=5
        )
        if stds.any():
            ax_a.fill_between(sizes, means - stds, means + stds, alpha=0.15, color=d["color"])
        # Add value labels at first and last point
        if len(means) > 0:
            ax_a.annotate(
                f"{means[0]:.3f}",
                (sizes[0], means[0]),
                textcoords="offset points",
                xytext=(-5, 8),
                fontsize=8,
                color=d["color"],
                fontweight="bold",
            )
            ax_a.annotate(
                f"{means[-1]:.3f}",
                (sizes[-1], means[-1]),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=8,
                color=d["color"],
                fontweight="bold",
            )

    ax_a.axvspan(1000, 10000, alpha=0.06, color="red", zorder=0)
    ax_a.axvspan(10000, 100000, alpha=0.06, color="orange", zorder=0)
    ax_a.axvspan(100000, 400000, alpha=0.06, color="green", zorder=0)
    ax_a.text(
        5000, 0.48, "Small data", ha="center", fontsize=10, color="#C0392B", fontweight="bold"
    )
    ax_a.text(
        35000, 0.48, "Medium data", ha="center", fontsize=10, color="#E67E22", fontweight="bold"
    )
    ax_a.text(
        200000, 0.48, "Large data", ha="center", fontsize=10, color="#27AE60", fontweight="bold"
    )

    ax_a.set_xscale("log")
    ax_a.set_xlabel("N training sequences", fontsize=13)
    ax_a.set_ylabel("In-Distribution Pearson R", fontsize=13)
    ax_a.set_title(
        "K562 MPRA — Data Scaling Behavior\n(Oracle-Labeled Training Data)",
        fontsize=14,
        fontweight="bold",
    )
    ax_a.set_ylim(0.42, 1.02)
    ax_a.legend(fontsize=11, loc="lower right", frameon=True, facecolor="white", edgecolor="gray")
    ax_a.grid(alpha=0.3, zorder=0)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    fig_a.tight_layout()
    fig_a.savefig(OUT / "scaling_curves_single.png", dpi=300, bbox_inches="tight")
    fig_a.savefig(OUT / "scaling_curves_single.pdf", bbox_inches="tight")
    plt.close(fig_a)
    print(f"Saved: {OUT / 'scaling_curves_single.png'}")


if __name__ == "__main__":
    plot_talk_figure()
