#!/usr/bin/env python3
"""3-panel bar plot: default-HP vs search-best vs search-stack across reservoirs/n_train.

Panels: ref_32k (in-dist activity), snv_delta_30k_mono (variant effect), ood (OOD designed).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("/tmp/stacking_pulls")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "scaling_barplots"
OUT.mkdir(parents=True, exist_ok=True)

RESERVOIRS = ["genomic", "random", "prm_1pct", "prm_20pct", "motif_grammar", "evoaug_heavy"]
N_TRAINS = [1000, 10000]
PANELS = [
    ("ref_32k", "Genomic Sequences (32k Ref)"),
    ("snv_delta_30k_mono", "SNV Effect (Δ on 30k mono)"),
    ("ood", "High-Activity Designed (22k OOD)"),
]
METHODS = ["default-HP", "search-best", "search-stack"]
COLORS = {"default-HP": "#888888", "search-best": "#1565C0", "search-stack": "#E8602C"}


def load(reservoir, n):
    base = DATA / reservoir / f"n{n}"
    default = best = stack = {p: float("nan") for p, _ in PANELS}
    default = {p: float("nan") for p, _ in PANELS}
    b_path = base / "baseline.json"
    if b_path.exists():
        b = json.loads(b_path.read_text())
        tm = b.get("test_metrics", {})
        for panel, _ in PANELS:
            default[panel] = tm.get(panel, {}).get("r", float("nan"))
    best = {p: float("nan") for p, _ in PANELS}
    stack = {p: float("nan") for p, _ in PANELS}
    s_path = base / "stacking_result.json"
    if s_path.exists():
        s = json.loads(s_path.read_text())
        for panel, _ in PANELS:
            p = s.get("panels", {}).get(panel, {})
            best[panel] = p.get("r_best_single", float("nan"))
            stack[panel] = p.get("r_stack", float("nan"))
    return default, best, stack


def main():
    data = {}
    for r in RESERVOIRS:
        for n in N_TRAINS:
            data[(r, n)] = load(r, n)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    x_labels = [f"{r}\nn={n}" for r in RESERVOIRS for n in N_TRAINS]
    n_groups = len(x_labels)
    x = np.arange(n_groups)
    bar_width = 0.27

    for ax_i, (panel_key, panel_label) in enumerate(PANELS):
        ax = axes[ax_i]
        for method_i, method in enumerate(METHODS):
            heights = []
            for r in RESERVOIRS:
                for n in N_TRAINS:
                    default, best, stack = data[(r, n)]
                    if method == "default-HP":
                        v = default[panel_key]
                    elif method == "search-best":
                        v = best[panel_key]
                    else:
                        v = stack[panel_key]
                    heights.append(v)
            offset = (method_i - 1) * bar_width
            bars = ax.bar(
                x + offset,
                heights,
                bar_width,
                label=method,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, heights):
                if val == val and abs(val) > 0.01:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.005 if val > 0 else -0.015),
                        f"{val:.2f}",
                        ha="center",
                        va="bottom" if val > 0 else "top",
                        fontsize=7,
                    )

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(f"Pearson R\n{panel_label}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x)
        if ax_i == 0:
            ax.legend(loc="upper right", fontsize=9, frameon=True)
            ax.set_title(
                "LegNet K562 scaling: default-HP vs HP-search-best vs HP-search-stack (oracle pseudolabels, chr 7/13 test)",
                fontsize=11,
                fontweight="bold",
                pad=8,
            )

    axes[-1].set_xticklabels(x_labels, fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", which="major", labelbottom=False)
    axes[-1].tick_params(axis="x", which="major", labelbottom=True)
    for tick in axes[-1].get_xticklabels():
        tick.set_rotation(0)

    fig.tight_layout()
    out_png = OUT / "scaling_barplot_3panel.png"
    out_pdf = OUT / "scaling_barplot_3panel.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
