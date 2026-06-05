"""Colorful, step-by-step schematic of the HP-optimization -> scaling-law-deploy
workflow (the 'search once per D, freeze configs, deploy everywhere' plan).

Three phases:
  A  Strategy bake-off      — every strategy family, deep runs, knee detection
  B  Ensemble composition   — ElasticNetCV over all strategy subsets, knee
  C  Freeze & deploy        — distill to N~=5 diverse configs per D, train everywhere

Run locally (matplotlib only, no data deps):
    python scripts/analysis/make_workflow_schematic.py --out_dir figures_schematics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Phase palette
A_BG, A_EDGE = "#eaf2fb", "#2f6db5"  # blue   — bake-off
B_BG, B_EDGE = "#fdf0e6", "#d97b29"  # orange — ensemble composition
C_BG, C_EDGE = "#eaf6ec", "#2f9e54"  # green  — deploy
INK = "#222222"
GREY = "#7f7f7f"

# Strategy-family chip colors
FAM = {
    "random": "#9aa0a6",
    "RayTune\n(ASHA/BOHB/PBT)": "#7e57c2",
    "Bayesian\n(Optuna-TPE)": "#26a69a",
    "Evolutionary\n(evo_* ours)": "#ef6c92",
    "LLM AutoResearch\n(opus/sonnet)": "#d7322e",
}


def _box(ax, x, y, w, h, text, edge, fc, fs=9, weight="normal", tc=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=fc,
            alpha=0.96,
            zorder=3,
        )
    )
    ax.text(
        x, y, text, ha="center", va="center", fontsize=fs, color=tc, fontweight=weight, zorder=4
    )


def _band(ax, y0, y1, color, label):
    ax.add_patch(
        FancyBboxPatch(
            (0.15, y0),
            13.7,
            y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0,
            facecolor=color,
            alpha=0.45,
            zorder=0,
        )
    )
    ax.text(
        0.38,
        (y0 + y1) / 2,
        label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=12,
        fontweight="bold",
        color=INK,
        zorder=1,
    )


def _arrow(ax, p0, p1, color=INK, lw=1.8, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=15,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
            zorder=2,
        )
    )


def _knee_inset(fig, rect, kind):
    """Small illustrative diminishing-returns curve with a marked knee."""
    ax = fig.add_axes(rect)
    x = np.linspace(0, 1, 60)
    if kind == "rounds":
        y = 0.62 + 0.17 * (1 - np.exp(-x * 7))
        kx = 0.16
        xlab, title = "cost (GPU-h)", "rounds knee"
    else:  # ensemble size
        y = 0.66 + 0.13 * (1 - np.exp(-x * 5))
        kx = 0.28
        xlab, title = "# configs (N*)", "N* knee"
    ax.plot(x, y, color="#333333", lw=2)
    ky = np.interp(kx, x, y)
    ax.scatter([kx], [ky], color="#d7322e", zorder=5, s=28)
    ax.axvline(kx, color="#d7322e", ls=":", lw=1)
    ax.fill_between(x[x >= kx], y[x >= kx], ky, color="#d7322e", alpha=0.10)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlab, fontsize=7)
    ax.set_ylabel("test R", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xticks([])
    ax.grid(alpha=0.25)
    return ax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures_schematics")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(
        7,
        10.62,
        "HP-optimization → scaling-law deployment workflow",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        7,
        10.18,
        "Goal: one consistent, size-matched set of N* model architectures per dataset size, "
        "reusable for ANY reservoir × acquisition combo — search strategy ONCE, deploy everywhere.",
        ha="center",
        fontsize=9.5,
        color=GREY,
        style="italic",
    )

    # ---- bands ----
    _band(ax, 6.7, 9.9, A_BG, "A  STRATEGY BAKE-OFF  (once)")
    _band(ax, 3.5, 6.4, B_BG, "B  RECIPE COMPOSITION  (once)")
    _band(ax, 0.25, 3.2, C_BG, "C  PER-D DEPLOY")

    # ===== Phase A =====
    ax.text(
        7.1,
        9.62,
        "Run every strategy family in isolation — deep (~50 rounds) × multi-seed",
        ha="center",
        fontsize=9.5,
        color=A_EDGE,
        fontweight="bold",
    )
    fam_x = np.linspace(2.0, 9.6, len(FAM))
    for x, (name, col) in zip(fam_x, FAM.items()):
        _box(ax, x, 8.95, 1.62, 0.78, name, col, "#ffffff", fs=7.4, tc=INK)
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.81, 8.95 - 0.39),
                0.10,
                0.78,
                boxstyle="square,pad=0",
                linewidth=0,
                facecolor=col,
                zorder=4,
            )
        )
    # grid chip
    _box(
        ax,
        12.0,
        8.95,
        2.6,
        1.5,
        "Test grid\n\nD ∈ {30k, 300k}\n× reservoirs:\ngenomic\nmotif_planted_v2\ndinuc_shuffle",
        A_EDGE,
        "#ffffff",
        fs=7.6,
    )
    for x in fam_x:
        _arrow(ax, (x, 8.55), (x, 8.05), GREY, lw=1.2)
    _box(
        ax,
        5.8,
        7.6,
        6.6,
        0.62,
        "train each config  →  save val/test preds + train_time_sec (cost)  →  *_meta.json",
        A_EDGE,
        "#ffffff",
        fs=8.2,
    )
    _arrow(ax, (5.8, 7.29), (5.8, 7.0), A_EDGE)
    _box(
        ax,
        5.8,
        6.95,
        6.6,
        0.5,
        "efficiency curve vs GPU-seconds (fair axis)  →  Kneedle + marginal-gain knee  →  optimal rounds / strategy",
        A_EDGE,
        "#dcebfb",
        fs=8.0,
        weight="bold",
    )
    _arrow(ax, (5.8, 6.7), (5.8, 6.42), B_EDGE)

    # ===== Phase B =====
    ax.text(
        7.1,
        6.12,
        "Pool harvested configs — find the best STRATEGY RECIPE (which strategies, how deep)",
        ha="center",
        fontsize=9.5,
        color=B_EDGE,
        fontweight="bold",
    )
    _box(
        ax,
        3.2,
        5.35,
        3.4,
        0.95,
        "ElasticNetCV(positive, cv=5)\nover ALL strategy subsets\n(m = 1, 2, 3, ... K)",
        B_EDGE,
        "#ffffff",
        fs=8.2,
    )
    _arrow(ax, (4.9, 5.35), (6.1, 5.35), B_EDGE)
    _box(
        ax,
        7.8,
        5.35,
        3.0,
        0.95,
        "curve: test R vs # strategies\n→ knee = 'enough'\n(diminishing returns)",
        B_EDGE,
        "#fbe3cd",
        fs=8.2,
        weight="bold",
    )
    _arrow(ax, (9.3, 5.35), (10.5, 5.35), B_EDGE)
    _box(
        ax,
        11.9,
        5.35,
        2.6,
        0.95,
        "FROZEN RECIPE:\nwinning strategy set\n+ rounds / strategy\n(reused at every D)",
        B_EDGE,
        "#ffffff",
        fs=8.2,
    )
    ax.text(
        7.0,
        4.05,
        "persist every intermediate point (per-round preds, costs, all subset scores) "
        "so knee criteria can be re-tuned later",
        ha="center",
        fontsize=8,
        color=GREY,
        style="italic",
    )
    _arrow(ax, (7.0, 3.75), (7.0, 3.18), C_EDGE)

    # ===== Phase C =====
    ax.text(
        7.1,
        2.92,
        "Use the winning recipe to produce frozen, transferable architectures",
        ha="center",
        fontsize=9.5,
        color=C_EDGE,
        fontweight="bold",
    )
    _box(
        ax,
        2.8,
        2.05,
        3.4,
        1.1,
        "run winning recipe per D\nD ∈ {10k,30k,100k,300k,1M}\nover representative reservoirs",
        C_EDGE,
        "#ffffff",
        fs=8.0,
    )
    _arrow(ax, (4.5, 2.05), (5.7, 2.05), C_EDGE)
    _box(
        ax,
        7.5,
        2.05,
        3.6,
        1.1,
        "distill to N* configs / D\nN* FIXED across D (fair scaling)\n• perf + HP/arch diversity\n• reservoir-TYPE coverage",
        C_EDGE,
        "#d7eede",
        fs=8.0,
        weight="bold",
    )
    _arrow(ax, (9.3, 2.05), (10.5, 2.05), C_EDGE)
    _box(
        ax,
        12.1,
        2.05,
        3.1,
        1.1,
        "DELIVERABLE:\nper-D table of N*\nfrozen architectures",
        C_EDGE,
        "#ffffff",
        fs=8.2,
        weight="bold",
    )
    ax.text(
        7.0,
        1.42,
        "VALIDATE transfer penalty: configs searched on reservoir A, scored on B vs natively-searched on B "
        "(small gap ⇒ search 1 reservoir/D) — also a selection criterion. All selection on ORACLE landscape.",
        ha="center",
        fontsize=8,
        color=GREY,
        style="italic",
    )
    # final deploy note
    _box(
        ax,
        7.0,
        0.72,
        9.2,
        0.62,
        "deploy: re-train (transfer configs, NOT weights) + ElasticNet-stack the frozen N* "
        "on EVERY reservoir × acquisition × D — no HP re-search",
        C_EDGE,
        "#d7eede",
        fs=8.4,
        weight="bold",
    )

    # ---- illustrative knee insets (figure coords) ----
    _knee_inset(fig, [0.70, 0.615, 0.095, 0.082], "rounds")
    _knee_inset(fig, [0.73, 0.345, 0.095, 0.082], "ensemble")

    for ext in ("png", "pdf"):
        fig.savefig(out / f"schematic_hp_workflow.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote workflow schematic to {out.resolve()}")


if __name__ == "__main__":
    main()
