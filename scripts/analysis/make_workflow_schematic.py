"""Colorful, step-by-step schematic of the HP-optimization -> scaling-law-deploy
workflow ('find the recipe once, deploy it at every dataset size').

Two steps:
  STEP 1  FIND THE RECIPE (run once, at D in {30k,300k} x 3 reservoirs)
    A  Strategy comparison   — every strategy family, deep, GPU-seconds knee -> rounds/strategy
    B  Recipe composition    — exhaustive all-subsets ElasticNetCV -> best strategy set @ knee
  STEP 2  DEPLOY PER DATASET SIZE
    run frozen recipe per D -> pick one global N* (size-matched) -> distill -> validate -> deploy

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

# Canvas size in data units (also used to map insets to figure fractions).
W, H = 17.0, 25.0

# Phase palette
S1_BG = "#eef4fc"  # step 1 panel  (blue tint)
S2_BG = "#eef8f0"  # step 2 panel  (green tint)
A_EDGE = "#2f6db5"  # blue   — comparison
B_EDGE = "#d97b29"  # orange — recipe composition
C_EDGE = "#2f9e54"  # green  — deploy
INK = "#1d1d1d"
GREY = "#6f6f6f"

# Strategy-family chip colors
FAM = {
    "random\n(baseline / null)": "#9aa0a6",
    "Optuna-TPE\n(Bayesian)": "#26a69a",
    "Ray Tune schedulers\n(ASHA / BOHB / PBT)": "#7e57c2",
    "evo_*\n(evolutionary, ours)": "#ef6c92",
    "llm_autoresearch\n(LLM AutoResearch)": "#d7322e",
}


def _box(ax, x, y, w, h, text, edge, fc, fs=16, weight="normal", tc=INK, lw=2.0):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=lw,
            edgecolor=edge,
            facecolor=fc,
            alpha=0.97,
            zorder=3,
        )
    )
    ax.text(
        x, y, text, ha="center", va="center", fontsize=fs, color=tc, fontweight=weight, zorder=4
    )


def _panel(ax, x0, y0, x1, y1, color, label, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=2.4,
            edgecolor=edge,
            facecolor=color,
            alpha=0.55,
            zorder=0,
        )
    )
    ax.text(
        x0 + 0.42,
        (y0 + y1) / 2,
        label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=24,
        fontweight="bold",
        color=edge,
        zorder=1,
    )


def _arrow(ax, p0, p1, color=INK, lw=2.4, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=22,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
            zorder=2,
        )
    )


def _knee_inset(fig, cx, cy, w, h, kind):
    """Small illustrative diminishing-returns curve with a marked knee, placed at
    data-coord center (cx, cy) with size (w, h)."""
    rect = [(cx - w / 2) / W, (cy - h / 2) / H, w / W, h / H]
    ax = fig.add_axes(rect)
    x = np.linspace(0, 1, 60)
    if kind == "rounds":
        y = 0.62 + 0.17 * (1 - np.exp(-x * 7))
        kx, xlab, title, col = 0.16, "GPU-seconds", "rounds knee", A_EDGE
    elif kind == "recipe":
        y = 0.66 + 0.13 * (1 - np.exp(-x * 5))
        kx, xlab, title, col = 0.30, "# strategies", "recipe knee", B_EDGE
    else:  # N*
        y = 0.67 + 0.12 * (1 - np.exp(-x * 6))
        kx, xlab, title, col = 0.28, "# configs (N*)", "N* knee", C_EDGE
    ax.plot(x, y, color="#333333", lw=2.4)
    ky = np.interp(kx, x, y)
    ax.scatter([kx], [ky], color=col, zorder=5, s=46)
    ax.axvline(kx, color=col, ls=":", lw=1.6)
    ax.fill_between(x[x >= kx], y[x >= kx], ky, color=col, alpha=0.12)
    ax.set_title(title, fontsize=19, color=col, fontweight="bold")
    ax.set_xlabel(xlab, fontsize=17)
    # No ylabel and no numeric ticks: the curves are illustrative, and the
    # "test R" label / y-tick numbers used to bleed left into neighbouring
    # boxes and arrows. The title already names each curve.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.25)
    return ax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures_schematics")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---------- header ----------
    ax.text(
        W / 2,
        24.45,
        "HP-optimization → scaling-law deployment workflow",
        ha="center",
        fontsize=33,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        W / 2,
        23.6,
        "Goal: one consistent, size-matched set of N* model architectures per dataset size,\n"
        "reusable for ANY reservoir × acquisition combo — find the search recipe ONCE, deploy everywhere.",
        ha="center",
        fontsize=18,
        color=GREY,
        style="italic",
    )

    # ---------- strategy-family palette ----------
    _box(ax, W / 2, 22.35, 16.2, 1.55, "", "#b9c2cc", "#f4f6f9", lw=2.0)
    ax.text(
        W / 2,
        22.88,
        "HP-SEARCH STRATEGY FAMILIES  —  compare ALL of them",
        ha="center",
        fontsize=18.5,
        fontweight="bold",
        color=INK,
        zorder=5,
    )
    fam_x = np.linspace(2.45, 14.55, len(FAM))
    for x, (name, col) in zip(fam_x, FAM.items()):
        _box(ax, x, 22.0, 2.85, 0.95, name, col, "#ffffff", fs=14, tc=INK)
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.425, 22.0 - 0.475),
                0.14,
                0.95,
                boxstyle="square,pad=0",
                linewidth=0,
                facecolor=col,
                zorder=4,
            )
        )

    # =================== STEP 1 ===================
    _panel(ax, 0.35, 11.3, 16.65, 21.1, S1_BG, "STEP 1 — FIND THE RECIPE (run ONCE)", A_EDGE)

    # ----- Phase A -----
    ax.text(
        8.9,
        20.55,
        "A  ·  STRATEGY COMPARISON — run every family deep, in isolation",
        ha="center",
        fontsize=21,
        fontweight="bold",
        color=A_EDGE,
    )
    ax.text(
        8.9,
        20.0,
        "~50 rounds × multiple downsample + HP-init seeds, for every (D × reservoir) cell",
        ha="center",
        fontsize=16,
        color=GREY,
    )
    _box(
        ax,
        3.25,
        17.9,
        3.8,
        2.3,
        "TEST GRID\n\nD ∈ {30k, 300k}\n×\ngenomic\nmotif_planted_v2\ndinuc_shuffle",
        A_EDGE,
        "#ffffff",
        fs=16,
    )
    _box(
        ax,
        9.05,
        18.75,
        6.9,
        1.15,
        "train each config  →  save val/test preds\n"
        "+ train_time_sec  +  LLM $ / latency   →   *_meta.json",
        A_EDGE,
        "#ffffff",
        fs=14,
    )
    _box(
        ax,
        9.05,
        16.95,
        6.9,
        1.55,
        "efficiency curve vs cumulative GPU-SECONDS\n(fair cost axis)\n"
        "→ Kneedle + marginal-gain knee,\nbootstrapped over seeds\n"
        "→ OPTIMAL # ROUNDS per strategy",
        A_EDGE,
        "#dcebfb",
        fs=13.5,
        weight="bold",
    )
    _arrow(ax, (5.2, 17.9), (5.7, 18.45), A_EDGE)
    _arrow(ax, (9.05, 18.17), (9.05, 17.73), A_EDGE)
    _knee_inset(fig, 14.45, 17.8, 3.4, 2.5, "rounds")

    # A -> B connector (routed in the clear gap above the Phase-B header)
    _arrow(ax, (9.2, 16.17), (9.2, 15.55), B_EDGE, lw=2.8)

    # ----- Phase B -----
    ax.text(
        8.9,
        15.05,
        "B  ·  RECIPE COMPOSITION — which strategies work best together",
        ha="center",
        fontsize=21,
        fontweight="bold",
        color=B_EDGE,
    )
    ax.text(
        8.9,
        14.5,
        "pool ALL harvested configs (best @ each strategy's knee), across strategies",
        ha="center",
        fontsize=16,
        color=GREY,
    )
    _box(
        ax,
        3.5,
        12.75,
        4.8,
        1.95,
        "EXHAUSTIVE all-subsets search\nElasticNetCV(positive=True, cv=5)\n"
        "fit on validation preds\nover m = 3, 4, 5, 6, … strategies",
        B_EDGE,
        "#ffffff",
        fs=16,
    )
    # The recipe knee plot lives in the middle column (replaces the old text box,
    # so it has a clean, non-overlapping home).
    _knee_inset(fig, 8.7, 13.1, 3.8, 1.85, "recipe")
    _box(
        ax,
        13.55,
        12.75,
        4.1,
        1.95,
        "FROZEN RECIPE\n{ winning strategies,\n# rounds each }\n\nreused at EVERY D",
        B_EDGE,
        "#ffffff",
        fs=16,
        weight="bold",
    )
    _arrow(ax, (5.9, 12.75), (6.7, 12.75), B_EDGE)
    _arrow(ax, (10.7, 12.75), (11.5, 12.75), B_EDGE)
    ax.text(
        8.9,
        11.5,
        "persist every intermediate point (per-round preds, costs, all subset scores) "
        "→ re-tune knee criteria later",
        ha="center",
        va="center",
        fontsize=14,
        color=GREY,
        style="italic",
    )

    # big step transition arrow (clear gap between the two panels)
    _arrow(ax, (8.5, 11.25), (8.5, 10.65), C_EDGE, lw=3.6)

    # =================== STEP 2 ===================
    _panel(ax, 0.35, 0.5, 16.65, 10.55, S2_BG, "STEP 2 — DEPLOY PER DATASET SIZE", C_EDGE)
    ax.text(
        8.9,
        10.0,
        "Run the frozen recipe at EACH D  →  freeze a size-matched architecture set",
        ha="center",
        fontsize=21,
        fontweight="bold",
        color=C_EDGE,
    )
    _box(
        ax,
        3.2,
        8.4,
        4.0,
        1.8,
        "run FROZEN RECIPE per D\nD ∈ {10k, 30k, 100k,\n300k, 1M}   (+3M later)\n"
        "over representative reservoirs",
        C_EDGE,
        "#ffffff",
        fs=16,
    )
    _box(
        ax,
        8.7,
        8.4,
        4.9,
        1.8,
        "forward-selection ElasticNet\ncurve per D\n"
        "→ pick ONE global N* near-knee\nACROSS all D\n(size-matched ⇒ fair scaling)",
        C_EDGE,
        "#d7eede",
        fs=16,
        weight="bold",
    )
    _arrow(ax, (5.2, 8.4), (6.25, 8.4), C_EDGE)
    _knee_inset(fig, 13.95, 8.35, 3.5, 2.3, "Nstar")

    _box(
        ax,
        4.6,
        5.6,
        5.7,
        1.8,
        "distill to N* configs / D — select by:\n"
        "• ORACLE test-Pearson  (consistent landscape)\n"
        "• HP / architecture diversity\n"
        "• reservoir-TYPE coverage",
        C_EDGE,
        "#ffffff",
        fs=16,
    )
    _box(
        ax,
        11.35,
        5.6,
        5.7,
        1.8,
        "VALIDATE transfer penalty\nconfigs from reservoir A scored on B\n"
        "vs natively-searched on B\n"
        "(small gap ⇒ search 1 reservoir/D;\nalso used as a selection criterion)",
        C_EDGE,
        "#fff6e6",
        fs=16,
    )
    _arrow(ax, (8.7, 7.5), (5.0, 6.5), C_EDGE)
    _arrow(ax, (8.7, 7.5), (11.0, 6.5), C_EDGE)

    _box(
        ax,
        4.6,
        3.0,
        5.7,
        1.25,
        "DELIVERABLE\nper-D table of N* frozen architectures",
        C_EDGE,
        "#d7eede",
        fs=16,
        weight="bold",
    )
    _box(
        ax,
        11.35,
        3.0,
        5.7,
        1.25,
        "DEPLOY: re-train\n(transfer CONFIGS, not weights)\n"
        "+ ElasticNet-stack on every reservoir\n× acquisition × D — NO HP re-search",
        C_EDGE,
        "#d7eede",
        fs=16,
        weight="bold",
    )
    _arrow(ax, (4.6, 4.7), (4.6, 3.65), C_EDGE)
    _arrow(ax, (11.35, 4.7), (11.35, 3.65), C_EDGE)
    _arrow(ax, (7.45, 3.0), (8.5, 3.0), C_EDGE)

    ax.text(
        8.9,
        1.3,
        "All model selection on the ORACLE sequence-function landscape\n"
        "D=30k prioritized; D=300k preemptible / resumable   •   one-time per D ⇒ budget can run high",
        ha="center",
        fontsize=15,
        color=GREY,
        style="italic",
    )

    for ext in ("png", "pdf"):
        fig.savefig(out / f"schematic_hp_workflow.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote workflow schematic to {out.resolve()}")


if __name__ == "__main__":
    main()
