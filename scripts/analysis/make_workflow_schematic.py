"""Colorful, step-by-step schematic of the HP-optimization -> scaling-law-deploy
workflow ('find the recipe once, deploy it at every dataset size').

Two steps:
  STEP 1  FIND THE RECIPE (run once, at D in {30k,300k} x 3 reservoirs)
    A  Strategy bake-off     — every strategy family, deep, GPU-seconds knee -> rounds/strategy
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
W, H = 17.0, 23.0

# Phase palette
S1_BG = "#eef4fc"  # step 1 panel  (blue tint)
S2_BG = "#eef8f0"  # step 2 panel  (green tint)
A_EDGE = "#2f6db5"  # blue   — bake-off
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


def _box(ax, x, y, w, h, text, edge, fc, fs=12, weight="normal", tc=INK, lw=2.0):
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
        fontsize=16,
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
    ax.set_title(title, fontsize=11, color=col, fontweight="bold")
    ax.set_xlabel(xlab, fontsize=9.5)
    ax.set_ylabel("test R", fontsize=9.5)
    ax.tick_params(labelsize=7)
    ax.set_xticks([])
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
        22.45,
        "HP-optimization → scaling-law deployment workflow",
        ha="center",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        W / 2,
        21.75,
        "Goal: one consistent, size-matched set of N* model architectures per dataset size,\n"
        "reusable for ANY reservoir × acquisition combo — find the search recipe ONCE, deploy everywhere.",
        ha="center",
        fontsize=13,
        color=GREY,
        style="italic",
    )

    # ---------- strategy-family palette ----------
    _box(
        ax,
        W / 2,
        20.05,
        16.2,
        1.45,
        "",
        "#b9c2cc",
        "#f4f6f9",
        lw=2.0,
    )
    ax.text(
        W / 2,
        20.55,
        "HP-SEARCH STRATEGY FAMILIES  —  bake off ALL of them",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=INK,
        zorder=5,
    )
    fam_x = np.linspace(2.4, 14.6, len(FAM))
    for x, (name, col) in zip(fam_x, FAM.items()):
        _box(ax, x, 19.75, 2.65, 0.92, name, col, "#ffffff", fs=10.5, tc=INK)
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.325, 19.75 - 0.46),
                0.13,
                0.92,
                boxstyle="square,pad=0",
                linewidth=0,
                facecolor=col,
                zorder=4,
            )
        )

    # =================== STEP 1 ===================
    _panel(ax, 0.35, 9.7, 16.65, 18.55, S1_BG, "STEP 1 — FIND THE RECIPE (run ONCE)", A_EDGE)

    # ----- Phase A -----
    ax.text(
        8.7,
        18.05,
        "A  ·  STRATEGY BAKE-OFF — run every family deep, in isolation",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=A_EDGE,
    )
    ax.text(
        8.7,
        17.55,
        "~50 rounds × multiple downsample + HP-init seeds, for every (D × reservoir) cell",
        ha="center",
        fontsize=11,
        color=GREY,
    )
    _box(
        ax,
        3.25,
        15.85,
        3.6,
        2.0,
        "TEST GRID\n\nD ∈ {30k, 300k}\n×\ngenomic\nmotif_planted_v2\ndinuc_shuffle",
        A_EDGE,
        "#ffffff",
        fs=11.5,
    )
    _box(
        ax,
        9.1,
        16.55,
        6.2,
        1.05,
        "train each config  →  save val/test preds\n"
        "+ train_time_sec  +  LLM $ / latency   →   *_meta.json",
        A_EDGE,
        "#ffffff",
        fs=11.5,
    )
    _box(
        ax,
        9.1,
        14.95,
        6.2,
        1.25,
        "efficiency curve vs cumulative GPU-SECONDS  (fair cost axis)\n"
        "→ Kneedle + marginal-gain knee, bootstrapped over seeds\n"
        "→ OPTIMAL # ROUNDS per strategy",
        A_EDGE,
        "#dcebfb",
        fs=11,
        weight="bold",
    )
    _arrow(ax, (5.05, 15.85), (6.0, 16.4), A_EDGE)
    _arrow(ax, (9.1, 16.02), (9.1, 15.58), A_EDGE)
    _knee_inset(fig, 14.0, 15.7, 3.0, 1.9, "rounds")

    # ----- Phase B -----
    ax.text(
        8.7,
        13.75,
        "B  ·  RECIPE COMPOSITION — which strategies work best together",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=B_EDGE,
    )
    ax.text(
        8.7,
        13.28,
        "pool ALL harvested configs (best @ each strategy's knee), across strategies",
        ha="center",
        fontsize=11,
        color=GREY,
    )
    _box(
        ax,
        3.45,
        11.6,
        4.7,
        1.85,
        "EXHAUSTIVE all-subsets search\nElasticNetCV(positive=True, cv=5)\n"
        "fit on validation preds\nover m = 3, 4, 5, 6, … strategies",
        B_EDGE,
        "#ffffff",
        fs=11,
    )
    _box(
        ax,
        8.7,
        11.6,
        4.0,
        1.85,
        "curve: test R vs # strategies\n→ knee = 'enough'\n(diminishing returns)\n"
        "→ best recipe AT the knee",
        B_EDGE,
        "#fbe3cd",
        fs=11,
        weight="bold",
    )
    _box(
        ax,
        13.6,
        11.6,
        4.0,
        1.85,
        "FROZEN RECIPE\n{ winning strategies,\n# rounds each }\n\nreused at EVERY D",
        B_EDGE,
        "#ffffff",
        fs=11.5,
        weight="bold",
    )
    _arrow(ax, (5.85, 11.6), (6.65, 11.6), B_EDGE)
    _arrow(ax, (10.75, 11.6), (11.55, 11.6), B_EDGE)
    _arrow(ax, (9.1, 14.3), (9.1, 12.6), B_EDGE, lw=2.8)
    _knee_inset(fig, 8.7, 10.35, 3.0, 1.35, "recipe")
    ax.text(
        13.6,
        10.35,
        "persist every intermediate point\n(per-round preds, costs, all subset\n"
        "scores) → re-tune knee criteria later",
        ha="center",
        va="center",
        fontsize=9.5,
        color=GREY,
        style="italic",
    )

    # big step transition arrow
    _arrow(ax, (8.5, 9.6), (8.5, 9.12), C_EDGE, lw=3.4)

    # =================== STEP 2 ===================
    _panel(ax, 0.35, 0.4, 16.65, 9.0, S2_BG, "STEP 2 — DEPLOY PER DATASET SIZE", C_EDGE)
    ax.text(
        8.7,
        8.55,
        "Run the frozen recipe at EACH D  →  freeze a size-matched architecture set",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=C_EDGE,
    )
    _box(
        ax,
        3.15,
        7.0,
        3.9,
        1.7,
        "run FROZEN RECIPE per D\nD ∈ {10k, 30k, 100k,\n300k, 1M}   (+3M later)\n"
        "over representative reservoirs",
        C_EDGE,
        "#ffffff",
        fs=11,
    )
    _box(
        ax,
        8.6,
        7.0,
        4.8,
        1.7,
        "forward-selection ElasticNet\ncurve per D\n"
        "→ pick ONE global N* near-knee\nACROSS all D\n(size-matched ⇒ fair scaling)",
        C_EDGE,
        "#d7eede",
        fs=11,
        weight="bold",
    )
    _arrow(ax, (5.15, 7.0), (6.15, 7.0), C_EDGE)
    _knee_inset(fig, 13.7, 7.0, 3.0, 1.7, "Nstar")

    _box(
        ax,
        4.6,
        4.55,
        5.6,
        1.6,
        "distill to N* configs / D — select by:\n"
        "• ORACLE test-Pearson  (consistent landscape)\n"
        "• HP / architecture diversity\n"
        "• reservoir-TYPE coverage",
        C_EDGE,
        "#ffffff",
        fs=11,
    )
    _box(
        ax,
        11.3,
        4.55,
        5.6,
        1.6,
        "VALIDATE transfer penalty\nconfigs from reservoir A scored on B\n"
        "vs natively-searched on B\n"
        "(small gap ⇒ search 1 reservoir/D;\nalso used as a selection criterion)",
        C_EDGE,
        "#fff6e6",
        fs=10.5,
    )
    _arrow(ax, (8.6, 6.15), (4.9, 5.35), C_EDGE)
    _arrow(ax, (8.6, 6.15), (11.0, 5.35), C_EDGE)

    _box(
        ax,
        4.6,
        2.45,
        5.6,
        1.1,
        "DELIVERABLE\nper-D table of N* frozen architectures",
        C_EDGE,
        "#d7eede",
        fs=12,
        weight="bold",
    )
    _box(
        ax,
        11.3,
        2.45,
        5.6,
        1.1,
        "DEPLOY: re-train (transfer CONFIGS, not weights)\n"
        "+ ElasticNet-stack on every\nreservoir × acquisition × D — NO HP re-search",
        C_EDGE,
        "#d7eede",
        fs=10.5,
        weight="bold",
    )
    _arrow(ax, (4.6, 3.75), (4.6, 3.0), C_EDGE)
    _arrow(ax, (11.3, 3.75), (11.3, 3.0), C_EDGE)
    _arrow(ax, (7.4, 2.45), (8.5, 2.45), C_EDGE)

    ax.text(
        8.7,
        1.05,
        "All model selection on the ORACLE sequence-function landscape   •   "
        "D=30k prioritized; D=300k preemptible / resumable   •   one-time per D ⇒ budget can run high",
        ha="center",
        fontsize=10.5,
        color=GREY,
        style="italic",
    )

    for ext in ("png", "pdf"):
        fig.savefig(out / f"schematic_hp_workflow.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote workflow schematic to {out.resolve()}")


if __name__ == "__main__":
    main()
