"""Schematics comparing the previous scaling-law data pipeline to the current
HP-search + ElasticNetCV-pruning ensemble pipeline, plus an explainer for the
'mixed6' HP-search composition with backing data.

Run locally (matplotlib only, no data deps):
    python scripts/analysis/make_pipeline_schematics.py --out_dir figures_schematics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PREV = "#4C72B0"
CURR = "#C44E52"
ACCENT = "#55A868"
GREY = "#7f7f7f"

# Representative ablation numbers — K562, D=30k, mixed_genomic_random reservoir.
# budget_sweep: composition -> [(n_models, test_oracle_pearson), ...]
BUDGET_SWEEP = {
    "random only": [(6, 0.7208), (7, 0.7217)],
    "best single strategy\n(llm_exploit)": [(6, 0.7412), (8, 0.7428)],
    "mixed6 (3 algorithmic only)": [(6, 0.7228), (12, 0.7306), (22, 0.7353)],
    "all 6 strategies (full pool)": [(6, 0.7343), (12, 0.7416), (24, 0.7464), (46, 0.7498)],
}
SWEEP_COLOR = {
    "random only": GREY,
    "best single strategy\n(llm_exploit)": PREV,
    "mixed6 (3 algorithmic only)": "#DD8452",
    "all 6 strategies (full pool)": CURR,
}


def _box(ax, xy, w, h, text, color, fc=None, fontsize=9, text_color="black"):
    fc = fc if fc is not None else color
    box = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        edgecolor=color,
        facecolor=fc,
        alpha=0.92,
    )
    ax.add_patch(box)
    ax.text(
        xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize, color=text_color, wrap=True
    )
    return xy


def _arrow(ax, p0, p1, color="black"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def fig_pipeline_comparison(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(
        3,
        10.6,
        "PREVIOUS\nscaling-law data pipeline",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=PREV,
    )
    ax.text(
        9,
        10.6,
        "CURRENT\nHP-search + prune + ElasticNetCV",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=CURR,
    )

    shared_top = [
        "Reservoir sampler\n(random / PRM / motif / evoaug / …)",
        "Generate D sequences\n(D = 3k … 1M)",
        "Label with frozen oracle\n(AlphaGenome S2F)",
    ]
    # shared inputs (drawn once per column for clarity)
    for cx, col in ((3, PREV), (9, CURR)):
        y = 9.6
        prev_xy = None
        for t in shared_top:
            xy = _box(ax, (cx, y), 3.6, 0.7, t, col, fc="#eef2f8" if col == PREV else "#fbeeee")
            if prev_xy:
                _arrow(ax, (cx, prev_xy[1] - 0.35), (cx, y + 0.35), col)
            prev_xy = xy
            y -= 1.15

    # PREVIOUS column lower
    prev_steps = [
        ("Small fixed HP grid\nlr x batch_size  (2 configs)", "#eef2f8"),
        ("Train 1 student arch\n(DREAM-RNN / AG / LegNet / FM)\nx few seeds", "#eef2f8"),
        ("Select SINGLE best model\nby validation Pearson", "#dbe4f0"),
        ("-> 1 model per cell\nscaling-law point", "#c9d6ea"),
    ]
    y = 9.6 - 3 * 1.15
    prev_xy = (3, y + 1.15)
    for t, fc in prev_steps:
        xy = _box(ax, (3, y), 3.6, 0.78, t, PREV, fc=fc)
        _arrow(ax, (3, prev_xy[1] - 0.39), (3, y + 0.39), PREV)
        prev_xy = xy
        y -= 1.25

    # CURRENT column lower
    curr_steps = [
        (
            "mixed6 HP search (6 strategies)\n3 LLM-AutoResearch + 3 algorithmic\nexpanded HP space, multi-round",
            "#fbeeee",
        ),
        (
            "Large DIVERSE model pool\n(tens-hundreds of configs)\nblock x optimizer x depth/width",
            "#fbeeee",
        ),
        (
            "ElasticNetCV(positive=True, cv=5)\nfit on val predictions\n-> PRUNE: drop zero/neg-coef models",
            "#f6dada",
        ),
        ("-> weighted ensemble\nof surviving configs", "#efc9c9"),
    ]
    y = 9.6 - 3 * 1.15
    prev_xy = (9, y + 1.15)
    for t, fc in curr_steps:
        xy = _box(ax, (9, y), 3.8, 0.85, t, CURR, fc=fc)
        _arrow(ax, (9, prev_xy[1] - 0.42), (9, y + 0.42), CURR)
        prev_xy = xy
        y -= 1.25

    # contrast callouts
    ax.text(6, 5.0, "vs", ha="center", va="center", fontsize=16, color="black", style="italic")
    ax.text(
        6,
        1.2,
        "Key change: don't pre-commit to one HP config or one composition.\n"
        "Generate a diverse pool, then let ElasticNetCV select & weight the subset.",
        ha="center",
        va="center",
        fontsize=10,
        color=ACCENT,
        fontweight="bold",
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"schematic_pipeline_comparison.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_mixed6_explained(out_dir: Path) -> None:
    fig = plt.figure(figsize=(13.5, 6.2))
    axL = fig.add_axes([0.02, 0.05, 0.5, 0.9])
    axR = fig.add_axes([0.60, 0.13, 0.37, 0.78])
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 10)
    axL.axis("off")
    axL.set_title("How 'mixed6' is composed", fontsize=13, fontweight="bold")

    # global <-> local axis
    axL.annotate(
        "", xy=(9.4, 1.2), xytext=(0.6, 1.2), arrowprops=dict(arrowstyle="<->", lw=1.5, color=GREY)
    )
    axL.text(1.0, 0.7, "global / exploration", fontsize=9, color=GREY)
    axL.text(7.2, 0.7, "local / refinement", fontsize=9, color=GREY)

    # LLM-driven row
    axL.text(
        5,
        9.3,
        "3 LLM-driven AutoResearch  (share a knowledge base)",
        ha="center",
        fontsize=10,
        color=CURR,
        fontweight="bold",
    )
    llm = [
        (2.0, "llm_default\n(opus)\nbroad", CURR),
        (5.0, "llm_diverse\n(sonnet)\nspread", CURR),
        (8.0, "llm_exploit\n(sonnet)\nrefine", CURR),
    ]
    for x, t, c in llm:
        _box(axL, (x, 7.7), 2.3, 1.1, t, c, fc="#fbeeee", fontsize=8)
    # KB
    _box(
        axL,
        (5, 6.0),
        5.2,
        0.55,
        "shared knowledge base (learnings passed between LLM jobs)",
        ACCENT,
        fc="#e8f3ec",
        fontsize=8,
    )

    # Algorithmic row
    axL.text(5, 5.0, "3 algorithmic", ha="center", fontsize=10, color=PREV, fontweight="bold")
    algo = [
        (2.0, "autoresearch\n_massive\nglobal", PREV),
        (5.0, "random\nnull /\nbaseline", GREY),
        (8.0, "autoresearch\n_batch\nlocal", PREV),
    ]
    for x, t, c in algo:
        _box(axL, (x, 3.6), 2.3, 1.1, t, c, fc="#eef2f8", fontsize=8)

    # converge note
    axL.text(
        5,
        2.1,
        "all 6 -> one pooled set of trained configs -> ElasticNetCV prune+weight",
        ha="center",
        fontsize=9,
        color="black",
        style="italic",
    )

    # RIGHT: backing data — budget sweep crossover
    for comp, pts in BUDGET_SWEEP.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        axR.plot(xs, ys, "-o", ms=4, lw=1.8, color=SWEEP_COLOR[comp], label=comp)
    axR.set_title(
        "Backing data: diversity pays off as the pool grows\n(K562, D=30k, mixed_genomic_random)",
        fontsize=10.5,
    )
    axR.set_xlabel("ensemble size (models drawn, matched budget)")
    axR.set_ylabel("test Pearson R (vs oracle)")
    axR.grid(alpha=0.3)
    axR.legend(fontsize=7.5, loc="lower right")
    axR.annotate(
        "single strategy\nplateaus early",
        xy=(8, 0.7428),
        xytext=(9, 0.731),
        fontsize=7.5,
        color=PREV,
        arrowprops=dict(arrowstyle="->", color=PREV, lw=1),
    )
    axR.annotate(
        "full pool keeps\nimproving -> wins",
        xy=(46, 0.7498),
        xytext=(20, 0.748),
        fontsize=7.5,
        color=CURR,
        arrowprops=dict(arrowstyle="->", color=CURR, lw=1),
    )
    axR.text(
        0.02,
        -0.22,
        "Historical cost win: Phase-4 mixed6 (153 models) test R=0.788  >  "
        "Phase-2 16-strategy (301 models) R=0.780 — half the models.",
        transform=axR.transAxes,
        fontsize=7.8,
        color=ACCENT,
    )

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"schematic_mixed6_explained.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures_schematics")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_pipeline_comparison(out)
    fig_mixed6_explained(out)
    print(f"wrote schematics to {out.resolve()}")


if __name__ == "__main__":
    main()
