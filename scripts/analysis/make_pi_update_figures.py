"""PI-update figures for the HP-optimization methodology (Phase 0 -> 1 -> 2).

Three self-contained figures (matplotlib only, no data deps):
  1. method_overview        - the three-phase pipeline (what we are doing & why)
  2. phase0_persona_result  - LLM persona screen: personas are complementary
  3. phase1_bakeoff_concept - the GPU-seconds efficiency-curve idea + live status

Backing numbers are the Phase-0 CONFIRM bundle (K562, genomic reservoir, D=30k,
3 covaried seeds, matched depth R=13); ensemble oracle-r via ElasticNetCV(positive,
cv=5) over each persona's best-solo-val atom. Update LIVE_COUNTS before re-running.

Run locally:
    python scripts/analysis/make_pi_update_figures.py --out_dir figures_schematics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
ORANGE = "#DD8452"
PURPLE = "#8172B3"
GREY = "#7f7f7f"

# --- Phase-0 confirm bundle (R=13, 3 seeds): ensemble oracle-r by persona subset ---
# mean, cross-seed SD
ENSEMBLE = {
    "diverse\n(best single)": (0.701, 0.013, GREY),
    "exploit+critic\n+diverse\n(chosen-3)": (0.7180, 0.0004, RED),
    "critic+diverse\n+explore\n(best size-3)": (0.7199, 0.0027, ORANGE),
    "all four\npersonas": (0.7223, 0.004, GREEN),
}
# diverse novel-axes effect (solo oracle-r)
DIVERSE_NV = {"nv0\n(no novel axes)": (0.6795, 0.0036), "nv1\n(novel axes)": (0.6959, 0.0063)}

# --- live anchor status: models trained per strategy (summed over 3 seeds), target 600 ---
LIVE_TARGET = 600  # 200 models/cell x 3 seeds
LIVE_COUNTS = {
    "evo_explore": 380,
    "random": 379,
    "evo_exploit": 343,
    "evo_knowledgeable": 317,
    "evo_single": 325,
    "optuna_cmaes": 308,
    "evo_batch": 296,
    "optuna_qmc": 284,
    "evo_adaptive": 284,
    "evo_massive": 234,
    "optuna_gp": 204,
    "optuna_tpe": 182,
    "llm_exploit_nv1": 46,
    "llm_critic_nv0": 19,
    "llm_diverse_nv1": 12,
    "llm_explore_nv1": 2,
}


def _box(ax, xy, w, h, text, edge, fc, fontsize=9, weight="normal", tc="black"):
    ax.add_patch(
        FancyBboxPatch(
            (xy[0] - w / 2, xy[1] - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=fc,
            alpha=0.95,
        )
    )
    ax.text(
        xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize, color=tc, fontweight=weight
    )


def _arrow(ax, p0, p1, color="black", lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def fig_method_overview(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.text(
        6,
        10.6,
        "Picking ONE compute-efficient HP-search recipe for training student models",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        6,
        10.1,
        "(LegNet students distilled from a frozen AlphaGenome oracle; goal = strong, "
        "diverse model ensemble at every dataset size D)",
        ha="center",
        fontsize=9.5,
        color=GREY,
        style="italic",
    )

    # Phase 0
    _box(ax, (2.2, 8.7), 3.6, 1.0, "PHASE 0\nLLM persona screen", BLUE, "#e7edf6", 12, "bold")
    _box(
        ax,
        (2.2, 7.2),
        3.8,
        1.5,
        "Which LLM 'proposer' personas\nhelp? Screen styles x novel axes.\n"
        "-> 4 complementary personas:\nexploit | critic | diverse | explore",
        BLUE,
        "#f2f5fb",
        8.5,
    )

    # Phase 1
    _box(ax, (6, 8.7), 3.6, 1.0, "PHASE 1\nstrategy bake-off", RED, "#fbeaea", 12, "bold")
    _box(
        ax,
        (6, 7.0),
        4.0,
        1.9,
        "Run 16 HP-search strategies DEEP\non ONE anchor cell\n"
        "(genomic, D=30k, random acq.)\n"
        "baselines | Bayesian-opt | evolutionary\n| Ray | the 4 LLM personas",
        RED,
        "#fdf3f3",
        8.5,
    )

    # Phase 2
    _box(ax, (9.8, 8.7), 3.6, 1.0, "PHASE 2\ndeploy", GREEN, "#e9f3ed", 12, "bold")
    _box(
        ax,
        (9.8, 7.2),
        3.6,
        1.5,
        "Freeze ONE recipe\n{strategies, compute}\nand reuse it at every\ndataset size D",
        GREEN,
        "#f1f8f3",
        9,
    )

    _arrow(ax, (4.1, 7.2), (4.0, 7.0), BLUE)
    _arrow(ax, (8.0, 7.0), (8.0, 7.2), RED)

    # the fair-currency callout under Phase 1
    _box(
        ax,
        (6, 4.7),
        7.6,
        1.25,
        "FAIR CURRENCY: ensemble accuracy (oracle-r) vs CUMULATIVE GPU-SECONDS\n"
        "knee of each strategy's curve = its optimal compute    |    curve height = which strategies win\n"
        "(absorbs the 'expensive proposals' confound automatically)",
        "#b58a00",
        "#fdf7e3",
        9,
        "normal",
    )
    _arrow(ax, (6, 6.05), (6, 5.35), "#b58a00")

    # OFAT transfer probes
    _box(
        ax,
        (6, 2.6),
        8.4,
        1.25,
        "TRANSFER CHECK (OFAT probes): perturb ONE axis off the anchor and confirm the ranking holds\n"
        "D: 30k -> 100k -> 300k     reservoir: genomic -> random -> motif-planted     "
        "acquisition: random -> uncertainty -> diversity",
        PURPLE,
        "#f1eff8",
        8.8,
    )
    _arrow(ax, (6, 4.05), (6, 3.25), PURPLE)

    ax.text(
        6,
        1.05,
        "Why one deep run per cell? A strategy's value IS the shape of its accuracy-vs-compute curve,\n"
        "so 'optimal compute' (the knee) and 'best strategy' (the height) are read from the SAME curves.",
        ha="center",
        fontsize=9,
        color=GREEN,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pi_method_overview.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_phase0_persona_result(out_dir: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})

    labels = list(ENSEMBLE)
    means = [ENSEMBLE[k][0] for k in labels]
    sds = [ENSEMBLE[k][1] for k in labels]
    colors = [ENSEMBLE[k][2] for k in labels]
    xs = range(len(labels))
    axL.bar(
        xs, means, yerr=sds, capsize=5, color=colors, edgecolor="black", linewidth=1.2, alpha=0.9
    )
    axL.set_xticks(list(xs))
    axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel("ensemble accuracy\n(oracle Pearson r)", fontsize=10)
    axL.set_ylim(0.69, 0.726)
    axL.set_title(
        "Phase 0: the LLM personas are COMPLEMENTARY\n"
        "(K562, genomic, D=30k, 3 seeds; ElasticNetCV ensemble)",
        fontsize=11,
        fontweight="bold",
    )
    for x, m in zip(xs, means):
        axL.text(x, m + 0.0012, f"{m:.3f}", ha="center", fontsize=8.5)
    axL.axhline(ENSEMBLE[labels[0]][0], color=GREY, ls="--", lw=1, alpha=0.7)
    axL.annotate(
        "ensembling 4 personas beats the\nbest single by ~0.021; the size-3\ncombos are statistically tied",
        xy=(3, 0.7223),
        xytext=(-0.45, 0.7248),
        ha="left",
        va="top",
        fontsize=8.5,
        color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
    )
    axL.grid(axis="y", alpha=0.3)

    # right: diverse nv0 vs nv1
    nlabels = list(DIVERSE_NV)
    nmeans = [DIVERSE_NV[k][0] for k in nlabels]
    nsds = [DIVERSE_NV[k][1] for k in nlabels]
    axR.bar(
        [0, 1],
        nmeans,
        yerr=nsds,
        capsize=5,
        color=[GREY, BLUE],
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9,
        width=0.55,
    )
    axR.set_xticks([0, 1])
    axR.set_xticklabels(nlabels, fontsize=9)
    axR.set_ylim(0.66, 0.71)
    axR.set_ylabel("solo accuracy (oracle r)", fontsize=10)
    axR.set_title(
        "'Novel off-menu HP axes'\nhelp the diverse persona\n(+0.016, all 3 seeds)",
        fontsize=10.5,
        fontweight="bold",
    )
    for x, m in zip([0, 1], nmeans):
        axR.text(x, m + 0.0015, f"{m:.3f}", ha="center", fontsize=9)
    axR.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pi_phase0_persona_result.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_phase1_status(out_dir: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1, 1]})

    # LEFT: the efficiency-curve concept (illustrative)
    import numpy as np

    g = np.linspace(0.2, 10, 100)
    curves = {
        "strong strategy": (RED, 0.78 - 0.30 * np.exp(-0.55 * g)),
        "baseline (random)": (GREY, 0.74 - 0.28 * np.exp(-0.5 * g)),
        "expensive-proposal\nstrategy": (ORANGE, 0.79 - 0.34 * np.exp(-0.22 * g)),
    }
    for name, (c, y) in curves.items():
        axL.plot(g, y, lw=2.2, color=c, label=name)
    # mark a knee
    axL.scatter(
        [2.2], [0.78 - 0.30 * np.exp(-0.55 * 2.2)], s=90, color=RED, zorder=5, edgecolor="black"
    )
    axL.annotate(
        "knee = optimal compute\n(Kneedle)",
        xy=(2.2, 0.69),
        xytext=(3.4, 0.60),
        fontsize=8.5,
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    axL.set_xlabel("cumulative GPU-seconds  (the fair currency)", fontsize=10)
    axL.set_ylabel("ensemble accuracy (oracle r)", fontsize=10)
    axL.set_title(
        "Phase 1 reads TWO answers off one curve set:\n"
        "where to stop (knee) + which strategy wins (height)",
        fontsize=11,
        fontweight="bold",
    )
    axL.legend(fontsize=8.5, loc="lower right")
    axL.grid(alpha=0.3)

    # RIGHT: live anchor progress
    items = sorted(LIVE_COUNTS.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v / LIVE_TARGET * 100 for _, v in items]
    cols = [
        RED
        if n.startswith("llm_")
        else (GREEN if n.startswith("evo_") else (BLUE if n.startswith("optuna_") else GREY))
        for n in names
    ]
    yy = range(len(names))
    axR.barh(list(yy), vals, color=cols, edgecolor="black", linewidth=0.7, alpha=0.9)
    axR.set_yticks(list(yy))
    axR.set_yticklabels(names, fontsize=8)
    axR.set_xlabel("% of 600-model target (200/cell x 3 seeds)", fontsize=9.5)
    axR.set_xlim(0, 105)
    axR.axvline(100, color="black", ls=":", lw=1)
    axR.set_title(
        "Live anchor status (genomic, D=30k)\n"
        "algo/evo nearly done; LLM personas ramping (explore = long pole)",
        fontsize=10.5,
        fontweight="bold",
    )
    from matplotlib.patches import Patch

    axR.legend(
        handles=[
            Patch(color=GREY, label="baseline"),
            Patch(color=BLUE, label="Bayesian-opt"),
            Patch(color=GREEN, label="evolutionary"),
            Patch(color=RED, label="LLM persona"),
        ],
        fontsize=7.5,
        loc="lower right",
    )
    axR.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pi_phase1_status.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures_schematics")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_method_overview(out)
    fig_phase0_persona_result(out)
    fig_phase1_status(out)
    print(f"wrote PI-update figures to {out.resolve()}")


if __name__ == "__main__":
    main()
