#!/usr/bin/env python3
"""Schematic of the mixed6 constant-budget HP-optimization workflow.

A PI-facing conceptual diagram (not data-driven) meant to be iterated on:
edit the box text / layout below and re-run. Renders to
``results/diagrams/hp_workflow.{png,pdf}``.

    python scripts/analysis/plot_hp_workflow_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────
C_GRID = "#E8EEF7"
C_GRID_E = "#3B6CB7"
C_LOOP = "#FFF3E0"
C_LOOP_E = "#E08214"
C_PROP = "#FDE7EF"
C_PROP_E = "#C2185B"
C_INFRA = "#EAF5EA"
C_INFRA_E = "#2E7D32"
C_OUT = "#F3E9FB"
C_OUT_E = "#7B3FA0"
INK = "#1a1a1a"


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fc,
    ec,
    *,
    fs=10,
    weight="normal",
    rounding=0.02,
    lw=1.6,
    ha="center",
    align="center",
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.006,rounding_size={rounding}",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            mutation_aspect=1.0,
            zorder=2,
        )
    )
    tx = x + w / 2 if align == "center" else x + 0.012
    ax.text(
        tx,
        y + h / 2,
        text,
        ha=ha,
        va="center",
        fontsize=fs,
        color=INK,
        weight=weight,
        zorder=3,
        linespacing=1.35,
    )


def arrow(ax, p0, p1, *, color=INK, lw=1.8, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=16,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
            linestyle=ls,
        )
    )


fig, ax = plt.subplots(figsize=(16, 9.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.5)
ax.axis("off")

ax.text(
    8,
    9.15,
    "Mixed6 Constant-Budget HP-Optimization Workflow",
    ha="center",
    va="center",
    fontsize=17,
    weight="bold",
    color=INK,
)
ax.text(
    8,
    8.72,
    "240 cells  ·  48 models / cell  ·  2,880 LegNet students total",
    ha="center",
    va="center",
    fontsize=11,
    color="#555",
)

# ── A. experiment grid (top-left) ────────────────────────────────────────────
box(ax, 0.4, 6.35, 6.7, 2.0, "", C_GRID, C_GRID_E, rounding=0.03)
ax.text(
    3.75,
    8.12,
    "1  Experiment grid  →  240 cells",
    ha="center",
    fontsize=12,
    weight="bold",
    color=C_GRID_E,
)
ax.text(
    0.65,
    7.62,
    "Reservoirs (10)    ×    Dataset size D (6)    ×    Strategy (4)",
    ha="left",
    fontsize=10.2,
    weight="bold",
    color=INK,
)
ax.text(
    0.65,
    7.15,
    "• Reservoirs: pool-sampling\n  schemes for the training set\n"
    "• D ∈ {3k, 10k, 30k,\n  100k, 300k, 1M}",
    ha="left",
    va="top",
    fontsize=9.3,
    color="#333",
    linespacing=1.4,
)
ax.text(
    4.25,
    7.15,
    "• LLM-default  (Opus 4.7)\n• LLM-diverse  (Sonnet 4.6)\n"
    "• LLM-exploit  (Sonnet 4.6)\n• Algorithmic  (15 core axes)",
    ha="left",
    va="top",
    fontsize=9.3,
    color="#333",
    linespacing=1.4,
)

# ── B. per-cell loop (center) ────────────────────────────────────────────────
box(ax, 0.4, 1.55, 9.3, 4.45, "", "#FFFDFA", C_LOOP_E, rounding=0.02, lw=1.4)
ax.text(
    5.05,
    5.66,
    "2  Per-cell search loop  —  R rounds (constant compute budget)",
    ha="center",
    fontsize=12,
    weight="bold",
    color=C_LOOP_E,
)

# loop nodes
box(
    ax,
    0.75,
    3.9,
    3.0,
    1.25,
    "Propose HP config\n(proposer)",
    C_PROP,
    C_PROP_E,
    fs=10,
    weight="bold",
)
box(
    ax,
    6.3,
    3.9,
    3.0,
    1.0,
    "Train LegNet student\non D-sized pool",
    C_LOOP,
    C_LOOP_E,
    fs=10,
    weight="bold",
)
box(ax, 6.3, 2.35, 3.0, 0.95, "Evaluate\nval Pearson", C_LOOP, C_LOOP_E, fs=10, weight="bold")
box(
    ax,
    0.75,
    2.35,
    3.0,
    0.95,
    "Append (config, score)\nto cell history",
    C_LOOP,
    C_LOOP_E,
    fs=9.6,
    weight="bold",
)

# loop arrows (clockwise)
arrow(ax, (3.75, 4.55), (6.3, 4.4), rad=-0.12)  # propose -> train
arrow(ax, (7.8, 3.9), (7.8, 3.3))  # train -> eval
arrow(ax, (6.3, 2.82), (3.75, 2.82), rad=-0.1)  # eval -> history
arrow(ax, (2.25, 3.3), (2.25, 3.9))  # history -> propose
ax.text(2.3, 3.62, "history-conditioned", ha="center", fontsize=8.6, style="italic", color=C_PROP_E)
ax.text(5.0, 4.74, "R rounds", ha="center", fontsize=8.6, style="italic", color="#888")

# proposer detail callout
box(
    ax,
    0.75,
    1.78,
    8.5,
    0.5,
    "Proposer = LLM (Claude CLI; cell history in prompt; style-specific; novel axes gated OFF)"
    "   |   or   Algorithmic (random / autoresearch over 15 core axes)",
    "#fff",
    C_PROP_E,
    fs=8.6,
    lw=1.0,
    rounding=0.04,
)

# round badge
ax.add_patch(mpatches.Circle((4.95, 3.62), 0.27, fc="#fff", ec=C_LOOP_E, lw=1.6, zorder=5))
ax.text(
    4.95, 3.62, "R", ha="center", va="center", fontsize=12, weight="bold", color=C_LOOP_E, zorder=6
)
ax.text(
    4.95,
    3.16,
    "← PI question:\noptimal R?",
    ha="center",
    va="top",
    fontsize=8.2,
    color=C_PROP_E,
    weight="bold",
    linespacing=1.2,
)

# checkpoint note
ax.text(
    7.8,
    3.46,
    "ckpt → *_meta.json (resume-safe)",
    ha="center",
    fontsize=7.8,
    style="italic",
    color="#777",
)

# ── C. orchestration (bottom strip) ──────────────────────────────────────────
box(ax, 0.4, 0.35, 15.2, 1.0, "", C_INFRA, C_INFRA_E, rounding=0.02)
ax.text(0.62, 1.12, "3  Orchestration", ha="left", fontsize=11, weight="bold", color=C_INFRA_E)
ax.text(
    0.62,
    0.66,
    "SLURM array over GPU tiers (fast / default / slow_nice; V100 fp32-safe routing)"
    "      •      Watchdog every 20 min: resubmit FAILED / TIMEOUT / preempted / "
    "rate-limited cells, resume from checkpoints, rebalance queues, exit at 0 jobs"
    "      •      Rate-limit policy: LLM jobs pause & resume — never fall back to random",
    ha="left",
    va="center",
    fontsize=8.8,
    color="#2a4a2a",
)

# ── D. downstream (right) ────────────────────────────────────────────────────
box(ax, 10.1, 3.55, 5.5, 2.45, "", C_OUT, C_OUT_E, rounding=0.02)
ax.text(12.85, 5.66, "4  Downstream", ha="center", fontsize=12, weight="bold", color=C_OUT_E)
box(
    ax,
    10.45,
    4.55,
    4.8,
    0.85,
    "Aggregate all 240 cells\n(2,880 trained configs + scores)",
    "#fff",
    C_OUT_E,
    fs=9.6,
)
box(
    ax,
    10.45,
    3.7,
    4.8,
    0.7,
    "Over-generate → cluster → N = 5 deploy HP configs",
    "#fff",
    C_OUT_E,
    fs=9.6,
    weight="bold",
)
arrow(ax, (12.85, 4.55), (12.85, 4.4), color=C_OUT_E)

ax.text(
    12.85,
    3.3,
    "(separately: auto-deploy of the novel-axes\nfeature commits once the search completes)",
    ha="center",
    va="top",
    fontsize=8.2,
    style="italic",
    color="#777",
    linespacing=1.3,
)

# ── cross-section arrows ─────────────────────────────────────────────────────
arrow(ax, (3.75, 6.35), (3.0, 5.15), color=C_GRID_E, lw=2.0, rad=0.15)  # grid -> loop
ax.text(
    2.6,
    5.95,
    "each cell\nruns the loop",
    ha="center",
    fontsize=8.4,
    color=C_GRID_E,
    style="italic",
    linespacing=1.2,
)
arrow(ax, (9.3, 4.4), (10.45, 4.97), color=C_OUT_E, lw=2.0, rad=0.1)  # loop -> downstream

fig.savefig(OUT / "hp_workflow.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "hp_workflow.pdf", bbox_inches="tight")
print("wrote", OUT / "hp_workflow.png")
print("wrote", OUT / "hp_workflow.pdf")
