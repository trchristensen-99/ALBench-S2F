"""Corrected FM schematic: ONE shared trunk that forks, and explicit centre-padding arithmetic."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.expanduser("~/Downloads/pi_meeting_aug19")
fig, ax = plt.subplots(figsize=(14.5, 7.6))
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 7.6)
ax.axis("off")


def box(x, y, w, h, t, fc, fs=9.5, weight="normal", ec="0.3"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1", fc=fc, ec=ec, lw=1.5
        )
    )
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight=weight)


def arr(x0, y0, x1, y1, c="0.35", lw=1.8):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=lw, color=c)
    )


ax.text(
    7.25,
    7.25,
    "Borzoi as STUDENT on the AlphaGenome landscape — ONE shared trunk, two heads",
    ha="center",
    fontsize=14,
    weight="bold",
)
ax.text(
    7.25,
    6.88,
    "AlphaGenome (all folds) is the ORACLE, so it cannot also be the student. "
    "Borzoi is the intermediate that tests the FM question now.",
    ha="center",
    fontsize=9,
    style="italic",
    color="0.35",
)

# ---- two inputs
box(0.15, 5.15, 2.5, 0.95, "MPRA element\n200 bp (episomal)", "#dbeafe", 9)
box(0.15, 1.35, 2.5, 0.95, "genomic window\n524,288 bp (native)", "#e2e8f0", 9)

# centre-pad detail
ax.text(1.4, 4.75, "centre-pad to 512 bp", ha="center", fontsize=8.5, color="0.25")
ax.add_patch(Rectangle((0.35, 4.25), 0.75, 0.32, fc="#f1f5f9", ec="0.55"))
ax.add_patch(Rectangle((1.10, 4.25), 0.60, 0.32, fc="#3b82f6", ec="0.3"))
ax.add_patch(Rectangle((1.70, 4.25), 0.75, 0.32, fc="#f1f5f9", ec="0.55"))
ax.text(0.72, 4.41, "156 bp\nzeros", ha="center", va="center", fontsize=6.2)
ax.text(1.40, 4.41, "200 bp", ha="center", va="center", fontsize=6.8, color="w", weight="bold")
ax.text(2.07, 4.41, "156 bp\nzeros", ha="center", va="center", fontsize=6.2)
ax.text(
    1.4,
    3.93,
    "(156 = (512-200)//2) -> all-zero one-hot is an\ninput the encoder NEVER saw in pretraining",
    ha="center",
    fontsize=7.0,
    color="#b45309",
)
ax.text(
    1.4,
    3.40,
    "alternative: pad with the REAL episomal construct",
    ha="center",
    fontsize=7.4,
    weight="bold",
    color="0.2",
)
for _x, _w, _c in (
    (0.30, 0.42, "#fca5a5"),
    (0.72, 0.52, "#fdba74"),
    (1.24, 0.52, "#3b82f6"),
    (1.76, 0.36, "#fdba74"),
    (2.12, 0.36, "#a7f3d0"),
):
    ax.add_patch(Rectangle((_x, 2.98), _w, 0.30, fc=_c, ec="0.5"))
for _xx, _tt, _col in (
    (0.51, "L-ad 15", "0.15"),
    (0.98, "promoter 36", "0.15"),
    (1.50, "element 200", "w"),
    (1.94, "R-ad 15", "0.15"),
    (2.30, "bc 15", "0.15"),
):
    ax.text(_xx, 3.13, _tt, ha="center", va="center", fontsize=5.2, color=_col)
ax.text(
    1.4,
    2.72,
    "= 281 bp real construct, then neutral fill  (--mpra_context plasmid)",
    ha="center",
    fontsize=6.6,
    color="0.3",
)
ax.text(
    1.4,
    2.34,
    "EMPIRICAL: construct padding currently COSTS MPRA\naccuracy (0.749 vs 0.812 zeros). Hypothesis: pooling\ndilution (constant bins diluted by mean-pool), under test",
    ha="center",
    fontsize=6.4,
    color="#b91c1c",
    style="italic",
)

arr(2.65, 5.62, 4.15, 4.55)
arr(2.65, 1.82, 4.15, 3.35)

# ---- the SHARED trunk (single box, both inputs converge)
box(
    4.15,
    3.30,
    3.0,
    1.35,
    "conv_dna → res_tower\nSHARED CONV TRUNK\n(trainable)",
    "#fde68a",
    10,
    weight="bold",
)
ax.text(
    5.65,
    3.05,
    "both inputs pass through the SAME trunk",
    ha="center",
    fontsize=7.8,
    color="#b45309",
    style="italic",
)

# ---- fork
arr(7.15, 4.25, 8.35, 5.35)
arr(7.15, 3.70, 8.35, 2.15)
ax.text(7.6, 4.95, "MPRA", fontsize=8, color="0.35", rotation=38)
ax.text(7.6, 2.75, "genomic", fontsize=8, color="0.35", rotation=-40)

box(8.35, 4.95, 2.6, 0.9, "pool → MPRA head", "#fae8ff", 9)
arr(10.95, 5.4, 11.9, 5.4)
box(11.9, 4.95, 2.4, 0.9, "activity (scalar)\nepisomal readout", "#dcfce7", 8.8)

box(8.35, 1.65, 2.6, 1.0, "transformer + U-Net\nFROZEN", "#cbd5e1", 9, weight="bold")
arr(10.95, 2.15, 11.9, 2.15)
box(11.9, 1.65, 2.4, 1.0, "human_head →\n7,611 tracks × 6,144 bins", "#dcfce7", 8.5)

ax.text(
    9.65,
    1.25,
    "6,144 bins × 32 bp = 196,608 bp centred in the 524,288 bp input\n"
    "= Borzoi's NATIVE output (crop.target_length = 6144)",
    ha="center",
    fontsize=7.6,
    color="0.3",
)

ax.text(
    0.15,
    0.72,
    "CL anchor path (genomic side): replay = real targets · distill = frozen-teacher targets · both\n"
    "Native step = 0.136 s / 18.5 GB — CHEAPER than one 512 bp MPRA step (0.248 s), so full context is affordable",
    fontsize=8.6,
    va="top",
    color="0.25",
)
ax.text(
    11.0,
    0.72,
    "Frozen: transformer + BatchNorm (11 layers)\nTrainable: conv trunk + MPRA head",
    fontsize=8.4,
    va="top",
    color="0.25",
)
fig.savefig(f"{OUT}/fig1_borzoi_setup.png", dpi=300, bbox_inches="tight")
print("rewrote fig1_borzoi_setup.png (shared trunk + centre-pad detail + native 6144-bin geometry)")
