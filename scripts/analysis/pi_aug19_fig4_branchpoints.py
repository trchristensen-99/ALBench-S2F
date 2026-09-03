"""Detailed Borzoi architecture with the candidate MPRA branch points, measured at 512 bp input.

Two things this figure exists to make visible:
 (1) the transformer sees only FOUR tokens at 512 bp (128x downsampling), so "the transformer helps
     MPRA" is very unlikely to be about attention -- the deeper branches also bring ~150M more
     parameters of convolution and a skip-refined representation.
 (2) the U-Net SKIPS are taken from the TRUNK (res_tower / unet1 outputs). So unfreezing the trunk
     changes what the frozen transformer path receives: freezing transformer WEIGHTS does not freeze
     the genomic pathway's BEHAVIOUR.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.expanduser("~/Downloads/pi_meeting_aug19")
fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.2)
ax.axis("off")


def box(x, y, w, h, t, fc, fs=8.5, weight="normal", ec="0.35"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec=ec, lw=1.4
        )
    )
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight=weight)


def arr(x0, y0, x1, y1, c="0.4", lw=1.6, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=lw, color=c, linestyle=ls
        )
    )


ax.text(
    7.5,
    7.85,
    "Borzoi architecture and MPRA branch points  (shapes measured at 512 bp input)",
    ha="center",
    fontsize=13.5,
    weight="bold",
)

# main encoder chain
chain = [
    (0.3, "input\n(4, 512)", "#f8fafc", "—"),
    (2.1, "conv_dna\n(512, 256)  2x", "#dbeafe", "31 K"),
    (4.0, "res_tower\n(1280, 16)  32x", "#fde68a", "18.6 M"),
    (6.1, "unet1\n(1536, 8)  64x", "#fed7aa", "9.8 M"),
    (8.2, "max_pool\n(1536, 4)  128x", "#fecaca", "—"),
    (10.3, "transformer\n(1536, 4)", "#e9d5ff", "126 M"),
]
for x, t, c, pr in chain:
    box(x, 5.4, 1.6, 1.0, t, c, 8.2)
    ax.text(x + 0.8, 5.15, pr, ha="center", fontsize=7, color="0.4")
for i in range(len(chain) - 1):
    arr(chain[i][0] + 1.6, 5.9, chain[i + 1][0], 5.9)

# upsampling path back down
box(10.3, 3.4, 1.6, 0.9, "upsample1\n(1536, 8)", "#e9d5ff", 8)
box(8.2, 3.4, 1.6, 0.9, "separable1\n(1536, 8)", "#e9d5ff", 8)
box(6.1, 3.4, 1.6, 0.9, "upsample0\n(1536, 16)", "#e9d5ff", 8)
box(4.0, 3.4, 1.6, 0.9, "separable0\n(1536, 16)", "#ddd6fe", 8.2, weight="bold")
arr(11.1, 5.4, 11.1, 4.3)
arr(10.3, 3.85, 9.8, 3.85)
arr(8.2, 3.85, 7.7, 3.85)
arr(6.1, 3.85, 5.6, 3.85)
box(2.0, 3.4, 1.6, 0.9, "human_head\n7611 x 6144", "#dcfce7", 8)
arr(4.0, 3.85, 3.6, 3.85)

# SKIP connections from the trunk
ax.add_patch(
    FancyArrowPatch(
        (6.9, 5.4),
        (9.0, 4.3),
        arrowstyle="-|>",
        mutation_scale=12,
        lw=2.0,
        color="#dc2626",
        linestyle="--",
        connectionstyle="arc3,rad=-0.3",
    )
)
ax.add_patch(
    FancyArrowPatch(
        (4.8, 5.4),
        (4.8, 4.3),
        arrowstyle="-|>",
        mutation_scale=12,
        lw=2.0,
        color="#dc2626",
        linestyle="--",
    )
)
ax.text(7.6, 4.72, "skip h1 (from unet1)", fontsize=7.4, color="#dc2626", rotation=-22)
ax.text(4.95, 4.75, "skip h0 (from res_tower)", fontsize=7.4, color="#dc2626")

# branch points
for x, lab, r in (
    (2.1, "branch=conv", "0.373"),
    (4.0, "branch=res_tower", "0.809"),
    (6.1, "branch=unet1", "0.849"),
    (4.0, "branch=full", "0.864"),
):
    y = 6.75 if lab != "branch=full" else 2.85
    ax.text(
        x + 0.8,
        y,
        f"▲ {lab}\nMPRA r={r}",
        ha="center",
        fontsize=7.6,
        weight="bold",
        color="#1d4ed8" if lab != "branch=full" else "#047857",
    )

# AG comparison
box(
    12.4,
    5.4,
    2.3,
    1.0,
    "AlphaGenome MPRA head\nreads encoder at\n128 bp / position",
    "#fce7f3",
    7.8,
)
ax.text(13.55, 5.15, "≈ Borzoi's max_pool stage (128x)", ha="center", fontsize=7, color="#9d174d")
ax.text(
    13.55, 4.75, "our branch=full is FINER\n(32 bp/bin)", ha="center", fontsize=7, color="#9d174d"
)

ax.text(
    0.3,
    2.35,
    "WHY 'the transformer helps' is probably NOT about attention:\n"
    "• at 512 bp the transformer sees only 4 tokens (128x downsampling) — almost no long-range interaction to model\n"
    "• deeper branches also add ~150 M parameters of convolution and a skip-REFINED representation at the same 32 bp resolution\n"
    "• branch=res_tower (1280 ch) vs branch=full (1536 ch) also differ in width, so depth and capacity are confounded",
    fontsize=8.2,
    va="top",
    color="0.2",
)
ax.text(
    0.3,
    1.05,
    "CONSEQUENCE OF THE SKIPS (red): h0/h1 are taken from the TRUNK. If the trunk is trainable, the frozen transformer path\n"
    "receives SHIFTED INPUTS — so freezing transformer WEIGHTS does not freeze the genomic pathway's BEHAVIOUR.\n"
    "This is why preservation is < 1.0 even with the transformer frozen, and why 'structural preservation' was overstated.",
    fontsize=8.2,
    va="top",
    color="#b91c1c",
)
fig.savefig(f"{OUT}/fig4_branch_points.png", dpi=300, bbox_inches="tight")
print("wrote fig4_branch_points.png")
