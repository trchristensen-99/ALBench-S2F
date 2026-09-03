"""Figures for the Aug-19 PI update: Borzoi FM setup schematic + cross-reservoir scaling curves."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.expanduser("~/Downloads/pi_meeting_aug19")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Fig 1: FM setup schematic
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")


def box(x, y, w, h, t, fc, fs=9.5, ec="0.3", weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1", fc=fc, ec=ec, lw=1.5
        )
    )
    ax.text(
        x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight=weight, wrap=True
    )


def arr(x0, y0, x1, y1, c="0.35", lw=1.8, style="-|>"):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, lw=lw, color=c)
    )


ax.text(
    7,
    6.65,
    "Borzoi as the STUDENT on the AlphaGenome sequence-function landscape",
    ha="center",
    fontsize=14,
    weight="bold",
)
ax.text(
    7,
    6.25,
    "AlphaGenome (all folds) is our ORACLE, so it cannot also be the student — "
    "scaling Borzoi is the intermediate that tests the FM question",
    ha="center",
    fontsize=9.5,
    style="italic",
    color="0.35",
)

# MPRA path
box(0.2, 4.3, 2.3, 1.0, "MPRA element\n200 bp\n(episomal)", "#dbeafe", 9.5)
arr(2.5, 4.8, 3.2, 4.8)
box(3.2, 4.3, 2.2, 1.0, "centre-pad\n→ 512 bp", "#e0f2fe", 9)
arr(5.4, 4.8, 6.1, 4.8)
box(
    6.1,
    3.9,
    3.0,
    1.8,
    "conv_dna → res_tower\n(CONV TRUNK, shared)\ntrainable",
    "#fde68a",
    9.5,
    weight="bold",
)
arr(9.1, 4.8, 9.9, 4.8)
box(9.9, 4.3, 1.8, 1.0, "pool →\nMPRA head", "#fae8ff", 9)
arr(11.7, 4.8, 12.4, 4.8)
box(12.4, 4.3, 1.4, 1.0, "activity\n(scalar)", "#dcfce7", 9)

# genomic path
box(0.2, 1.3, 2.3, 1.0, "genomic window\n524,288 bp", "#e2e8f0", 9.5)
arr(2.5, 1.8, 6.1, 1.8)
arr(7.6, 3.9, 7.6, 3.0, c="0.5", style="<->")
ax.text(7.85, 3.4, "shared", fontsize=8.5, color="0.4", rotation=90, va="center")
box(6.1, 1.3, 3.0, 1.0, "transformer + U-Net\nFROZEN", "#cbd5e1", 9.5, weight="bold")
arr(9.1, 1.8, 9.9, 1.8)
box(9.9, 1.3, 1.8, 1.0, "human_head\n(original)", "#e2e8f0", 9)
arr(11.7, 1.8, 12.4, 1.8)
box(12.4, 1.3, 1.4, 1.0, "7,611\ntracks", "#dcfce7", 9)

ax.text(
    7.6,
    0.55,
    "CL anchor: real genomic windows at NATIVE 524 kb (0.136 s/step, 18.5 GB — cheaper than an MPRA step)\n"
    "replay = real targets · distill = frozen-teacher targets · both = real signal + teacher stabiliser",
    ha="center",
    fontsize=9,
    color="0.25",
)
ax.text(0.2, 3.15, "KEY CHOICES", fontsize=9, weight="bold", color="#b45309")
ax.text(
    0.2,
    2.75,
    "• full encoder, crop skipped\n• transformer frozen (512 bp ≈ 8 tokens)\n"
    "• BatchNorm frozen (11 layers)\n• branch point sweepable",
    fontsize=8.5,
    va="top",
    color="0.25",
)
fig.savefig(f"{OUT}/fig1_borzoi_setup.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: cross-reservoir scaling
D = np.array([3000, 10000, 30000, 100000, 300000])
res = {
    "evoaug_heavy": (
        [0.7747, 0.8197, 0.8556, 0.8910, 0.9135],
        [0.4402, 0.4498, 0.4841, 0.5490, 0.6188],
    ),
    "phylogenetic_zoonomia": (
        [0.7783, 0.8294, 0.8584, 0.8924, 0.9123],
        [0.4306, 0.4705, 0.5090, 0.5569, 0.6046],
    ),
    "motif_planted_v2": (
        [0.7534, 0.8065, 0.8492, 0.8846, 0.9068],
        [0.4868, 0.5292, 0.5831, 0.6392, 0.7152],
    ),
    "genomic": ([0.7684, 0.8082, 0.8432, 0.8755, 0.9023], [0.4180, 0.4812, 0.4893, 0.5405, 0.6049]),
    "dinuc_shuffle": (
        [0.6887, 0.7537, 0.8334, 0.8567, 0.8786],
        [0.4187, 0.4387, 0.4662, 0.5202, 0.6021],
    ),
    "random": ([0.6720, 0.7741, 0.8087, 0.8495, 0.8738], [0.4454, 0.5144, 0.5314, 0.6155, 0.6704]),
}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cols = plt.cm.tab10(np.linspace(0, 1, 10))
for i, (name, (g, o)) in enumerate(res.items()):
    for ax_, v, lab in ((axes[0], g, "in-distribution (genomic)"), (axes[1], o, "OOD")):
        ax_.plot(D, v, "o-", color=cols[i], label=name, lw=1.8, ms=5)
for ax_, ttl in zip(axes, ["In-distribution test (genomic)", "OOD test"]):
    ax_.set_xscale("log")
    ax_.set_xlabel("D (oracle-labelled training sequences)")
    ax_.set_ylabel("Pearson r")
    ax_.set_title(ttl, fontsize=11)
    ax_.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle(
    "Borzoi fine-tuned on 6 reservoirs — the ranking DEPENDS ON THE EVAL SET",
    fontsize=12.5,
    weight="bold",
)
fig.text(
    0.5,
    -0.02,
    "in-distribution: evoaug/phylo lead, random worst  |  OOD: motif_planted_v2 leads by a wide margin (0.715 vs 0.605 genomic)\n"
    "→ 'most informative data' is not a single ranking; it depends on the capability you want to unlock",
    ha="center",
    fontsize=9,
    color="0.3",
)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_reservoir_scaling.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote fig1_borzoi_setup.png, fig2_reservoir_scaling.png ->", OUT)
