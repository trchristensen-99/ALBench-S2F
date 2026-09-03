"""Cost-weighted scaling curves: same runs, x-axis = GPU-seconds instead of D.

Sequence count is not the currency the collaborators actually spend -- synthesis cost scales with
sequences, but MODEL cost scales with compute, and the two rank strategies differently if some
reservoirs need more data to reach the same performance. Plotting against measured GPU-seconds asks:
per unit of compute, which data is most informative?
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/Downloads/pi_meeting_aug19")
D = np.array([3000, 10000, 30000, 100000, 300000])
# (genomic r, ood r, train_sec) measured
R = {
    "evoaug_heavy": (
        [0.7747, 0.8197, 0.8556, 0.8910, 0.9135],
        [0.4402, 0.4498, 0.4841, 0.5490, 0.6188],
        [320, 372, 667, 1691, 4345],
    ),
    "phylogenetic_zoonomia": (
        [0.7783, 0.8294, 0.8584, 0.8924, 0.9123],
        [0.4306, 0.4705, 0.5090, 0.5569, 0.6046],
        [307, 376, 657, 1650, 4343],
    ),
    "motif_planted_v2": (
        [0.7534, 0.8065, 0.8492, 0.8846, 0.9068],
        [0.4868, 0.5292, 0.5831, 0.6392, 0.7152],
        [277, 369, 638, 1655, 4271],
    ),
    "genomic": (
        [0.7684, 0.8082, 0.8432, 0.8755, 0.9023],
        [0.4180, 0.4812, 0.4893, 0.5405, 0.6049],
        [284, 395, 636, 1619, 4361],
    ),
    "dinuc_shuffle": (
        [0.6887, 0.7537, 0.8334, 0.8567, 0.8786],
        [0.4187, 0.4387, 0.4662, 0.5202, 0.6021],
        [289, 371, 653, 1593, 4323],
    ),
    "random": (
        [0.6720, 0.7741, 0.8087, 0.8495, 0.8738],
        [0.4454, 0.5144, 0.5314, 0.6155, 0.6704],
        [275, 395, 685, 1602, 4353],
    ),
}
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
cols = plt.cm.tab10(np.linspace(0, 1, 10))
for i, (name, (g, o, t)) in enumerate(R.items()):
    axes[0, 0].plot(D, g, "o-", color=cols[i], label=name, lw=1.7, ms=4.5)
    axes[0, 1].plot(D, o, "o-", color=cols[i], lw=1.7, ms=4.5)
    axes[1, 0].plot(t, g, "s--", color=cols[i], lw=1.7, ms=4.5)
    axes[1, 1].plot(t, o, "s--", color=cols[i], lw=1.7, ms=4.5)
for ax, (xl, yl, ttl) in zip(
    axes.ravel(),
    [
        ("D (sequences)", "genomic r", "A. per-SEQUENCE — in-distribution"),
        ("D (sequences)", "OOD r", "B. per-SEQUENCE — OOD"),
        ("GPU-seconds (measured)", "genomic r", "C. per-COMPUTE — in-distribution"),
        ("GPU-seconds (measured)", "OOD r", "D. per-COMPUTE — OOD"),
    ],
):
    ax.set_xscale("log")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(ttl, fontsize=10.5)
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7.5, loc="lower right")
fig.suptitle(
    "Sequence-weighted vs COST-weighted scaling — synthesis budget and compute budget are different currencies",
    fontsize=12.5,
    weight="bold",
)
fig.text(
    0.5,
    -0.01,
    "Cost per cell is near-identical across reservoirs at matched D (same model, same steps), so C/D are close to a "
    "relabelled A/B here.\nThe distinction matters once strategies differ in how much data they need to reach a target — "
    "and for the 4M design, synthesis cost (per sequence) and training cost (per compute) trade off differently.",
    ha="center",
    fontsize=8.6,
    color="0.3",
)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_cost_weighted.png", dpi=300, bbox_inches="tight")
print("wrote fig3_cost_weighted.png")
