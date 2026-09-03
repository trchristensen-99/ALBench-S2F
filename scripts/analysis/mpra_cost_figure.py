"""Stacked cost bars per synthesis route, from MPRA_cost_model.xlsx.

Bottom of each stack = costs identical across routes (researcher time, assay reagents + sequencing,
equipment). Top = route-specific synthesis. Drawn this way because the shared assay floor dominates:
a cheaper synthesis route saves far less than it appears, and at 8M cell handling sets the price.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

OUT = os.path.expanduser("~/Downloads/joint_PI_meeting_aug20")
os.makedirs(OUT, exist_ok=True)

sizes = ["540K", "720K", "1M", "8M"]
N = np.array([540_000, 720_000, 1_000_000, 8_000_000])

labor = np.array([18_880, 20_790, 23_440, 70_540])
reagseq = np.array([16_198, 18_664, 19_900, 79_920])
equip = np.array([648, 747, 796, 3_197])

syn_random = np.array([300, 300, 300, 300])
syn_genscript = np.array([15_000, 17_500, 20_000, 55_000])
prm_template = np.array([15_000, 15_000, 15_000, 18_210])
prm_eppcr = np.array([2_600, 2_600, 2_600, 2_600])

routes = [
    ("Random", [("Sequence synthesis", syn_random, "#f59e0b", None)]),
    ("GenScript", [("Sequence synthesis", syn_genscript, "#f59e0b", None)]),
    (
        "EP-PCR",
        [
            ("Sequence synthesis", prm_template, "#f59e0b", None),
            ("EP-PCR reagents + labour", prm_eppcr, "#b45309", "//"),
        ],
    ),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={"width_ratios": [2.1, 1]})
ax = axes[0]
w, gap = 0.24, 0.05
xs = np.arange(len(sizes)) * 1.15
positions, ticklabels = [], []

for i, (label, syn_parts) in enumerate(routes):
    pos = xs + (i - 1) * (w + gap)
    positions += list(pos)
    ticklabels += [label] * len(pos)
    bottom = np.zeros(len(sizes))
    for seg, vals, col in (
        ("Researcher time", labor, "#1e3a8a"),
        ("MPRA reagents + sequencing", reagseq, "#3b82f6"),
        ("Equipment", equip, "#93c5fd"),
    ):
        ax.bar(
            pos,
            vals,
            w,
            bottom=bottom,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            label=seg if i == 0 else None,
        )
        bottom = bottom + vals
    for nm, vals, col, hatch in syn_parts:
        show = (i == 0 and nm == "Sequence synthesis") or (nm.startswith("EP-PCR"))
        ax.bar(
            pos,
            vals,
            w,
            bottom=bottom,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
            label=nm if show else None,
        )
        bottom = bottom + vals
    for x, tot in zip(pos, bottom):
        ax.text(x, tot + 4_000, f"${tot / 1000:.0f}k", ha="center", fontsize=7.6, weight="bold")

ax.set_xticks(positions)
# angled so adjacent route labels do not crowd each other; anchored at the right edge so each label
# sits under its own bar rather than drifting left
ax.set_xticklabels(ticklabels, fontsize=8.0, rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", length=0, pad=1)

# group labels on a second row, in axes coords so they never collide with the tick labels
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for x, s in zip(xs, sizes):
    ax.text(
        x,
        -0.155,
        f"{s} sequences",
        transform=trans,
        ha="center",
        va="top",
        fontsize=10,
        weight="bold",
    )
    ax.axvline(x + (1.5 * w + gap), color="0.85", lw=0.8, zorder=0)

ax.set_ylim(0, 245_000)
ax.set_yticks(np.arange(0, 245_001, 50_000))
ax.set_yticklabels([f"${v // 1000}k" for v in np.arange(0, 245_001, 50_000)])
ax.set_ylabel("Total cost (USD)")
ax.set_title(
    "Fixed assay costs dominate; synthesis route is the smaller lever",
    fontsize=11.5,
    weight="bold",
    pad=10,
)
ax.legend(fontsize=8.2, loc="upper left", framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

# ---- panel 2: all-in cost per sequence
ax2 = axes[1]
floor = labor + reagseq + equip
for (label, syn_parts), col, mk in zip(routes, ["#16a34a", "#dc2626", "#7c3aed"], ["o", "s", "^"]):
    syn = sum(v for _, v, _, _ in syn_parts)
    ax2.plot(N, (floor + syn) / N, mk + "-", color=col, lw=1.9, ms=6.5, label=label)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Library size (sequences)")
ax2.set_ylabel("$ per sequence (all-in)")
ax2.set_title("All-in cost per sequence", fontsize=11.5, weight="bold", pad=10)
ax2.grid(alpha=0.3, which="both")
ax2.set_axisbelow(True)
ax2.legend(fontsize=8.5)

fig.suptitle(
    "MPRA data-collection cost by synthesis route (Gosai / Tewhey protocol basis)",
    fontsize=13.5,
    weight="bold",
)
fig.subplots_adjust(bottom=0.21, top=0.88)
fig.savefig(f"{OUT}/fig_mpra_costs.png", dpi=300, bbox_inches="tight")
print("wrote fig_mpra_costs.png")
