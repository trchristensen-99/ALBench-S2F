"""'Next PDB' framing schematic: the training CORPUS is the deliverable; informativeness is one
input; students = CNN + foundation models; eval = applications. -> pi_meeting_figs/"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.expanduser("~/Downloads/pi_meeting_figs")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(figsize=(13.5, 6.2))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 6.2)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=10.5, weight="bold", ec="0.25"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec=ec, lw=1.6
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, wrap=True
    )


def arrow(x0, y0, x1, y1, color="0.3", lw=2.2):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=lw, color=color)
    )


# construction (left)
box(0.2, 4.3, 2.5, 1.1, "Reservoir strategies\nGENERATE candidates", "#dbeafe", 10)
box(0.2, 2.9, 2.5, 1.1, "Acquisition\nSELECT informative subset", "#dcfce7", 10)
# the corpus (center — the deliverable)
box(3.4, 3.2, 2.7, 2.0, "THE CORPUS\n= the 'next PDB'\n(training dataset)", "#fde68a", 13)
ax.text(4.75, 5.35, "the DELIVERABLE", ha="center", fontsize=10.5, weight="bold", color="#b45309")
arrow(2.7, 4.8, 3.4, 4.4)
arrow(2.7, 3.45, 3.4, 3.9)
# oracle labels the corpus
box(3.4, 1.4, 2.7, 1.0, "Oracle = AlphaGenome all-folds\n(labels the corpus)", "#e2e8f0", 9.5)
arrow(4.75, 2.4, 4.75, 3.2, color="0.55")
# students (train on corpus)
box(6.8, 4.1, 2.9, 1.3, "Train STUDENTS\n• from-scratch CNN\n• foundation models", "#fae8ff", 10)
box(
    6.8,
    2.5,
    2.9,
    1.3,
    "Foundation models\nBorzoi/Flashzoi (+ NTv3)\nfull fine-tune ± CL-replay",
    "#f3e8ff",
    9.3,
)
arrow(6.1, 4.4, 6.8, 4.6)
arrow(6.1, 3.9, 6.8, 3.2)
# applications (right)
box(
    10.3,
    3.1,
    3.0,
    2.1,
    "Evaluate across\nAPPLICATIONS\n• variant interp. (eQTL)\n• sequence design\n• regulatory grammar",
    "#dcfce7",
    10,
)
arrow(9.7, 4.6, 10.3, 4.4)
arrow(9.7, 3.1, 10.3, 3.6)

ax.text(
    6.75,
    5.9,
    "Building the next PDB: which training data unlocks sequence-function model capabilities?",
    ha="center",
    fontsize=14,
    weight="bold",
    color="0.1",
)
ax.text(
    6.75,
    0.55,
    "Central question: what data is most informative to unlock model CAPABILITIES (not just fit one assay). "
    "Single strategy may not hold a power law → MIX sources + scale; eventually multiple cell types + perturbations.",
    ha="center",
    fontsize=9.5,
    style="italic",
    color="0.3",
    wrap=True,
)
fig.tight_layout()
fig.savefig(f"{OUT}/schematic_nextpdb.png", dpi=300, bbox_inches="tight")
print("WROTE schematic_nextpdb.png")
