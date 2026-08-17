"""PI-meeting schematics: (1) pipeline flow, (2) informativeness mockup.

Fig 2 uses the REAL slope-variance fits (Aug 11): reservoir differences are a LEVEL
effect (common slope ~-0.275, different intercepts); eval-TYPE sets the RATE
(in-distribution ~-0.28 vs OOD/structural ~-0.35). Measured D range 1e4-1e5 is solid;
extrapolation to 1e6 is dashed and labeled.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT = os.path.expanduser("~/Downloads/pi_meeting_figs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Fig 1: pipeline
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")

def box(x, y, w, h, text, fc, ec="0.25", fs=11, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight=weight, wrap=True)

def arrow(x0, y0, x1, y1, color="0.3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                 lw=2.0, color=color))

# three-axis core row
box(0.2, 3.4, 2.2, 1.1, "Reservoir\nGENERATE\nsequences", "#dbeafe")
box(2.9, 3.4, 2.2, 1.1, "Acquisition\nSELECT\ninformative subset", "#dcfce7")
box(5.6, 3.4, 2.2, 1.1, "HP search\nPROPOSE\nconfigs", "#fef9c3")
box(8.3, 3.4, 2.2, 1.1, "Train student\nLegNet CNNs", "#fae8ff")
box(11.0, 3.4, 1.8, 1.1, "Val-selected\nElasticNet\nENSEMBLE", "#ffe4e6")
for x0, x1 in [(2.4, 2.9), (5.1, 5.6), (7.8, 8.3), (10.5, 11.0)]:
    arrow(x0, 3.95, x1, 3.95)

# oracle (top, feeds labels)
box(3.5, 1.0, 3.0, 0.95, "AlphaGenome-S2 ORACLE\n(provides labels, r≈0.97)", "#e2e8f0", fs=11)
for tx in [1.3, 4.0, 6.7]:
    arrow(5.0, 1.95, tx, 3.4, color="0.55")
# eval battery (bottom right, oracle-labeled)
box(9.2, 0.9, 3.6, 1.15, "EVAL BATTERY (oracle-labeled)\ngenomic · SNV · OOD · structural",
    "#e2e8f0", fs=10.5)
arrow(11.9, 3.4, 11.5, 2.05, color="0.55")
arrow(6.5, 1.5, 9.2, 1.5, color="0.55")

ax.text(0.2, 4.75, "Three independent axes", fontsize=12.5, weight="bold", color="#1e3a8a")
ax.text(6.5, 0.35,
        "Deliverable: a fixed, reweightable ENSEMBLE MENU that reaches ~best-possible on any reservoir×acquisition combo",
        ha="center", fontsize=10.5, style="italic", color="0.25")
ax.set_title("Pipeline: oracle-distilled sequence-to-function models", fontsize=15, weight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/schematic_pipeline.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: informativeness
logD = np.linspace(3, 6, 100)          # D = 1e3 .. 1e6
Dmeas = (logD >= 4) & (logD <= 5)      # measured range solid; else dashed

def line(ax, slope, intercept, color, label, lw=2.4):
    y = intercept + slope * logD
    ax.plot(logD[Dmeas], y[Dmeas], color=color, lw=lw, label=label)
    ax.plot(logD[~Dmeas & (logD < 4.01)], y[~Dmeas & (logD < 4.01)], color=color, lw=lw, ls=":")
    ax.plot(logD[~Dmeas & (logD > 4.99)], y[~Dmeas & (logD > 4.99)], color=color, lw=lw, ls=":")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))

# Panel A: reservoir = LEVEL (common slope, real intercepts)
S = -0.275
res = [("evoaug", 2.48, "#b91c1c"), ("genomic", 2.32, "#1d4ed8"),
       ("dinuc", 2.22, "#0891b2"), ("gc_matched", 2.02, "#16a34a"),
       ("motif", 1.93, "#9333ea")]
for name, b, c in res:
    line(a1, S, b, c, f"{name}")
a1.axvspan(4, 5, color="0.9", zorder=0)
a1.text(4.5, a1.get_ylim()[1] if False else 1.55, "measured", ha="center", fontsize=9, color="0.4")
a1.set_title("Sequence SOURCE → LEVEL (offset)\ncommon slope ≈ −0.275 (indistinguishable), different intercepts",
             fontsize=12, weight="bold")
a1.set_xlabel("log₁₀ D  (number of measured sequences)", fontsize=12)
a1.set_ylabel("log₁₀ test-MSE  (genomic eval)", fontsize=12)
a1.legend(fontsize=10, framealpha=0.9, loc="upper right")
a1.annotate("horizontal gap =\n'effective-sample multiplier'\n(each evoaug seq worth k genomic)",
            xy=(4.3, 2.48 + S * 4.3), xytext=(3.15, 0.55),
            fontsize=9.5, color="0.2",
            arrowprops=dict(arrowstyle="->", color="0.4"))

# Panel B: eval TYPE = RATE (real slopes)
evals = [("genomic (in-dist)", -0.284, 2.1, "#1d4ed8"),
         ("SNV", -0.286, 2.15, "#0891b2"),
         ("structural variants", -0.356, 4.6, "#ea580c"),
         ("OOD", -0.349, 5.2, "#b91c1c")]
for name, s, b, c in evals:
    line(a2, s, b, c, f"{name}  (m={s:.2f})")
a2.axvspan(4, 5, color="0.9", zorder=0)
a2.set_title("Eval TARGET type → RATE (slope)\nin-dist/SNV ≈ −0.28  vs  OOD/structural ≈ −0.35 (significant)",
             fontsize=12, weight="bold")
a2.set_xlabel("log₁₀ D  (number of measured sequences)", fontsize=12)
a2.set_ylabel("log₁₀ test-MSE", fontsize=12)
a2.legend(fontsize=10, framealpha=0.9, loc="upper right")
a2.annotate("steeper → more measured\nsequences pay off faster\non OOD/structural targets",
            xy=(4.8, -0.349 * 4.8 + 5.2), xytext=(3.1, 1.2),
            fontsize=9.5, color="0.2", arrowprops=dict(arrowstyle="->", color="0.4"))

fig.suptitle("Sequence informativeness = which sequences to measure, per eval-target",
             fontsize=15, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/schematic_informativeness.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("WROTE", f"{OUT}/schematic_pipeline.png", "and", f"{OUT}/schematic_informativeness.png")
