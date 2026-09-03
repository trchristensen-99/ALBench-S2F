"""Ensembling protocol + assumptions + supporting evidence — one-page slide."""

import os
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Downloads/pi_meeting_figs")
os.makedirs(OUT, exist_ok=True)

fig = plt.figure(figsize=(13.5, 8.5))
fig.patch.set_facecolor("white")
fig.text(0.04, 0.955, "METHOD", fontsize=12, weight="bold", color="#2563eb")
fig.text(
    0.04,
    0.915,
    "Ensembling protocol — steps, assumptions & evidence",
    fontsize=20,
    weight="bold",
    color="0.1",
)

rows = [
    ["Step", "What we do", "Key assumption", "Supporting evidence"],
    [
        "0. Oracle labels",
        "AG-S2 labels the\nreservoir/acquisition seqs",
        "Oracle ≈ real measurement",
        "r = 0.97 in-dist / 0.98 OOD /\n0.81 SNV-delta",
    ],
    [
        "1. HP search",
        "GP/TPE/evo + LLM propose\nconfigs; train CNNs (ES)",
        "~50 rounds enough;\n100-ep cap rarely binds",
        "best-so-far plateaus ~50 rounds;\n~98% early-stop by ~epoch 30",
    ],
    [
        "2. Pool cap",
        "Keep top-40 by VAL Pearson",
        "Capping is free",
        "flat: top-5 0.865 = top-40 =\ntop-160 0.863 (uncapped not better)",
    ],
    [
        "3. Greedy ElasticNet\n   ensemble",
        "Val-selected greedy add;\nElasticNetCV(+) refit each step",
        "Reaches ~best single;\nknee is small; low regret",
        "knee 4-5; ensemble >= hindsight-best\nsingle +0.02-0.04 (97-100%); OOD -0.01",
    ],
    [
        "4. Strategy pooling",
        "Pool models across ALL\nHP strategies (not one)",
        "A few strategies suffice",
        "greedy-forward knee ~4 strategies\n(strategy_marginal figure)",
    ],
    [
        "5. Multi-seed (opt.)",
        "Add a 2nd data/init seed",
        "Worthwhile but not uniform",
        "+0.006 avg; genomic/evoaug +0.01-0.016,\nmotif/gc ~0",
    ],
    [
        "6. Deploy menu",
        "Fixed reservoir-agnostic menu;\nreweight per deployment",
        "One menu ~ each combo's best;\nweights adapt, not the menu",
        "reservoir-LOSO 6/7 near-best;\nacq-LOSO uncertainty -0.010, diversity -0.003",
    ],
]
ax = fig.add_axes([0.03, 0.06, 0.94, 0.78])
ax.axis("off")
tbl = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.2)
tbl.scale(1, 2.5)
colw = [0.15, 0.24, 0.24, 0.37]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("0.82")
    cell.set_width(colw[c])
    if r == 0:
        cell.set_facecolor("#1e3a8a")
        cell.set_text_props(color="white", weight="bold")
    elif r in (3, 4):  # highlight the two ensemble steps
        cell.set_facecolor("#eef4ff")

fig.text(
    0.04,
    0.035,
    "Open item (for PI): the val set must match the target distribution — a single genomic val is honest for "
    "genomic/SNV/structural but not OOD (regret -0.01; motif val inflated +0.14, chr-split doesn't fix it). "
    "Recommended: target-matched val.",
    fontsize=9.5,
    style="italic",
    color="0.3",
    wrap=True,
)
fig.savefig(f"{OUT}/protocol_evidence.png", dpi=300, bbox_inches="tight")
print("WROTE", f"{OUT}/protocol_evidence.png")
