"""Test/eval-set construction table (how each held-out set is built) — matches the reservoir/
acquisition IMPLEMENTED style so the PI can give feedback. -> pi_meeting_figs/test_set_construction.png
Content from scripts/build_comprehensive_test_sets.py + PROVENANCE.json (ag_s2_chrsplit_v1)."""

import os
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Downloads/pi_meeting_figs")

rows = [
    [
        "Test set",
        "Regime",
        "Construction (all from chr19/21/X held-out genomic, oracle-labeled)",
        "n",
    ],
    [
        "genomic",
        "in-distribution",
        "real K562-MPRA held-out sequences (the reference target)",
        "31,435",
    ],
    ["snv", "variant-effect", "SNV ref/alt pairs; scored on Δ = alt − ref activity", "29,383"],
    [
        "ood",
        "out-of-distribution",
        "high-activity DESIGNED sequences (off the genomic manifold)",
        "~22,000",
    ],
    [
        "dinuc_shuffle",
        "composition",
        "dinucleotide-preserving shuffle (Altschul–Erickson approx.)",
        "32,000",
    ],
    [
        "sub_low / med / high",
        "substitution",
        "5% / 20% / 50% of positions → random A/C/G/T",
        "32,000 ea",
    ],
    [
        "ins_low / med / high",
        "insertion",
        "5% / 20% / 50% random bases inserted, re-cropped to 200 bp",
        "32,000 ea",
    ],
    [
        "del_low / med / high",
        "deletion",
        "5% / 20% / 50% of positions deleted, re-padded to 200 bp",
        "32,000 ea",
    ],
    ["translocation", "structural", "one large block swap (~50% of the sequence)", "32,000"],
    ["inversion", "structural", "reverse of ~50% of the sequence", "32,000"],
    ["random_10k / 32k", "random", "uniform random sequences (null baseline)", "10k / 32k"],
    ["ctrl_neg", "control", "control-negative sequences", "—"],
]
colw = [0.16, 0.15, 0.57, 0.12]

fig = plt.figure(figsize=(13, 8.5))
fig.patch.set_facecolor("white")
fig.text(0.04, 0.955, "REFERENCE", fontsize=12, weight="bold", color="#2563eb")
fig.text(
    0.04, 0.905, "Test / eval-set battery — construction", fontsize=19, weight="bold", color="0.1"
)
fig.text(
    0.04,
    0.84,
    "All sets are chromosome-split held-out (chr19/21/X), oracle-labeled (oracle_id=full856k_clean, "
    "version ag_s2_chrsplit_v1). Substitution/insertion/deletion sweep mutation intensity 5→50%.",
    fontsize=11,
    color="0.2",
    va="top",
)
ax = fig.add_axes([0.03, 0.07, 0.94, 0.66])
ax.axis("off")
t = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", loc="center")
t.auto_set_font_size(False)
t.set_fontsize(9.4)
t.scale(1, 1.9)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("0.82")
    cell.set_width(colw[c])
    if r == 0:
        cell.set_facecolor("#1e3a8a")
        cell.set_text_props(color="white", weight="bold")
    elif rows[r][1] in ("out-of-distribution",):
        cell.set_facecolor("#fee2e2")  # OOD row highlighted (the exception in results)
    elif rows[r][1] == "structural":
        cell.set_facecolor("#ffedd5")  # structural regime (faster-scaling)
fig.text(
    0.04,
    0.035,
    "Two scaling regimes (from results): in-dist + SNV ~−0.28; OOD + structural ~−0.35 (faster). "
    "Open: which regimes to weight in the general selection objective? (see test/eval decision slide)",
    fontsize=10,
    style="italic",
    color="0.3",
    wrap=True,
)
fig.savefig(f"{OUT}/test_set_construction.png", dpi=300, bbox_inches="tight")
print("WROTE test_set_construction.png")
