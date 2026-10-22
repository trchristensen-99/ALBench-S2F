"""Meeting figures: (1) log-log scaling curves, (2) reservoir x eval heatmaps.

Scaling laws are power laws in D, so both panels are plotted on log-log axes
where a power law is a straight line: MSE directly, and Pearson as the error
term (1 - r), which is what actually decays. Plain r vs log D curves bend and
hide the exponent, so the slope -- the thing we care about -- is unreadable.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "outputs/figures"
os.makedirs(FIG, exist_ok=True)
rows = list(csv.DictReader(open(f"{FIG}/eval_matrix.csv")))
rows = [r for r in rows if r["complete"] == "1"]

# ---------------- 1. scaling curves ----------------
EVALS = ["genomic", "ood", "snv_delta"]
by_arm = defaultdict(lambda: defaultdict(dict))
for r in rows:
    for e in EVALS:
        if r.get(e):
            by_arm[r["arm"]][e][int(r["D"])] = float(r[e])

fig, axes = plt.subplots(1, len(EVALS), figsize=(5.2 * len(EVALS), 4.6))
cmap = plt.get_cmap("tab20")
arms = sorted({a for a in by_arm if any(len(by_arm[a][e]) >= 2 for e in EVALS)})
colors = {a: cmap(i % 20) for i, a in enumerate(arms)}
for ax, e in zip(axes, EVALS):
    for a in arms:
        pts = sorted(by_arm[a][e].items())
        if len(pts) < 2:
            continue
        D = np.array([p[0] for p in pts], float)
        r = np.array([p[1] for p in pts], float)
        err = 1.0 - r                      # error decays as a power law; r does not
        ax.plot(D, err, "o-", color=colors[a], label=a, lw=1.6, ms=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("added sequences D  (base=0)")
    ax.set_ylabel(f"1 - Pearson r   [{e}]")
    ax.set_title(f"{e}: error vs scale (log-log)")
    ax.grid(alpha=.3, which="both", lw=.5)
axes[-1].legend(fontsize=6.5, ncol=2, loc="upper right")
fig.suptitle("Scaling: a power law is a straight line here; slope = scaling exponent", y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/scaling_curves.png", dpi=170, bbox_inches="tight")
print("  wrote scaling_curves.png   arms with >=2 D:", len(arms))

# ---------------- 2. reservoir x eval heatmaps ----------------
COLS = ["genomic", "random_32k", "dinuc_shuffle", "sub_high", "ins_high", "del_high",
        "translocation", "inversion", "ood", "snv_delta"]
for D in (100000, 30000):
    sub = [r for r in rows if int(r["D"]) == D]
    if len(sub) < 2:
        continue
    sub.sort(key=lambda r: -float(r["genomic"] or 0))
    M = np.array([[float(r[c]) if r.get(c) else np.nan for c in COLS] for r in sub])
    fig, ax = plt.subplots(figsize=(1.05 * len(COLS) + 3.4, 0.46 * len(sub) + 2.4))
    # per-column z-scores: columns sit at very different difficulty (ood ~0.3 vs
    # inversion ~0.85), so raw shading would only ever show "which eval is easy"
    Z = (M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-9)
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels([r["arm"] for r in sub], fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=6.6,
                        color="white" if abs(Z[i, j]) > 1.3 else "black")
    ax.set_xlabel("eval set  (-> test distribution)")
    ax.set_ylabel("reservoir  (-> training distribution)")
    ax.set_title(f"D={D:,} added, base=0, deploy ensemble n=5\n"
                 f"cell = Pearson r; shading = z-score WITHIN each eval column")
    fig.colorbar(im, ax=ax, shrink=.75, label="z within eval set")
    fig.tight_layout(); fig.savefig(f"{FIG}/eval_heatmap_d{D}.png", dpi=170, bbox_inches="tight")
    print(f"  wrote eval_heatmap_d{D}.png  ({len(sub)} arms)")
