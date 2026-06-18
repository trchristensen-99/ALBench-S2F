"""Deploy-ensemble construction figure (single-val regime).

Provisional numbers from the live D=30k genomic bake-off sketches (Jun 18 2026).
Panel A: deploy-faithful oracle-r vs ensemble size N* (single val set, mean of 3
seed deployments) — vs the inflated seed-averaged-predictions curve and the best
single model. Panel B: cross-seed agreement of the top-weighted configs per HP
axis (descriptive: where the search converges vs stays dataset-specific).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/Users/christen/Downloads/pi_update_figures"
BLUE, RED, GREEN, ORANGE, GREY = "#4C72B0", "#C44E52", "#55A868", "#DD8452", "#7f7f7f"

# ---- Panel A data (single-val deploy, mean-of-metrics over 3 seeds) ----
N = np.array([3, 5, 8, 12, 20])  # 20 ~ "all-active"
mean_sv = np.array([0.7565, 0.7633, 0.7658, 0.7660, 0.7666])
sd_sv = np.array([0.0130, 0.0107, 0.0090, 0.0094, 0.0097])
seed_sv = {
    "seed42": [0.7569, 0.7651, 0.7674, 0.7693, 0.7706],
    "seed43": [0.7722, 0.7755, 0.7759, 0.7755, 0.7759],
    "seed44": [0.7404, 0.7494, 0.7540, 0.7532, 0.7532],
}
seedavg_pred = np.array([0.7903, 0.7894, 0.7885, 0.7874, 0.7865])  # NOT deployable
single_best = 0.7308

# ---- Panel B data (cross-seed spread of top-8-by-weight configs) ----
# numeric axes: normalized spread (max-min)/|mean| across per-seed medians
axes_num = [
    ("n_layers", 0.14),
    ("pct_start", 0.30),
    ("ks", 0.35),
    ("width_base", 0.60),
    ("batch_size", 0.60),
    ("lr", 1.08),
    ("conv_dropout", 1.56),
    ("dense_dropout", 1.66),
    ("weight_decay", 1.73),
]
# categorical axes: agree -> small proxy, disagree -> large proxy
axes_cat = [
    ("block_class", 0.10, True),
    ("use_shift_aug", 0.10, True),
    ("use_evoaug", 0.18, True),
    ("optimizer", 1.30, False),
    ("lr_schedule", 1.45, False),
]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 5.4))

# ===== Panel A =====
for s, ys in seed_sv.items():
    axA.plot(N, ys, color=GREY, lw=1.0, alpha=0.5, marker="o", ms=3, zorder=2)
axA.annotate(
    "per-seed deployments",
    (20, seed_sv["seed44"][-1]),
    xytext=(6, -14),
    textcoords="offset points",
    fontsize=8,
    color=GREY,
)
axA.errorbar(
    N,
    mean_sv,
    yerr=sd_sv,
    color=BLUE,
    lw=2.6,
    marker="o",
    ms=8,
    capsize=4,
    mfc="white",
    mec=BLUE,
    mew=2,
    zorder=4,
    label="single-val deploy (mean ± SD, 3 seeds)",
)
axA.plot(
    N,
    seedavg_pred,
    color=GREY,
    ls="--",
    lw=2.0,
    marker="s",
    ms=5,
    zorder=3,
    label="seed-averaged predictions (NOT deployable)",
)
axA.axhline(
    single_best, color=RED, ls=":", lw=2.0, zorder=1, label=f"best single model ({single_best:.3f})"
)

# knee marker at N*=8
axA.axvline(8, color=GREEN, ls="-", lw=1.0, alpha=0.4)
axA.annotate("knee  N*≈8", (8, 0.748), color=GREEN, fontsize=10, ha="center")
axA.annotate(
    "",
    xy=(8, mean_sv[2]),
    xytext=(8, single_best),
    arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3),
)
axA.annotate(
    "+0.035\nvs single", (8.3, (mean_sv[2] + single_best) / 2), color=RED, fontsize=9, va="center"
)
axA.annotate(
    "seed-avg inflates\n~+0.02 (3× hidden ensemble)",
    (12, 0.7885),
    color=GREY,
    fontsize=8.5,
    ha="center",
)

axA.set_xlabel("deploy ensemble size  N*  (top members by stack weight)")
axA.set_ylabel("oracle correlation (genomic test)")
axA.set_title(
    "Deploy is single-val: ~8 members, +0.035 over best single\n"
    "(D=30k genomic bake-off, provisional)",
    fontsize=11.5,
)
axA.set_xticks(N)
axA.set_xticklabels([3, 5, 8, 12, "all\n(~20)"])
axA.set_ylim(0.725, 0.795)
axA.grid(True, axis="y", ls=":", alpha=0.5)
axA.legend(fontsize=8.4, loc="lower right")

# ===== Panel B =====
rows = [(n, v, None) for n, v in axes_num] + [(n, v, a) for n, v, a in axes_cat]
rows.sort(key=lambda r: r[1])
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
colors = [GREEN if v < 0.5 else (ORANGE if v < 1.0 else RED) for v in vals]
y = np.arange(len(rows))
bars = axB.barh(y, vals, color=colors, alpha=0.85)
for i, (n, v, cat) in enumerate(rows):
    if cat is not None:  # categorical axis: hatch
        bars[i].set_hatch("//")
        bars[i].set_edgecolor("white")
axB.axvline(0.5, color=GREY, ls="--", lw=1.0)
axB.set_yticks(y)
axB.set_yticklabels(labels, fontsize=9)
axB.set_xlabel("cross-seed spread of top-weighted configs  (low = converges)")
axB.set_title(
    "Where the search converges (backbone) vs\n"
    "stays dataset-specific (regularization) — descriptive, not a constraint",
    fontsize=11,
)
axB.annotate("converges\n(backbone)", (0.18, 0.5), color=GREEN, fontsize=9, ha="center")
axB.annotate(
    "dataset-specific\n(let ElasticNet weight)",
    (1.35, len(rows) - 1.6),
    color=RED,
    fontsize=9,
    ha="center",
)
axB.text(
    0.97,
    -0.13,
    "hatched = categorical axis",
    transform=axB.transAxes,
    fontsize=7.5,
    color=GREY,
    ha="right",
)
axB.set_xlim(0, 1.9)
axB.grid(True, axis="x", ls=":", alpha=0.4)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/pi_deploy_ensemble_construction.{ext}", dpi=160, bbox_inches="tight")
print(f"wrote {OUT}/pi_deploy_ensemble_construction.png")
