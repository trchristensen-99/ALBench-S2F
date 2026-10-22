"""Figure pack for the collaborator meeting, built from outputs/scaling_v2.

Three figures:
  1. scaling curves  - ensemble Pearson vs D, one panel per eval set, coloured by
     reservoir family so the family signal is readable at a glance
  2. eval matrix     - reservoir x eval-set heatmap at a chosen D
  3. activity        - oracle-activity distribution per reservoir, which is the
     "what sequences are we actually sampling" view

Reads the CSV the curve script already writes, so figures cannot disagree with the
tables. Cells still filling are included; the per-D model cap is applied upstream.
"""
from __future__ import annotations
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Peter's presentation style: readable from across a room, not up close.
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "lines.linewidth": 2.6,
    "lines.markersize": 7,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
})

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
OUT = f"{REPO}/outputs/scaling_v2/figures"
CSV = f"{REPO}/outputs/scaling_v2/analysis/curves.csv"
PANELS = ["genomic", "ood", "motif_planted", "snv_ref"]
FAMILY = {
    "random": "synthetic", "gc_matched": "synthetic", "dinuc_shuffle": "synthetic",
    "genomic": "genomic", "zoonomia_flank_primates": "genomic",
    "zoonomia_flank_mammals": "genomic",
    "motif_ct_dense": "motif", "motif_ct_enriched": "motif", "motif_jaspar": "motif",
    "evoaug_paper2023": "evoaug", "evoaug_prm005": "evoaug", "evoaug_prm010": "evoaug",
    "d3_steps64": "model-guided", "d3_steps256": "model-guided",
}
FCOL = {"synthetic": "#8c8c8c", "genomic": "#1f6fb4", "motif": "#c4452d",
        "evoaug": "#2e8b57", "model-guided": "#7b3fa0"}


def load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r["D"] = int(r["D"]); r["ensemble_pearson"] = float(r["ensemble_pearson"])
        r["n_models"] = int(r["n_models"])
    return rows


def fig_curves(rows):
    """One panel per eval set. Arms are coloured by family and separated within a
    family by line style, with a single shared legend underneath -- inline
    end-of-line labels collided badly once there were 14 arms."""
    panels = [p for p in PANELS if any(r["eval_set"] == p for r in rows)]
    arms = sorted({r["reservoir"] for r in rows},
                  key=lambda a: (FAMILY.get(a, "z"), a))
    styles = {}
    seen = {}
    for a in arms:
        f = FAMILY.get(a, "other")
        i = seen.get(f, 0); seen[f] = i + 1
        styles[a] = ["-", "--", ":", "-."][i % 4]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 7.0), sharex=True)
    axes = np.atleast_1d(axes)
    handles = {}
    for ax, es in zip(axes, panels):
        sub = [r for r in rows if r["eval_set"] == es]
        for a in arms:
            v = {r["D"]: r["ensemble_pearson"] for r in sub if r["reservoir"] == a}
            if not v:
                continue
            Ds = sorted(v)
            ln, = ax.plot(Ds, [v[d] for d in Ds], styles[a], marker="o", ms=3.5,
                          lw=1.7, color=FCOL.get(FAMILY.get(a, "other"), "#333"),
                          alpha=0.9)
            handles.setdefault(a, ln)
        ax.set_xscale("log")
        ax.set_xlabel("training sequences D")
        ax.set_title(es, fontsize=18)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("ensemble Pearson r")
    fig.legend([handles[a] for a in arms if a in handles],
               [a for a in arms if a in handles],
               loc="lower center", ncol=4, fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Reservoir scaling curves (K562, oracle_v3, val-selected ensembles)",
                 fontsize=20)
    fig.tight_layout(rect=[0, 0.20, 1, 0.96])
    fig.savefig(f"{OUT}/scaling_curves.png", dpi=170, bbox_inches="tight")
    print("  wrote scaling_curves.png")


def fig_matrix(rows, D):
    sub = [r for r in rows if r["D"] == D]
    if not sub:
        print(f"  no rows at D={D}"); return
    res = sorted({r["reservoir"] for r in sub}, key=lambda x: (FAMILY.get(x, "z"), x))
    ev = sorted({r["eval_set"] for r in sub})
    M = np.full((len(res), len(ev)), np.nan)
    for r in sub:
        M[res.index(r["reservoir"]), ev.index(r["eval_set"])] = r["ensemble_pearson"]
    fig, ax = plt.subplots(figsize=(2 + 0.78 * len(ev), 2 + 0.62 * len(res)))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ev))); ax.set_xticklabels(ev, rotation=55, ha="right", fontsize=13)
    ax.set_yticks(range(len(res))); ax.set_yticklabels(res, fontsize=13)
    for i in range(len(res)):
        for j in range(len(ev)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=11,
                        color="w" if M[i, j] < np.nanmean(M) else "k")
    fig.colorbar(im, ax=ax, label="ensemble Pearson r", fraction=0.025)
    ax.set_title(f"reservoir (train) x eval set, D={D:,}", fontsize=18)
    fig.tight_layout()
    fig.savefig(f"{OUT}/eval_matrix_d{D}.png", dpi=170)
    print(f"  wrote eval_matrix_d{D}.png")


def fig_activity():
    P = f"{REPO}/outputs/grid_pools/k562_cumulative"
    arms = [a for a in FAMILY if os.path.exists(f"{P}/{a}_d30000.npz")]
    if not arms:
        print("  no pools for activity figure"); return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for a in sorted(arms, key=lambda x: FAMILY.get(x, "z")):
        z = np.load(f"{P}/{a}_d30000.npz", allow_pickle=True)
        y = np.asarray(z["oracle_labels"], float).ravel()
        xs = np.linspace(-3, 6, 200)
        from scipy.stats import gaussian_kde
        ax.plot(xs, gaussian_kde(y)(xs), lw=1.6, alpha=0.85,
                color=FCOL.get(FAMILY.get(a, "other"), "#333"),
                label=a if FAMILY.get(a) in ("genomic", "motif") else None)
    gen = np.load(f"{P}/genomic_d30000.npz", allow_pickle=True)["oracle_labels"]
    ax.axvline(float(np.quantile(np.asarray(gen, float), 0.90)), ls="--", c="k", lw=1,
               label="genomic 90th pct")
    ax.set_xlabel("oracle activity"); ax.set_ylabel("density")
    ax.set_title("What each reservoir actually samples (D=30k pools)", fontsize=18)
    ax.legend(fontsize=12, ncol=2, frameon=False); ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/activity_distributions.png", dpi=170)
    print("  wrote activity_distributions.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    fig_curves(rows)
    for D in (30000, 100000):
        fig_matrix(rows, D)
    fig_activity()
    print(f"\n  figures in {OUT}")
