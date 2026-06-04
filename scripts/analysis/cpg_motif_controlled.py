"""Rigorous test of CpG → activity relationship, controlling for motif content.

Three complementary analyses:
  1. ctrl_neg distribution vs broader genomic distribution (plot)
  2. Activity per CpG decile, BUT stratified by motif-content quintile,
     using the chr_train pool (n=315k) — disentangles CpG signal from
     motif-driven activity.
  3. Regression: K562_log2FC ~ motif_score + CpG_density (joint effects)
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/cpg_motif_controlled"

# 9 K562-relevant motifs (matches MotifPlantedV2Sampler set)
K562_MOTIFS = ["TATAAA", "CACGTG", "GGGCGG", "CCAAT", "GATA", "AGATAA", "TGACGTCA", "TGAGTCA"]


def cpg_density(seq: str) -> float:
    seq = seq.upper()
    if len(seq) <= 1:
        return 0.0
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G") / (
        len(seq) - 1
    )


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / max(1, len(seq))


def motif_score(seq: str) -> int:
    """Sum of motif hits (single-strand) across the 9 K562 motifs.
    Captures TF-binding-site content as a proxy for regulatory potential."""
    seq = seq.upper()
    return sum(seq.count(m) for m in K562_MOTIFS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading Gosai full dataset...")
    df = pd.read_csv(REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False)
    print(f"  loaded {len(df):,}")

    # Subsample for speed in CpG/motif computation (full takes ~3 min)
    df = df.dropna(subset=["K562_log2FC", "sequence"]).copy()
    print(f"  with K562_log2FC: {len(df):,}")

    print("\nComputing per-sequence features (CpG, GC, motif_score) ~2 min...")
    df["cpg"] = df["sequence"].apply(cpg_density)
    df["gc"] = df["sequence"].apply(gc_content)
    df["motif"] = df["sequence"].apply(motif_score)

    # === Analysis 1: ctrl_neg vs broader genomic distribution ===
    print("\n=== Activity distributions ===")
    ctrl_neg = df[df["class"] == "ctrl_neg"]
    ctrl_emvar = df[df["class"] == "ctrl_emvar"]
    gtex = df[df["data_project"] == "GTEX"]
    ukbb = df[df["data_project"] == "UKBB"]
    cre = df[df["data_project"] == "CRE"]

    fig, ax = plt.subplots(figsize=(12, 7))
    bins = np.linspace(-2, 8, 80)
    for label, sub, color in [
        ("ALL Gosai (n=" + f"{len(df):,})", df, "#7f7f7f"),
        ("GTEx variants (n=" + f"{len(gtex):,})", gtex, "#0072B2"),
        ("UKBB variants (n=" + f"{len(ukbb):,})", ukbb, "#56B4E9"),
        ("CRE peaks (n=" + f"{len(cre):,})", cre, "#009E73"),
        ("ctrl_neg intergenic (n=" + f"{len(ctrl_neg):,})", ctrl_neg, "#D55E00"),
        ("ctrl_emvar positives (n=" + f"{len(ctrl_emvar):,})", ctrl_emvar, "#CC79A7"),
    ]:
        y = sub["K562_log2FC"].values
        hist, edges = np.histogram(y, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(
            centers,
            hist,
            color=color,
            label=f"{label}  μ={y.mean():+.2f}  σ={y.std():.2f}",
            linewidth=2.0,
            alpha=0.85,
        )
    ax.set_xlabel("K562 log2FC (experimental)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("ctrl_neg vs broader Gosai activity distributions", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-2, 8)
    fig.tight_layout()
    fig.savefig(OUT / "ctrl_neg_vs_genomic.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "ctrl_neg_vs_genomic.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ctrl_neg_vs_genomic plot")

    # === Analysis 2: CpG vs activity stratified by motif quintile ===
    print("\n=== CpG slope by motif-content quintile (controls for motif-driven activity) ===")
    df["motif_q"] = pd.qcut(df["motif"], 5, labels=False, duplicates="drop")
    print(
        f"{'motif quintile':<18}  {'motif range':>14}  {'n':>10}  {'CpG slope':>10}  {'CpG p':>10}  {'pearson r':>10}"
    )
    cpg_slopes = []
    for q in sorted(df["motif_q"].dropna().unique()):
        sub = df[df["motif_q"] == q]
        m_lo, m_hi = sub["motif"].min(), sub["motif"].max()
        x = sub["cpg"].values
        y = sub["K562_log2FC"].values
        # slope from linear regression
        slope, intercept = np.polyfit(x, y, 1)
        r, p = pearsonr(x, y)
        cpg_slopes.append(slope)
        print(
            f"motif q={int(q):<3}        [{m_lo:>3}-{m_hi:>3}]  {len(sub):>10,}  {slope:>10.3f}  {p:>10.1e}  {r:>+10.4f}"
        )
    print(f"\n  mean CpG slope across motif quintiles: {np.mean(cpg_slopes):.3f}")

    # === Analysis 3: bivariate regression — CpG and motifs jointly ===
    print("\n=== Joint regression: K562 ~ motif_score + CpG_density ===")
    from numpy.linalg import lstsq

    X = np.column_stack([np.ones(len(df)), df["motif"].values, df["cpg"].values])
    y = df["K562_log2FC"].values
    beta, *_ = lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid_var = np.var(y - yhat)
    total_var = np.var(y)
    r2 = 1 - resid_var / total_var
    print(f"  intercept      = {beta[0]:+.4f}")
    print(f"  motif_score β  = {beta[1]:+.4f} (per additional motif hit)")
    print(f"  CpG density β  = {beta[2]:+.4f} (per unit CpG fraction)")
    print(f"  joint R²       = {r2:.4f}")

    # CpG-only regression for comparison
    X1 = np.column_stack([np.ones(len(df)), df["cpg"].values])
    b1, *_ = lstsq(X1, y, rcond=None)
    yhat1 = X1 @ b1
    r2_cpg_only = 1 - np.var(y - yhat1) / total_var
    print(f"\n  CpG-only β     = {b1[1]:+.4f}  (R²={r2_cpg_only:.4f})")

    # Motif-only regression
    X2 = np.column_stack([np.ones(len(df)), df["motif"].values])
    b2, *_ = lstsq(X2, y, rcond=None)
    yhat2 = X2 @ b2
    r2_motif_only = 1 - np.var(y - yhat2) / total_var
    print(f"  motif-only β   = {b2[1]:+.4f}  (R²={r2_motif_only:.4f})")

    print(f"\n  Variance explained:")
    print(f"    by CpG alone:        {r2_cpg_only * 100:.2f}%")
    print(f"    by motifs alone:     {r2_motif_only * 100:.2f}%")
    print(f"    by both jointly:     {r2 * 100:.2f}%")
    print(f"    incremental CpG over motifs:    {(r2 - r2_motif_only) * 100:.2f}%")
    print(f"    incremental motifs over CpG:    {(r2 - r2_cpg_only) * 100:.2f}%")

    # Save summary
    import json

    summary = {
        "ctrl_neg_distribution": {
            "n": len(ctrl_neg),
            "mean": float(ctrl_neg["K562_log2FC"].mean()),
            "std": float(ctrl_neg["K562_log2FC"].std()),
            "cpg_mean": float(ctrl_neg["cpg"].mean()),
            "cpg_range": [
                float(ctrl_neg["cpg"].quantile(0.05)),
                float(ctrl_neg["cpg"].quantile(0.95)),
            ],
        },
        "cpg_slope_by_motif_quintile": [
            {"q": int(q), "slope": float(s)}
            for q, s in zip(sorted(df["motif_q"].dropna().unique()), cpg_slopes)
        ],
        "joint_regression": {
            "intercept": float(beta[0]),
            "motif_beta": float(beta[1]),
            "cpg_beta": float(beta[2]),
            "joint_r2": float(r2),
            "cpg_only_r2": float(r2_cpg_only),
            "motif_only_r2": float(r2_motif_only),
        },
    }
    (OUT / "cpg_motif_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT}/cpg_motif_summary.json")


if __name__ == "__main__":
    main()
