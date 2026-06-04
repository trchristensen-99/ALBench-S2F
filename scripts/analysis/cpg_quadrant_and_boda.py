"""Causal disambiguation tests for CpG vs motif-driven activity.

1. Quadrant analysis: split real Gosai sequences by HIGH/LOW CpG × HIGH/LOW motif.
   If sequences with HIGH CpG and LOW motif have HIGH activity → CpG drives activity
   If sequences with LOW CpG and HIGH motif have HIGH activity → motifs drive activity
2. BODA-designed inflation test: compare CpG distributions of natural Gosai (n=798k)
   vs OOD-designed BODA sequences (n=22962) — is CpG content much higher in BODA?
3. CpG slope by source category: GTEx vs UKBB vs CRE vs BODA-designed
4. CpG-rich, motif-poor outlier analysis: what activity do these have?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/cpg_quadrant_and_boda"

K562_MOTIFS = ["TATAAA", "CACGTG", "GGGCGG", "CCAAT", "GATA", "AGATAA", "TGACGTCA", "TGAGTCA"]


def cpg_density(seq: str) -> float:
    seq = seq.upper()
    if len(seq) <= 1:
        return 0.0
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G") / (
        len(seq) - 1
    )


def motif_score(seq: str) -> int:
    seq = seq.upper()
    return sum(seq.count(m) for m in K562_MOTIFS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading Gosai natural data (n=798k)...")
    nat = pd.read_csv(
        REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False
    )
    nat = nat.dropna(subset=["K562_log2FC", "sequence"]).copy()
    print(f"  loaded {len(nat):,}")

    print("Loading BODA-designed OOD (n≈23k)...")
    ood = pd.read_csv(REPO / "data/k562/test_sets/test_ood_designed_k562.tsv", sep="\t")
    ood = ood.dropna(subset=["K562_log2FC", "sequence"]).copy()
    print(f"  loaded {len(ood):,}")

    print("\nComputing CpG + motif features for both datasets...")
    for df in [nat, ood]:
        df["cpg"] = df["sequence"].apply(cpg_density)
        df["motif"] = df["sequence"].apply(motif_score)

    # === 1. Quadrant analysis (natural data only — REAL labels) ===
    print("\n=== 1. Quadrant analysis: CpG × motif content (natural Gosai 798k) ===")
    cpg_med = nat["cpg"].median()
    motif_med = nat["motif"].median()
    print(f"  CpG median: {cpg_med:.4f}    motif median: {motif_med}")

    nat["cpg_bin"] = (nat["cpg"] >= cpg_med).map({True: "HI", False: "LO"})
    nat["motif_bin"] = (nat["motif"] >= motif_med).map({True: "HI", False: "LO"})

    print(
        f"\n{'quadrant':<22}  {'n':>10}  {'K562 mean':>10}  {'K562 std':>9}  {'CpG mean':>9}  {'motif mean':>10}"
    )
    for cpg_v in ["LO", "HI"]:
        for mot_v in ["LO", "HI"]:
            sub = nat[(nat["cpg_bin"] == cpg_v) & (nat["motif_bin"] == mot_v)]
            print(
                f"  CpG={cpg_v}, motif={mot_v:<5}  {len(sub):>10,}  {sub['K562_log2FC'].mean():>+10.3f}  "
                f"{sub['K562_log2FC'].std():>9.3f}  {sub['cpg'].mean():>9.4f}  {sub['motif'].mean():>10.2f}"
            )

    # === 2. BODA inflation test ===
    print("\n=== 2. BODA-designed OOD vs natural Gosai: CpG distribution ===")
    print(
        f"{'set':<28}  {'n':>10}  {'CpG mean':>9}  {'CpG std':>8}  {'q05':>7}  {'q50':>7}  {'q95':>7}"
    )
    for label, src in [
        ("Natural Gosai (798k)", nat),
        ("BODA-designed OOD (23k)", ood),
        ("Gosai ctrl_neg (intergenic)", nat[nat["class"] == "ctrl_neg"]),
        ("Gosai CRE peaks", nat[nat["data_project"] == "CRE"]),
    ]:
        c = src["cpg"]
        print(
            f"  {label:<28}  {len(src):>10,}  {c.mean():>9.4f}  {c.std():>8.4f}  "
            f"{c.quantile(0.05):>7.4f}  {c.quantile(0.50):>7.4f}  {c.quantile(0.95):>7.4f}"
        )

    # Also report activity by category
    print("\n=== K562 activity by source category ===")
    for label, src in [
        ("Natural Gosai (798k)", nat),
        ("BODA-designed OOD (23k)", ood),
        ("Gosai ctrl_neg", nat[nat["class"] == "ctrl_neg"]),
        ("Gosai CRE peaks", nat[nat["data_project"] == "CRE"]),
    ]:
        v = src["K562_log2FC"]
        c = src["cpg"]
        # Slope via linregress within this set
        r, _ = pearsonr(c.values, v.values) if len(src) > 5 else (np.nan, np.nan)
        slope, intercept = np.polyfit(c, v, 1) if len(src) > 5 else (np.nan, np.nan)
        print(
            f"  {label:<28}  n={len(src):>10,}  K562 μ={v.mean():>+7.3f}  σ={v.std():>6.3f}  "
            f"CpG-slope={slope:>+8.3f}  pearson_r={r:>+7.4f}"
        )

    # === 3. CpG-rich + motif-poor outliers ===
    print("\n=== 3. CpG-rich + motif-poor outliers (top 1% CpG, bottom 25% motif) ===")
    cpg_q99 = nat["cpg"].quantile(0.99)
    motif_q25 = nat["motif"].quantile(0.25)
    outliers = nat[(nat["cpg"] >= cpg_q99) & (nat["motif"] <= motif_q25)]
    print(f"  n outliers = {len(outliers):,}")
    print(
        f"  outlier CpG mean = {outliers['cpg'].mean():.4f}  (vs nat mean {nat['cpg'].mean():.4f})"
    )
    print(
        f"  outlier motif mean = {outliers['motif'].mean():.2f}  (vs nat mean {nat['motif'].mean():.2f})"
    )
    print(
        f"  outlier K562 mean = {outliers['K562_log2FC'].mean():+.3f}  (vs nat mean {nat['K562_log2FC'].mean():+.3f})"
    )

    print("\n=== Motif-rich + CpG-poor outliers (top 1% motif, bottom 25% CpG) ===")
    motif_q99 = nat["motif"].quantile(0.99)
    cpg_q25 = nat["cpg"].quantile(0.25)
    rev = nat[(nat["motif"] >= motif_q99) & (nat["cpg"] <= cpg_q25)]
    print(f"  n outliers = {len(rev):,}")
    print(f"  outlier motif mean = {rev['motif'].mean():.2f}")
    print(f"  outlier CpG mean = {rev['cpg'].mean():.4f}")
    print(
        f"  outlier K562 mean = {rev['K562_log2FC'].mean():+.3f}  (vs nat mean {nat['K562_log2FC'].mean():+.3f})"
    )

    # === 4. CpG distribution + activity scatter — plot ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: CpG distributions
    ax = axes[0]
    bins = np.linspace(0, 0.15, 50)
    for label, c, color in [
        ("Natural Gosai (798k)", nat["cpg"], "#0072B2"),
        ("BODA-designed OOD (23k)", ood["cpg"], "#D55E00"),
        ("Gosai ctrl_neg", nat[nat["class"] == "ctrl_neg"]["cpg"], "#999999"),
        ("Gosai CRE peaks", nat[nat["data_project"] == "CRE"]["cpg"], "#009E73"),
    ]:
        h, e = np.histogram(c, bins=bins, density=True)
        ax.plot(0.5 * (e[:-1] + e[1:]), h, color=color, lw=2, label=f"{label}  μ={c.mean():.4f}")
    ax.set_xlabel("CpG density", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("CpG content distributions", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.25)

    # Right: CpG-vs-activity, color by source
    ax = axes[1]
    # Sample to avoid plot crush
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(nat), size=min(20000, len(nat)), replace=False)
    nat_s = nat.iloc[sample_idx]
    ax.scatter(
        nat_s["cpg"],
        nat_s["K562_log2FC"],
        s=2,
        c="#0072B2",
        alpha=0.15,
        label=f"Natural Gosai (sampled n=20k of {len(nat):,})",
    )
    ax.scatter(
        ood["cpg"],
        ood["K562_log2FC"],
        s=4,
        c="#D55E00",
        alpha=0.4,
        label=f"BODA-designed OOD (n={len(ood):,})",
    )
    ax.set_xlabel("CpG density", fontsize=13)
    ax.set_ylabel("K562 log2FC (experimental)", fontsize=13)
    ax.set_title("CpG vs activity by source", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 0.15)
    ax.set_ylim(-3, 10)

    fig.tight_layout()
    fig.savefig(OUT / "cpg_distributions_and_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "cpg_distributions_and_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT}/cpg_distributions_and_scatter.png")


if __name__ == "__main__":
    main()
