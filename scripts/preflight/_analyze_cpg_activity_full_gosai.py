"""Analyze CpG-activity relationship across the FULL Gosai dataset
(n=798k), not just ctrl_neg. This gives much wider CpG content range
and lets us check if the null relationship holds across the regime.

Also bins by CpG density and reports activity per bin to see if there's
a non-linear relationship that pearson correlation might miss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / max(1, len(seq))


def cpg_density(seq: str) -> float:
    seq = seq.upper()
    if len(seq) <= 1:
        return 0.0
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G") / (
        len(seq) - 1
    )


def main():
    print("Loading full Gosai dataset (n=798k)...")
    gosai = pd.read_csv(
        REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False
    )
    print(f"  loaded {len(gosai):,}")

    print("\nComputing CpG density and GC content for ALL sequences (slow ~2 min)...")
    gosai["cpg"] = gosai["sequence"].apply(cpg_density)
    gosai["gc"] = gosai["sequence"].apply(gc_content)

    # Separate by data subset
    print("\n=== CpG density distribution by subset ===")
    print(
        f"{'subset':<25}  {'n':>10}  {'mean':>7}  {'std':>6}  {'q05':>6}  {'q25':>6}  {'q50':>6}  {'q75':>6}  {'q95':>6}"
    )
    for label, sub in [
        ("ALL Gosai (n=798k)", gosai),
        ("ctrl_neg (n=503)", gosai[gosai["class"] == "ctrl_neg"]),
        ("ctrl_emvar", gosai[gosai["class"] == "ctrl_emvar"]),
        ("data_project=GTEx", gosai[gosai["data_project"] == "GTEX"]),
        ("data_project=UKBB", gosai[gosai["data_project"] == "UKBB"]),
        ("data_project=CRE", gosai[gosai["data_project"] == "CRE"]),
    ]:
        c = sub["cpg"]
        print(
            f"{label:<25}  {len(sub):>10,}  {c.mean():>7.4f}  {c.std():>6.4f}  "
            f"{c.quantile(0.05):>6.4f}  {c.quantile(0.25):>6.4f}  {c.quantile(0.50):>6.4f}  "
            f"{c.quantile(0.75):>6.4f}  {c.quantile(0.95):>6.4f}"
        )

    # CpG vs K562 activity, across full dataset
    print("\n=== CpG density vs K562 activity (full Gosai, all classes) ===")
    sub = gosai.dropna(subset=["K562_log2FC", "cpg"])
    r, p_r = pearsonr(sub["cpg"], sub["K562_log2FC"])
    rho, p_rho = spearmanr(sub["cpg"], sub["K562_log2FC"])
    print(f"  pearson r = {r:+.4f} (p={p_r:.2g})")
    print(f"  spearman ρ = {rho:+.4f} (p={p_rho:.2g})")
    print(f"  n = {len(sub):,}")

    # Bin by CpG density and report activity
    print("\n=== Activity per CpG-density bin (deciles) ===")
    sub = sub.copy()
    sub["cpg_bin"] = pd.qcut(sub["cpg"], 10, labels=False, duplicates="drop")
    print(
        f"{'bin':>3}  {'cpg_low':>8}  {'cpg_high':>8}  {'n':>8}  {'K562_mean':>10}  {'K562_std':>9}  {'K562_med':>9}"
    )
    for b in sorted(sub["cpg_bin"].dropna().unique()):
        bin_data = sub[sub["cpg_bin"] == b]
        cpg_lo, cpg_hi = bin_data["cpg"].min(), bin_data["cpg"].max()
        v = bin_data["K562_log2FC"]
        print(
            f"{int(b):>3}  {cpg_lo:>8.4f}  {cpg_hi:>8.4f}  {len(v):>8,}  "
            f"{v.mean():>+10.3f}  {v.std():>9.3f}  {v.median():>+9.3f}"
        )

    # Same for ctrl_neg (smaller n, cleaner)
    print("\n=== ctrl_neg only: activity per CpG quintile (n=503, narrow CpG range) ===")
    cn = gosai[gosai["class"] == "ctrl_neg"].copy()
    cn["cpg_bin"] = pd.qcut(cn["cpg"], 5, labels=False, duplicates="drop")
    for b in sorted(cn["cpg_bin"].dropna().unique()):
        bin_data = cn[cn["cpg_bin"] == b]
        cpg_lo, cpg_hi = bin_data["cpg"].min(), bin_data["cpg"].max()
        v = bin_data["K562_log2FC"]
        print(
            f"  bin {int(b)}  [{cpg_lo:.4f}-{cpg_hi:.4f}]  n={len(v)}  "
            f"K562 mean={v.mean():+.3f}  std={v.std():.3f}"
        )

    # Theoretical: truly random 200-bp ACGT sequence's CpG density distribution
    print("\n=== Theoretical CpG density for truly random 200-bp ACGT sequence ===")
    print("  P(CG) per position = 1/16 = 0.0625")
    print("  Expected CpG count in 200bp: 12.4")
    print("  Std (binomial sqrt(199*0.0625*0.9375)): 3.4")
    print("  ⇒ CpG density: mean 0.062, std 0.017, 95%CI ≈ [0.029, 0.097]")
    print()
    print("  Sahu N170 (truly random 170bp): CpG density mean ~0.063, range ~[0.025, 0.100]")
    print("  Gosai ctrl_neg (genomic non-reg): CpG density mean 0.009, range ~[0.000, 0.030]")
    print("  → ctrl_neg covers ONLY the LOW-CpG regime; Sahu N170 would cover much wider.")


if __name__ == "__main__":
    main()
