"""Analyze Gosai's ctrl_neg category — the real episomal ground truth
for genomic non-regulatory DNA (n=503).

Reports:
- Distribution stats per cell line (K562 / HepG2 / SKNSH)
- CpG content vs measured activity correlation (in real data, is the
  CpG-shortcut the oracle learned actually present in the labels?)
- GC content vs activity (sanity check)
- Comparison vs other Gosai categories that we expect to be inactive
  (UKBB random variants, GTEx random) — does ctrl_neg match those?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]


def gc_content(seq: str) -> float:
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return 0.0
    return (seq.count("G") + seq.count("C")) / n


def cpg_count(seq: str) -> int:
    """Count CG dinucleotides."""
    seq = seq.upper()
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G")


def cpg_density(seq: str) -> float:
    """CpG count normalized by sequence length."""
    if len(seq) <= 1:
        return 0.0
    return cpg_count(seq) / (len(seq) - 1)


def main():
    print("Loading Gosai dataset...")
    gosai = pd.read_csv(
        REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False
    )
    print(f"  total: {len(gosai):,}")

    cn = gosai[gosai["class"] == "ctrl_neg"].copy()
    print(f"  ctrl_neg: n={len(cn)}")

    # Compute features
    print("\nComputing GC + CpG features for ctrl_neg sequences...")
    cn["gc"] = cn["sequence"].apply(gc_content)
    cn["cpg_density"] = cn["sequence"].apply(cpg_density)
    cn["seq_len"] = cn["sequence"].str.len()

    print(f"\n=== ctrl_neg: distribution per cell line ===")
    for cell in ["K562", "HepG2", "SKNSH"]:
        col = f"{cell}_log2FC"
        v = cn[col].dropna()
        print(
            f"  {cell:8s}: n={len(v):,}  mean={v.mean():+.3f}  std={v.std():.3f}  "
            f"median={v.median():+.3f}  q05={v.quantile(0.05):+.3f}  q95={v.quantile(0.95):+.3f}"
        )

    print(f"\n=== ctrl_neg: GC content distribution ===")
    print(
        f"  GC: mean={cn['gc'].mean():.3f}  std={cn['gc'].std():.3f}  "
        f"q05={cn['gc'].quantile(0.05):.3f}  q95={cn['gc'].quantile(0.95):.3f}"
    )
    print(
        f"  CpG density: mean={cn['cpg_density'].mean():.4f}  std={cn['cpg_density'].std():.4f}  "
        f"q05={cn['cpg_density'].quantile(0.05):.4f}  q95={cn['cpg_density'].quantile(0.95):.4f}"
    )
    print(
        f"  Seq len: median={int(cn['seq_len'].median())}, range=[{cn['seq_len'].min()}, {cn['seq_len'].max()}]"
    )

    print(f"\n=== Correlation: feature vs activity (Pearson r, Spearman ρ) ===")
    print(f"   In real Gosai ctrl_neg measurements, does CpG/GC predict K562 activity?")
    for cell in ["K562", "HepG2", "SKNSH"]:
        col = f"{cell}_log2FC"
        sub = cn.dropna(subset=[col])
        if len(sub) < 3:
            continue
        for feat in ["gc", "cpg_density"]:
            r, p_r = pearsonr(sub[feat], sub[col])
            rho, p_rho = spearmanr(sub[feat], sub[col])
            print(
                f"  {cell:6s} ~ {feat:13s}: pearson r={r:+.3f} (p={p_r:.2g})  "
                f"spearman ρ={rho:+.3f} (p={p_rho:.2g})  n={len(sub)}"
            )

    # Compare ctrl_neg vs the broader K562 distribution + GTEx (which we'd
    # expect to be similar to ctrl_neg since GTEx are mostly non-regulatory variants)
    print(f"\n=== Distribution context ===")
    for cls in ["ctrl_neg", "GTEx", "UKBB", "ctrl_emvar"]:
        sub = gosai[gosai["class"].fillna("").str.contains(cls, regex=False)]
        if len(sub) == 0 and cls in {"GTEx", "UKBB"}:
            sub = gosai[gosai["data_project"] == cls]
        v = sub["K562_log2FC"].dropna()
        if len(v) >= 100:
            print(
                f"  {cls:12s}: n={len(v):>7,}  K562 mean={v.mean():+.3f}  std={v.std():.3f}  median={v.median():+.3f}"
            )

    # Save the ctrl_neg panel as a parquet for fast loading by eval scripts
    out = REPO / "data/k562/gosai_ctrl_neg.parquet"
    cn[["sequence", "K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC", "gc", "cpg_density"]].to_parquet(
        out
    )
    print(f"\nSaved ctrl_neg panel: {out}")
    print(f"  ({len(cn)} sequences, ready for use in bias_eval)")


if __name__ == "__main__":
    main()
