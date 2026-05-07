"""Generate Sahu-corrected negative-augmentation TSV.

Replaces the Agarwal-lentiviral-derived labels (mean -0.45) with
labels sampled from Gosai's actual ctrl_neg distribution
(mean +0.27, std 0.49 — real K562 episomal MPRA measurements
for genomic non-regulatory DNA), with a small CpG-density tilt
matching Sahu's truly-random N150 STARR-seq finding (~25% activity
increase from low to high CpG, mapped to ~+0.05 log2FC).

Output: data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE_TSV = REPO / "data/synthetic_negatives/dinuc_shuffled_negatives.tsv"
OUT_DIR = REPO / "data/synthetic_negatives_sahu"

# Empirical Gosai ctrl_neg distribution
CTRL_NEG_MEAN = 0.27
CTRL_NEG_STD = 0.49
# Sahu CpG tilt: low-CpG bin count=1.52, high-CpG bin count=1.90 → ~25% activity increase
# Mapped to log2FC scale (gentle linear tilt anchored at typical CpG=0.06):
# label = ctrl_neg_mean + cpg_tilt_per_unit * (cpg_density - 0.06)
# Where 25% increase across [0.03, 0.10] CpG range translates to ~+0.06 log2FC delta.
CPG_TILT_PER_UNIT = 0.86  # +0.86 per +1.0 CpG density (i.e. +0.06 across 0.03-0.10 range)
CPG_PIVOT = 0.06  # average CpG density of truly random ACGT


def cpg_density(seq: str) -> float:
    seq = seq.upper()
    if len(seq) <= 1:
        return 0.0
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G") / (
        len(seq) - 1
    )


def main():
    print(f"Reading {SOURCE_TSV} ...")
    df = pd.read_csv(SOURCE_TSV, sep="\t")
    print(f"  loaded {len(df):,} sequences")
    print(
        f"  current label mean: {df['K562_log2FC'].mean():+.3f}, std: {df['K562_log2FC'].std():.3f}"
    )

    print("\nComputing CpG density per sequence...")
    df["cpg"] = df["sequence"].apply(cpg_density)
    print(
        f"  CpG mean: {df['cpg'].mean():.4f}, q05-q95: [{df['cpg'].quantile(0.05):.4f}, {df['cpg'].quantile(0.95):.4f}]"
    )

    print("\nSampling Sahu-corrected labels from Gosai ctrl_neg distribution + CpG tilt...")
    rng = np.random.default_rng(42)
    base = rng.normal(loc=CTRL_NEG_MEAN, scale=CTRL_NEG_STD, size=len(df))
    tilt = CPG_TILT_PER_UNIT * (df["cpg"].to_numpy() - CPG_PIVOT)
    new_labels = base + tilt
    df["K562_log2FC"] = new_labels.astype(np.float32)
    df["category"] = "dinuc_shuffled_sahu_corrected"
    print(f"  new label mean: {new_labels.mean():+.3f}, std: {new_labels.std():.3f}")
    print(f"  by CpG bin (deciles):")
    df["_cpg_bin"] = pd.qcut(df["cpg"], 10, labels=False, duplicates="drop")
    for b in sorted(df["_cpg_bin"].dropna().unique()):
        sub = df[df["_cpg_bin"] == b]
        print(
            f"    bin {int(b)}  cpg={sub['cpg'].mean():.4f}  label={sub['K562_log2FC'].mean():+.3f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tsv = OUT_DIR / "dinuc_shuffled_sahu.tsv"
    df.drop(columns=["cpg", "_cpg_bin"]).to_csv(out_tsv, sep="\t", index=False)
    print(f"\nSaved {out_tsv}")

    # Metadata for provenance
    (OUT_DIR / "metadata.json").write_text(
        '{"source": "Sahu-corrected: Gosai ctrl_neg dist (mean=0.27, std=0.49) + '
        f'CpG tilt {CPG_TILT_PER_UNIT}/unit pivot at {CPG_PIVOT}", '
        f'"n_sequences": {len(df)}, "seed": 42}}\n'
    )


if __name__ == "__main__":
    main()
