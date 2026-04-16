#!/usr/bin/env python3
"""Generate motif-conditional negative augmentation data.

The key insight: random DNA with high CpG should be INACTIVE, but
designed sequences with high CpG AND TF motifs should be ACTIVE.
We need to teach the model this distinction.

Generates:
  1. Random DNA at various CpG levels → labeled as inactive (~0.27, Gosai ctrl_neg mean)
  2. Random DNA WITH planted JASPAR motifs + high CpG → keep oracle-predicted labels
     (these are synthetic "functional" sequences that should be active)
  3. CpG-depleted versions of training sequences → same label as original
     (teaches: removing CpG shouldn't change activity of real regulatory elements)

Output: TSV files ready for the --negatives_path flag in train_stage2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Known K562 TF motifs (consensus sequences from JASPAR)
K562_MOTIFS = {
    "GATA1": "AGATAAGG",
    "GATA2": "AGATAAGG",
    "TAL1": "CAGATG",
    "KLF1": "CCACACCCT",
    "SP1": "GGGCGGG",
    "NFE2": "TGCTGAGTCA",
    "MYC": "CACGTG",
    "AP1": "TGAGTCA",
    "ETS1": "GGAAGTG",
    "CTCF": "CCACCAGGGGGCGC",
    "YY1": "CCGCCATNT",
    "NRF1": "GCGCATGCGC",
}


def gen_random_seq(length, rng):
    return "".join(rng.choice(list("ACGT"), length))


def gen_cpg_controlled(length, cpg_freq, rng):
    """Generate random sequence with controlled CpG frequency."""
    n_cpg = int(cpg_freq * (length - 1))
    seq = list(gen_random_seq(length, rng))
    placed = 0
    positions = rng.permutation(length - 1)
    for pos in positions:
        if placed >= n_cpg:
            break
        seq[pos] = "C"
        seq[pos + 1] = "G"
        placed += 1
    return "".join(seq)


def plant_motifs(seq, motifs, rng, n_motifs=2):
    """Plant random TF motifs into a sequence."""
    seq = list(seq)
    motif_names = list(motifs.keys())
    for _ in range(n_motifs):
        motif_name = rng.choice(motif_names)
        motif = motifs[motif_name].replace("N", rng.choice(list("ACGT")))
        pos = rng.integers(0, len(seq) - len(motif))
        for j, base in enumerate(motif):
            seq[pos + j] = base
    return "".join(seq)


def cpg_deplete(seq):
    """Remove all CpG dinucleotides: CG → TG."""
    return seq.replace("CG", "TG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/neg_aug_motif_conditional")
    parser.add_argument(
        "--n-random",
        type=int,
        default=20000,
        help="Number of random inactive sequences per CpG level",
    )
    parser.add_argument(
        "--n-motif", type=int, default=10000, help="Number of motif-containing sequences"
    )
    parser.add_argument(
        "--n-deplete", type=int, default=20000, help="Number of CpG-depleted training seq copies"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    GOSAI_CTRL_NEG_MEAN = 0.27  # Real mean activity of ctrl_neg sequences
    GOSAI_CTRL_NEG_STD = 0.50  # Approximate std

    all_seqs, all_labels, all_cats = [], [], []

    # 1. Random DNA at various CpG levels → inactive labels
    print("Generating random inactive sequences...")
    for cpg_freq in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
        for _ in range(args.n_random // 6):
            seq = gen_cpg_controlled(200, cpg_freq, rng)
            # Label drawn from ctrl_neg distribution
            label = rng.normal(GOSAI_CTRL_NEG_MEAN, GOSAI_CTRL_NEG_STD)
            all_seqs.append(seq)
            all_labels.append(float(label))
            all_cats.append(f"random_cpg{cpg_freq:.2f}")

    # 2. Random DNA with planted motifs + high CpG → higher activity labels
    # These teach: CpG + motifs = active (preserving the real relationship)
    print("Generating motif-containing sequences...")
    for _ in range(args.n_motif):
        cpg_freq = rng.uniform(0.03, 0.10)
        seq = gen_cpg_controlled(200, cpg_freq, rng)
        seq = plant_motifs(seq, K562_MOTIFS, rng, n_motifs=rng.integers(1, 4))
        # Higher activity label (motifs make it functional)
        label = rng.normal(1.5, 0.8)  # Moderate activity
        all_seqs.append(seq)
        all_labels.append(float(label))
        all_cats.append("motif_high_cpg")

    # 3. CpG-depleted versions of real training sequences → same label
    # Teaches: CpG removal shouldn't change activity of real regulatory elements
    print("Generating CpG-depleted training copies...")
    train_path = REPO / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
    if train_path.exists():
        df = pd.read_csv(train_path, sep="\t", low_memory=False)
        # Sample from training chromosomes (not chr7,13,19,21,X)
        test_val_chrs = {7, 13, 19, 21, "X", "chr7", "chr13", "chr19", "chr21", "chrX"}
        train_df = df[~df["chr"].isin(test_val_chrs)].dropna(subset=["sequence", "K562_log2FC"])
        sample_idx = rng.choice(len(train_df), min(args.n_deplete, len(train_df)), replace=False)
        for idx in sample_idx:
            row = train_df.iloc[idx]
            seq = str(row["sequence"])[:200]
            depleted = cpg_deplete(seq)
            all_seqs.append(depleted)
            all_labels.append(float(row["K562_log2FC"]))
            all_cats.append("cpg_depleted_train")

    # Save full dataset
    out_df = pd.DataFrame(
        {
            "sequence": all_seqs,
            "K562_log2FC": all_labels,
            "category": all_cats,
        }
    )
    out_path = out_dir / "motif_conditional_negaug.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(out_df)} sequences to {out_path}")

    # Also save subsets for different experiments
    # A. Random-only (no motifs, no depleted) — pure "CpG alone = inactive" signal
    random_only = out_df[out_df["category"].str.startswith("random_")]
    random_only.to_csv(out_dir / "random_inactive_only.tsv", sep="\t", index=False)
    print(f"  random_inactive_only: {len(random_only)}")

    # B. Motif + random — teaches both "CpG alone = inactive" and "CpG + motifs = active"
    motif_plus = out_df[
        out_df["category"].isin(
            [c for c in out_df["category"].unique() if "random" in c or "motif" in c]
        )
    ]
    motif_plus.to_csv(out_dir / "motif_plus_random.tsv", sep="\t", index=False)
    print(f"  motif_plus_random: {len(motif_plus)}")

    # C. Full (all three types)
    print(f"  full: {len(out_df)}")

    # Print category summary
    print("\nCategory summary:")
    for cat in sorted(out_df["category"].unique()):
        sub = out_df[out_df["category"] == cat]
        print(f"  {cat}: n={len(sub)}, label_mean={sub['K562_log2FC'].mean():.3f}")


if __name__ == "__main__":
    main()
