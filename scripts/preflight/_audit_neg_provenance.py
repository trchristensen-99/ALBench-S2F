"""Audit the provenance of every negative-augmentation TSV: are the
labels REAL Gosai measurements, or synthetic samples from some other
distribution? Print mean labels per type + how many sequences are
literally present in Gosai's MPRA data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main():
    gosai = pd.read_csv(
        REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False
    )
    print(f"Gosai dataset: {len(gosai):,} sequences with measured K562 episomal MPRA")
    cn = gosai[gosai["class"] == "ctrl_neg"]
    print(
        f"Gosai ctrl_neg: n={len(cn)}, K562 mean={cn['K562_log2FC'].mean():+.3f}, std={cn['K562_log2FC'].std():.3f}, median={cn['K562_log2FC'].median():+.3f}"
    )
    print()

    gosai_seqs = set(gosai["sequence"].astype(str))

    print("=== Negative TSVs — provenance audit ===")
    print(f"{'TSV':<55}  {'rows':>8}  {'mean_label':>11}  {'in_Gosai':>10}")
    for tsv_path in sorted((REPO / "data/synthetic_negatives").glob("*.tsv")):
        df = pd.read_csv(tsv_path, sep="\t")
        n = len(df)
        if "K562_log2FC" not in df.columns:
            continue
        mean = df["K562_log2FC"].mean()
        matches = (
            df["sequence"].astype(str).isin(gosai_seqs).sum() if "sequence" in df.columns else 0
        )
        print(f"{tsv_path.name:<55}  {n:>8,}  {mean:>+11.3f}  {matches:>10,}")


if __name__ == "__main__":
    main()
