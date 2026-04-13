#!/usr/bin/env python
"""Generate the 500K breakthrough dataset for aggressive CpG debiasing.

Creates: data/neg_aug_breakthrough/breakthrough_500k.tsv
- 200K CpG-enriched versions of training sequences (same labels)
- 200K CpG-depleted versions of training sequences (same labels)
- 90K random sequences at 6 CpG levels with ctrl_neg-level labels
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def add_cpg(seq, rng, n_add=15):
    result = list(seq.upper())
    gc_pos = [i for i in range(len(result) - 1) if result[i] == "G" and result[i + 1] == "C"]
    if gc_pos and n_add > 0:
        chosen = rng.choice(gc_pos, size=min(n_add, len(gc_pos)), replace=False)
        for pos in chosen:
            result[pos] = "C"
            result[pos + 1] = "G"
    return "".join(result)


def cpg_deplete(seq):
    result = list(seq.upper())
    for i in range(len(result) - 1):
        if result[i] == "C" and result[i + 1] == "G":
            result[i] = "T"
    return "".join(result)


def gen_cpg_controlled(n, cpg_freq, rng):
    n_cpg = max(0, round(cpg_freq * 199))
    seqs = []
    for _ in range(n):
        bases = ["G"] * 50 + ["C"] * 50 + ["A"] * 50 + ["T"] * 50
        rng.shuffle(bases)
        seq = list(bases)
        for i in range(199):
            if seq[i] == "C" and seq[i + 1] == "G":
                seq[i + 1] = rng.choice(["A", "T", "C"])
        if n_cpg > 0:
            positions = rng.choice(199, size=min(n_cpg, 99), replace=False)
            for pos in positions:
                seq[pos] = "C"
                seq[pos + 1] = "G"
        seqs.append("".join(seq))
    return seqs


def main():
    rng = np.random.default_rng(999)
    out_dir = REPO / "data" / "neg_aug_breakthrough"
    out_dir.mkdir(parents=True, exist_ok=True)

    gosai = pd.read_csv(
        REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt",
        sep="\t",
        low_memory=False,
        usecols=["sequence", "K562_log2FC", "chr"],
    )
    test_chrs = {"chr7", "chr13", "chr19", "chr21", "chrX"}
    train = gosai[~gosai["chr"].isin(test_chrs)].dropna(subset=["sequence", "K562_log2FC"])

    N = 200000
    sample = train.sample(N, random_state=501)
    seqs, labels = [], []

    print("Generating 200K enriched + 200K depleted...")
    for _, row in sample.iterrows():
        seq = str(row["sequence"])[:200]
        label = row["K562_log2FC"]
        seqs.append(add_cpg(seq, rng, n_add=15))
        labels.append(label)
        seqs.append(cpg_deplete(seq))
        labels.append(label)

    print("Generating 90K multi-CpG random negatives...")
    for cpg in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
        neg_seqs = gen_cpg_controlled(15000, cpg, rng)
        neg_labels = list(rng.normal(0.15, 0.20, 15000))
        seqs.extend(neg_seqs)
        labels.extend(neg_labels)

    out_path = out_dir / "breakthrough_500k.tsv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "K562_log2FC", "category"])
        for seq, label in zip(seqs, labels):
            w.writerow([seq, f"{label:.6f}", "synthetic"])

    print(f"Generated {len(seqs):,} sequences -> {out_path}")


if __name__ == "__main__":
    main()
