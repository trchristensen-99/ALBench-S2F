#!/usr/bin/env python
"""HepG2 labels for the 856k oracle cache, in the cache's own order, so a HepG2 oracle can reuse the K562 cache and fold map.

The stage-2 trainer reads `cache_dir/all_labels.npy`, one float per cached sequence in the order ref -> alt -> OOD that `build_full_oracle_cache.load_all_sequences` fixes. Everything that order depends on
(the sequences, their AlphaGenome embeddings, the 10-fold pool map) is cell-type independent, so only the label vector has to be rebuilt. Writes the vector plus a mask of the rows HepG2 actually measured.

Coverage is not complete: the library's reference sequences and the designed set were assayed in HepG2, the SNV alt alleles were not (our only alt table carries K562 columns under both the k562 and hepg2
directories). Rows without a HepG2 measurement are NaN in the labels and False in the mask, and the trainer must skip them rather than treat them as zero.

    python scripts/build_hepg2_oracle_labels.py --out outputs/oracle_hepg2/all_labels_hepg2.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/outputs/oracle_hepg2/all_labels_hepg2.npy")
    a = ap.parse_args()

    from scripts.build_full_oracle_cache import load_all_sequences

    seqs, k562 = load_all_sequences()
    n = len(seqs)
    print(f"cache order: {n:,} sequences, K562 labels finite {np.isfinite(k562).mean() * 100:.1f}%")

    lab = np.full(n, np.nan, np.float32)
    # 1. reference sequences: the library table carries all three cell types
    ref = pd.read_csv(
        f"{ROOT}/data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False
    )
    by_seq = dict(zip(ref["sequence"], ref["HepG2_log2FC"].astype(np.float32)))
    # 2. designed sequences: measured in HepG2 as their own table
    ood = pd.read_csv(f"{ROOT}/data/hepg2/test_sets/test_ood_designed_hepg2.tsv", sep="\t")
    by_seq.update(zip(ood["sequence"], ood["HepG2_log2FC"].astype(np.float32)))
    print(f"HepG2 measurements available for {len(by_seq):,} distinct sequences")

    for i, s in enumerate(seqs):
        v = by_seq.get(s)
        if v is not None:
            lab[i] = v
    mask = np.isfinite(lab)
    print(f"matched {mask.sum():,} of {n:,} cached rows ({mask.mean() * 100:.1f}%)")
    # the three blocks of the cache, so the gap is attributable
    n_ref = len(ref)
    bounds = {
        "ref": (0, n_ref),
        "alt": (n_ref, n - len(ood) if n - len(ood) > n_ref else n),
        "ood": (max(n - len(ood), n_ref), n),
    }
    for name, (lo, hi) in bounds.items():
        if hi > lo:
            print(f"  {name:4s} rows {lo:,}-{hi:,}: {mask[lo:hi].mean() * 100:5.1f}% labelled")
    good = lab[mask]
    print(
        f"HepG2 label distribution: mean {good.mean():.3f} sd {good.std():.3f} min {good.min():.2f} max {good.max():.2f}"
    )
    print(f"correlation with K562 where both exist: {np.corrcoef(good, k562[mask])[0, 1]:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.save(a.out, lab)
    np.save(a.out.replace(".npy", "_mask.npy"), mask)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
