"""Emit a fold id for every row of the 856,252-sequence oracle pool, in pool order.

Keeping the pool ORDER unchanged matters: all_labels.npy and the cached embeddings are aligned to
`build_full_oracle_cache.load_all_sequences()`, so changing the composition would invalidate them.
Only the fold ASSIGNMENT changes here.

Pool layout (from load_all_sequences):
    rows 0            .. 798,063   Table S2, every measured row (both R and A alleles)
    rows 798,064      .. 833,289   the redundant alt block from the hashfrag SNV file
    rows 833,290      .. 856,251   designed high-activity sequences

Chromosome-based assignment has a useful side effect: the redundant alt block is the same variants
as rows already in the Table S2 block, so both copies get the SAME fold. Under the old random split
those copies landed in different folds ~90% of the time, which is what made duplicated sequences
unusable out-of-fold. They are still double-weighted within their fold, but they no longer leak.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="data/k562/oracle_folds_v2.json")
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    ap.add_argument("--snv", default="data/k562/test_sets/deprecated_hashfrag/"
                                    "test_snv_pairs_hashfrag.tsv")
    ap.add_argument("--out", default="data/k562/oracle_poolmap_v2.npy")
    args = ap.parse_args()

    spec = json.load(open(args.folds))
    c2f = spec["chrom_to_fold"]
    n_folds = spec["n_folds"]
    rng = np.random.default_rng(spec["seed"])

    from scripts.build_full_oracle_cache import load_all_sequences

    seqs, _ = load_all_sequences()
    pool = len(seqs)

    df = pd.read_csv(args.table, sep="\t", usecols=["IDs", "chr"], low_memory=False)
    ref_fold = (df["chr"].astype(str).str.replace("chr", "", regex=False)
                .map(c2f).to_numpy())
    n_ref = len(ref_fold)

    snv = pd.read_csv(args.snv, sep="\t", usecols=["IDs_ref"])
    alt_chrom = snv["IDs_ref"].astype(str).str.split(":", expand=True)[0]
    alt_fold = alt_chrom.map(c2f).to_numpy()
    n_alt = len(alt_fold)

    n_des = pool - n_ref - n_alt
    des_fold = rng.integers(0, n_folds, size=max(0, n_des))

    fold = np.concatenate([ref_fold, alt_fold, des_fold]).astype(float)
    print(f"[layout] table {n_ref:,} + alt {n_alt:,} + designed {n_des:,} = "
          f"{n_ref + n_alt + n_des:,}  (pool {pool:,})")
    assert n_ref + n_alt + n_des == pool, "pool layout mismatch - cache would misalign"

    n_nan = int(np.isnan(fold).sum())
    if n_nan:
        # unplaced contigs have no fold; drop them from training rather than guessing
        print(f"[warn] {n_nan:,} rows have no chromosome mapping -> marked -1 (excluded)")
    fold = np.nan_to_num(fold, nan=-1).astype(np.int8)

    cnt = np.bincount(fold[fold >= 0], minlength=n_folds)
    print(f"[folds] sizes {cnt.tolist()}  (min {cnt.min():,} max {cnt.max():,}, "
          f"ratio {cnt.max() / cnt.min():.2f})")

    # the payoff: do duplicated sequences now share a fold?
    from collections import defaultdict
    seen = defaultdict(set)
    for i, s in enumerate(seqs):
        seen[str(s)].add(int(fold[i]))
    split = sum(1 for v in seen.values() if len(v) > 1)
    print(f"[check] duplicated sequences landing in >1 fold: {split:,} of {len(seen):,} unique "
          f"(was ~31,793 under the random split)")

    np.save(args.out, fold)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
