"""Keep only ref/alt pairs whose BOTH alleles were held out by the SAME oracle fold.

A delta is only out-of-fold if one model scored both alleles; pairs split across folds would give a
difference of two models' calibration offsets. Since each sequence is held out by exactly one of ten
folds, roughly 1/10 of pairs qualify. Filtering BEFORE the predict stage avoids scoring the ~90% of
sequences whose pairs are unusable, cutting the GPU work tenfold.
"""

import argparse
import os

import numpy as np

SEED, N_FOLDS = 42, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/k562/test_sets_mono_genomewide/snv_oracle.npz")
    ap.add_argument("--out_dir", default="data/k562/test_sets_mono_samefold")
    args = ap.parse_args()

    from scripts.build_full_oracle_cache import load_all_sequences

    all_seqs, _ = load_all_sequences()
    pool = len(all_seqs)
    perm = np.random.default_rng(seed=SEED).permutation(pool)
    fold_size = pool // N_FOLDS
    owner_row = np.full(pool, -1, np.int8)
    for k in range(N_FOLDS):
        e = (k + 1) * fold_size if k < N_FOLDS - 1 else pool
        owner_row[perm[k * fold_size : e]] = k

    first, ambiguous = {}, set()
    for i, sq in enumerate(all_seqs):
        sq = str(sq)
        f = int(owner_row[i])
        if sq in first:
            if first[sq] != f:
                ambiguous.add(sq)
        else:
            first[sq] = f

    z = np.load(args.pairs, allow_pickle=True)
    r = [str(x) for x in z["ref_sequences"]]
    a = [str(x) for x in z["alt_sequences"]]

    def own(sq):
        return -1 if (sq not in first or sq in ambiguous) else first[sq]

    fr = np.array([own(x) for x in r], np.int8)
    fa = np.array([own(x) for x in a], np.int8)
    keep = (fr >= 0) & (fa >= 0) & (fr == fa)
    print(f"[pairs] {len(r):,} total")
    print(f"        both alleles mapped: {int(((fr >= 0) & (fa >= 0)).sum()):,}")
    print(f"        SAME fold          : {int(keep.sum()):,} ({keep.mean():.1%})")
    print(f"        per-fold counts    : {np.bincount(fr[keep], minlength=N_FOLDS).tolist()}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "snv_oracle.npz")
    d = {
        k: (z[k][keep] if getattr(z[k], "shape", ()) and z[k].shape[:1] == (len(r),) else z[k])
        for k in z.files
    }
    d["n_pairs"] = int(keep.sum())
    d["fold"] = fr[keep]
    np.savez_compressed(out, **d)
    print(
        f"[wrote] {out} -> {int(keep.sum()) * 2:,} sequences to score "
        f"(~{int(keep.sum()) * 2 / N_FOLDS:,.0f} per fold task)"
    )


if __name__ == "__main__":
    main()
