"""Stage A of the out-of-fold oracle evaluation: map every battery sequence to the fold that HELD IT OUT.

Why this is needed. The deployed oracle label for a battery sequence is `oracle_mean`, the average of
all 10 fold models. But `experiments/train_oracle_s2_fullcv.py` trains on a RANDOM permutation of the
full 856,252-sequence pool with no chromosome exclusion, so each battery sequence sat in the TRAINING
split of 9 folds and was held out by exactly 1. Every published battery number is therefore
9/10 in-sample, and the r = 0.974 on absolute activity actually EXCEEDS the assay's own reliability
ceiling of 0.943 -- which is only possible if the model has partly fit the measurement noise.

The split is fully reproducible, so no retraining is required:
    perm      = np.random.default_rng(seed=42).permutation(856252)
    fold k val = perm[k*fold_size : (k+1)*fold_size]      fold_size = 856252 // 10
    (the last fold absorbs the remainder)

This stage emits, per fold, the battery row indices that fold held out. Stage B then predicts each
sequence with ONLY its held-out fold -- roughly 114k forward passes total, one tenth of a full
battery rescore, and trivially parallel as a 10-task array.
"""

import argparse
import json
import os

import numpy as np

SEED = 42
N_FOLDS = 10
BATTERY = {
    "genomic": ("genomic_oracle.npz", ["sequences"]),
    "ood": ("ood_oracle.npz", ["sequences"]),
    "ctrl_neg": ("ctrl_neg_oracle.npz", ["sequences"]),
    "snv": ("snv_oracle.npz", ["ref_sequences", "alt_sequences"]),
}


def fold_of(idx, pool, n_folds=N_FOLDS):
    """Vectorised inverse of the training split: which fold held out each pool row."""
    perm = np.random.default_rng(seed=SEED).permutation(pool)
    fold_size = pool // n_folds
    owner = np.empty(pool, dtype=np.int8)
    for k in range(n_folds):
        s = k * fold_size
        e = s + fold_size if k < n_folds - 1 else pool
        owner[perm[s:e]] = k
    return owner[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery_dir", default="data/k562/test_sets_ag_s2_chrsplit")
    ap.add_argument("--cache_dir", default="outputs/oracle_full856k_clean")
    ap.add_argument("--out", default="outputs/oracle_oof/foldmap.npz")
    args = ap.parse_args()

    from scripts.build_full_oracle_cache import load_all_sequences

    all_seqs, _ = load_all_sequences()
    pool = len(all_seqs)
    print(f"[pool] {pool:,} sequences (fold_size = {pool // N_FOLDS:,})")
    index = {}
    for i, s in enumerate(all_seqs):
        index.setdefault(str(s), i)  # first occurrence wins; duplicates share a fold below
    print(f"[pool] {len(index):,} unique sequences ({pool - len(index):,} duplicates)")

    out, summary = {}, {}
    for name, (fn, keys) in BATTERY.items():
        path = os.path.join(args.battery_dir, fn)
        if not os.path.exists(path):
            print(f"[skip] {fn} missing")
            continue
        z = np.load(path, allow_pickle=True)
        for key in keys:
            seqs = [str(x) for x in z[key]]
            idx = np.array([index.get(s, -1) for s in seqs], dtype=np.int64)
            found = idx >= 0
            owner = np.full(len(seqs), -1, dtype=np.int8)
            owner[found] = fold_of(idx[found], pool)
            tag = f"{name}:{key}"
            out[f"{tag}|owner"] = owner
            out[f"{tag}|pool_idx"] = idx
            counts = np.bincount(owner[found], minlength=N_FOLDS).tolist()
            summary[tag] = {
                "n": len(seqs),
                "in_pool": int(found.sum()),
                "not_in_pool": int((~found).sum()),
                "per_fold": counts,
            }
            print(
                f"  {tag:<22} n={len(seqs):>7,}  in_pool={int(found.sum()):>7,}  "
                f"missing={int((~found).sum()):>6,}  per-fold={counts}"
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump({"pool": pool, "seed": SEED, "n_folds": N_FOLDS, "sets": summary}, f, indent=2)
    tot = sum(v["in_pool"] for v in summary.values())
    print(
        f"\n[oof] {tot:,} sequences need ONE forward pass each "
        f"(~{tot / N_FOLDS:,.0f} per fold task)"
    )
    print(f"[oof] wrote {args.out}")


if __name__ == "__main__":
    main()
