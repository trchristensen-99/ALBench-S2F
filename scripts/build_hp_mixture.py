"""Build a reservoir-BALANCED mixture for unbiased FM hyperparameter tuning.

Why: tuning the FM fine-tune HPs on any single reservoir (e.g. genomic) would bias the whole
reservoir comparison toward whatever that reservoir likes. Instead we tune ONCE on an equal-parts
mixture of all compared reservoirs, then FREEZE those HPs for every cell. No per-cell HP search
(that is what makes the from-scratch MPRA-LegNet pipeline expensive), and no per-reservoir advantage.

Pair with scripts/analysis/hp_rank_stability.py, which verifies the reservoir RANKING is invariant to
the HP choice — if it is, the comparison is robust regardless of which config we froze.
"""

import argparse
import os

import numpy as np

DEFAULT_RESERVOIRS = [
    "evoaug_heavy",
    "motif_planted_v2",
    "phylogenetic_zoonomia",
    "random",
    "dinuc_shuffle",
]
GENOMIC_CACHE = "outputs/chr_split_cache/chr_train_ref_only.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservoirs", nargs="*", default=DEFAULT_RESERVOIRS)
    ap.add_argument("--include_genomic", action="store_true", default=True)
    ap.add_argument("--D", type=int, default=30000, help="total mixture size")
    ap.add_argument(
        "--src_D", type=int, default=100000, help="per-reservoir cache size to draw from"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="outputs/reservoir_cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sources = []
    if args.include_genomic:
        sources.append(("genomic", GENOMIC_CACHE))
    for r in args.reservoirs:
        sources.append((r, f"{args.cache_dir}/k562_{r}_d{args.src_D}_seed{args.seed}.npz"))

    per = args.D // len(sources)
    rng = np.random.default_rng(args.seed)
    seqs, labs, origin = [], [], []
    for name, path in sources:
        if not os.path.exists(path):
            print(f"  SKIP {name}: missing {path}")
            continue
        z = np.load(path, allow_pickle=True)
        s, y = z["sequences"], z["oracle_labels"].astype(np.float32)
        idx = rng.choice(len(s), size=min(per, len(s)), replace=False)
        seqs += [str(s[i]) for i in idx]
        labs.append(y[idx])
        origin += [name] * len(idx)
        print(f"  + {name}: {len(idx)}")

    y = np.concatenate(labs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, sequences=np.array(seqs), oracle_labels=y, origin=np.array(origin)
    )
    print(f"[mixture] wrote {args.out}: n={len(seqs)} from {len(set(origin))} reservoirs")


if __name__ == "__main__":
    main()
