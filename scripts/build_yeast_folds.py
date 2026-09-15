"""Build the 10-fold split for the yeast DREAM-RNN oracle.

Mirrors the human AG oracle protocol (scripts/build_oracle_folds_v2.py) so the two
oracles are comparable: ten folds, and each model later takes one fold as TEST and
the next as VAL, leaving eight to train on.

WHY RANDOM FOLDS RATHER THAN CHROMOSOME FOLDS. The human oracle splits by chromosome
because its sequences are genomic and homologous regions would otherwise leak between
folds. DREAM yeast sequences are random 80-mers in a fixed promoter scaffold, so there
is no genomic coordinate to split on and a random split is the correct analogue.

WHAT STILL HAS TO BE CHECKED. Identical sequences must not straddle a fold boundary,
or a model is evaluated on a sequence it trained on. With 6M random 80-mers exact
duplicates should be rare, but "should be rare" is not a guarantee: the library was
synthesised, and synthesis collapses some designs. So duplicates are detected and
assigned as a group, and the count is reported rather than assumed to be zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", default="data/yeast/train.txt")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/yeast/oracle_folds_v2.npy")
    ap.add_argument("--meta-out", default="data/yeast/oracle_folds_v2.json")
    args = ap.parse_args()

    train = REPO / args.train if not Path(args.train).is_absolute() else Path(args.train)
    print(f"reading {train} ...", flush=True)
    seqs: list[str] = []
    with open(train) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                seqs.append(parts[0])
    n = len(seqs)
    print(f"  {n:,} sequences")

    # Group identical sequences so a duplicate cannot straddle a fold boundary.
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(seqs):
        groups[s].append(i)
    n_unique = len(groups)
    n_dup_rows = n - n_unique
    print(
        f"  {n_unique:,} distinct sequences; {n_dup_rows:,} duplicate rows "
        f"({100 * n_dup_rows / n:.3f}%)"
    )

    rng = np.random.default_rng(args.seed)
    keys = list(groups.keys())
    order = rng.permutation(len(keys))
    fold = np.empty(n, dtype=np.int8)
    # Assign whole groups round-robin over a shuffled order: keeps folds balanced
    # while guaranteeing every copy of a sequence lands in the same fold.
    for rank, gi in enumerate(order):
        f = rank % args.n_folds
        for idx in groups[keys[gi]]:
            fold[idx] = f

    counts = np.bincount(fold, minlength=args.n_folds)
    print(
        f"  fold sizes: min {counts.min():,} max {counts.max():,} "
        f"(ratio {counts.max() / max(counts.min(), 1):.3f})"
    )

    # Verify the property we actually care about.
    split = sum(1 for _, idxs in groups.items() if len({int(fold[i]) for i in idxs}) > 1)
    print(f"  duplicate groups split across folds: {split} (must be 0)")
    if split:
        raise SystemExit("fold assignment split a duplicate group; refusing to write")

    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, fold)
    meta = {
        "n": n,
        "n_unique": n_unique,
        "n_duplicate_rows": n_dup_rows,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "fold_sizes": counts.tolist(),
        "split_duplicate_groups": split,
        "source": str(train),
        "note": (
            "Random folds: DREAM yeast sequences are random 80-mers with no genomic "
            "coordinate, so the chromosome split used for the human oracle does not "
            "apply. Identical sequences are assigned as a group."
        ),
    }
    meta_out = (
        REPO / args.meta_out if not Path(args.meta_out).is_absolute() else Path(args.meta_out)
    )
    meta_out.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} and {meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
