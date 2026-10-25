"""Slice a raw labelled pool (scripts/build_pool.py output) into standard replicate files with nested prefix sizes.

    python scripts/program_oct22/slice_pool.py --raw DIR --strategy S --name N --size 30000 --reps 1 2 3 --prefix 10000 30000

Writes outputs/grid_pools/k562_reps/{N}_r{r}_d{p}.npz for every prefix p <= size. Refuses pools that are short, not oracle_v3-labelled or have non-finite labels.
`--name` differs from `--strategy` when a rebuilt pool must not overwrite an older one (e.g. the Zoonomia rebuild after the validation-chromosome fix).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stage2_lib import REPS, SEED, save_pool  # noqa: E402

from albench.pools import PoolSpec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--strategy", required=True)
ap.add_argument("--name", default="")
ap.add_argument("--size", type=int, required=True)
ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--prefix", type=int, nargs="+", required=True)
args = ap.parse_args()
name = args.name or args.strategy
for r in args.reps:
    src = str(
        PoolSpec(
            strategy=args.strategy, params={}, size=args.size, generation_seed=SEED(r), task="k562"
        ).path(args.raw)
    )
    z = np.load(src, allow_pickle=True)
    seqs, lab = list(z["sequences"]), z["oracle_labels"]
    if len(seqs) < args.size or str(z["oracle_id"]) != "oracle_v3" or not np.isfinite(lab).all():
        raise SystemExit(
            f"BAD raw pool {src}: n={len(seqs)} oracle={z['oracle_id']} finite={np.isfinite(lab).all()}"
        )
    print(f"r{r}: {src} n={len(seqs):,} unique={len(set(seqs)):,}", flush=True)
    for p in sorted(args.prefix):
        save_pool(f"{REPS}/{name}_r{r}_d{p}.npz", seqs[:p], lab[:p], "oracle_v3", name, SEED(r))
        print(f"  wrote {name}_r{r}_d{p}", flush=True)
