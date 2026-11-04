#!/usr/bin/env python
"""Nested prefixes at the intermediate sizes the active-learning arms stop at (55k and 80k on the way from 30k to 105k).

Every canonical pool is already nested, so a size that is not on the standard ladder is the first N sequences of the next size up: no generation and no oracle call, and a 55k set still contains the 30k one, which is
what the active-learning comparison needs (3 rounds of 25k against 1 round of 75k must differ only in how the data was chosen, not in which pool it came from).

    python scripts/slice_intermediate_pools.py [--sizes 55000,80000] [--reservoirs genomic,...] [--dry]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
NEST = f"{ROOT}/outputs/grid_pools/k562_nested"
REPS = f"{ROOT}/outputs/grid_pools/k562_reps"
LADDER = (10000, 30000, 100000, 300000)


def slice_one(src: str, dst: str, n: int, dry: bool) -> str:
    z = np.load(src, allow_pickle=True)
    have = len(z["sequences"])
    if have < n:
        return f"SKIP {os.path.basename(dst)}: source holds {have:,} < {n:,}"
    out = {
        k: (z[k][:n] if getattr(z[k], "shape", None) and z[k].shape[:1] == (have,) else z[k])
        for k in z.files
    }
    if not dry:
        np.savez_compressed(dst, **out)
    return f"{'would write' if dry else 'wrote'} {os.path.basename(dst)}  n={n:,} from {os.path.basename(src)} ({have:,})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="55000,80000")
    ap.add_argument("--reservoirs", default="genomic")
    ap.add_argument(
        "--reps", default="1,2,3", help="replicate pools to slice as well; empty for none"
    )
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for res in a.reservoirs.split(","):
        for n in (int(x) for x in a.sizes.split(",")):
            up = min((s for s in LADDER if s >= n), default=None)
            if up is None:
                print(f"SKIP {res} {n}: above the ladder")
                continue
            print(slice_one(f"{NEST}/{res}_d{up}.npz", f"{NEST}/{res}_d{n}.npz", n, a.dry))
            for r in (x for x in a.reps.split(",") if x):
                src = f"{REPS}/{res}_r{r}_d{up}.npz"
                if os.path.exists(src):
                    print("  " + slice_one(src, f"{REPS}/{res}_r{r}_d{n}.npz", n, a.dry))


if __name__ == "__main__":
    main()
