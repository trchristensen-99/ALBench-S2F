"""Share of each replicate pool's sequences that also sit in the canonical nested prefix of the same size (the LegNet bank's training set).

The loop runs train on `k562_reps` and the bank on `k562_nested`; how much they share decides whether a stack of both is one training set or two.

    python scripts/analysis/pool_overlap.py      (cpuq)
"""

from __future__ import annotations

import hashlib

import numpy as np

G = "outputs/grid_pools"


def keys(p: str) -> set:
    z = np.load(p, allow_pickle=True)
    s = z["sequences"]
    if s.dtype == object or s.dtype.kind in "US":
        return set(map(str, s))
    a = np.ascontiguousarray(s.reshape(len(s), -1))
    return {hashlib.md5(r.tobytes()).hexdigest() for r in a}


def main() -> None:
    for D in (10000, 30000, 100000):
        can = keys(f"{G}/k562_nested/genomic_d{D}.npz")
        for r in (1, 2, 3):
            rep = keys(f"{G}/k562_reps/genomic_r{r}_d{D}.npz")
            print(
                f"D={D} r{r}: {len(rep)} seqs, {len(rep & can) / len(rep):.3f} also in the canonical prefix"
            )


if __name__ == "__main__":
    main()
