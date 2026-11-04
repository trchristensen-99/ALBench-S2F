#!/usr/bin/env python
"""An unlabelled mutagenesis pool seeded only on sequences that are open chromatin in K562 (or HepG2).

Why: of the 798k Gosai library, 1.41% overlaps a merged K562 DNase peak and 2.94% a HepG2 one (`accessible_pool_activity.py`). Those windows are about twice as active, so if regulatory grammar is what
mutagenesis is meant to dissect, seeding on them should give a more informative pool than seeding on the library at large. There are only ~11k such sequences in K562, far too few to be a reservoir by
themselves, but ample as mutagenesis seeds: each seed carries many variants. This writes the pool with no labels; `scripts/label_pool.py` adds oracle_v3 ones, as for every other reservoir.

The test and validation chromosomes are excluded here, not downstream: chr7 and chr13 are test, chr19, chr21 and chrX are validation.

    python scripts/build_accessible_seed_pool.py --cell K562 --n 30000 --out outputs/grid_pools/k562_unlabelled/accessible_k562_d30000.npz
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from accessible_pool_activity import TABLE, WIN, load_peaks, overlaps  # noqa: E402

HOLDOUT = {"7", "13", "19", "21", "X"}  # test chr7/13, validation chr19/21/X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="K562", choices=["K562", "HepG2"])
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument(
        "--rate", type=float, default=0.10, help="per-base substitution rate, as PRM 10%"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(TABLE), delimiter="\t"))
    chrom = np.array([r["chr"] for r in rows])
    pos = np.array([int(r["IDs"].split(":")[1]) for r in rows])
    seqs = np.array([r["sequence"] for r in rows])
    peaks = load_peaks(
        f"/grid/wsbs/home_norepl/christen/ALBench-S2F/data/encode_accessibility/{a.cell}_peaks_merged.bed"
    )
    acc = overlaps(peaks, chrom, pos - WIN, pos + WIN)
    keep = (
        acc
        & ~np.isin(chrom, list(HOLDOUT))
        # the library holds a few windows that are shorter or carry an N; the insert standard is a clean 200-mer
        & np.array([len(s) == 200 and set(s) <= set("ACGT") for s in seqs])
    )
    seeds = np.unique(seqs[keep])
    print(
        f"{a.cell}: {acc.sum()} accessible, {keep.sum()} after holdout chromosomes, {len(seeds)} unique seeds"
    )
    if len(seeds) == 0:
        raise SystemExit("no seeds")

    rng = np.random.default_rng(a.seed)
    arr = (
        np.frombuffer(b"".join(s.encode() for s in seeds), np.uint8).reshape(len(seeds), -1).copy()
    )
    pick = rng.integers(0, len(seeds), size=a.n)  # seeds recur; each draw gets its own mutations
    out = arr[pick]
    nt = np.frombuffer(b"ACGT", np.uint8)
    mut = rng.random(out.shape) < a.rate
    repl = nt[rng.integers(0, 4, size=out.shape)]
    same = repl == out  # a draw equal to the original base is a no-op, so rotate it
    repl[same] = nt[
        (np.searchsorted(nt, out[same]) + 1 + rng.integers(0, 3, size=int(same.sum()))) % 4
    ]
    out[mut] = repl[mut]
    got = np.array([bytes(r).decode() for r in out])
    frac = float(
        np.mean(
            [
                sum(x != y for x, y in zip(g, seeds[p])) / len(g)
                for g, p in list(zip(got, pick))[:2000]
            ]
        )
    )
    print(f"wrote {len(got)} sequences, realised substitution rate {frac:.4f} (target {a.rate})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        sequences=got,
        strategy=f"accessible_{a.cell.lower()}",
        seed=a.seed,
        seed_sequences=len(seeds),
        mutation_rate=a.rate,
    )


if __name__ == "__main__":
    main()
