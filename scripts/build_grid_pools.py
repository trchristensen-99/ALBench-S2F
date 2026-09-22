#!/usr/bin/env python
"""Materialise the scaling-grid pools with nested prefixes and DISJOINT replicates.

Two properties the scaling curves depend on:

NESTED SUBSAMPLING (within a replicate)
    A pool is one fixed permutation; curve point at increment I is prefix[:I]. So
    +10k is a strict subset of +30k is a strict subset of +100k ... Adding data
    only ever ADDS sequences, never swaps them, which is what makes a point-to-point
    difference attributable to the extra data rather than to a reshuffle.

DISJOINT REPLICATES (across replicates)
    Replicates must not share sequences, or their agreement understates true
    variance. MEASURED cross-seed overlap at n=20,000:

        genomic             6.4%   <- 1,271 collisions vs 1,270 expected by birthday
        random              0.0%
        dinuc_shuffle       0.0%
        motif_shared_core   0.0%
        zoonomia_orthologs  0.0%
        mutagenesis         0.0%

    So reseeding is sufficient for every SYNTHESISED strategy, but `genomic` draws
    from a fixed set of 314,981 real CREs and reseeding merely resamples it. Genomic
    replicates are therefore PARTITIONED into disjoint blocks of one shared
    permutation.

    Consequence: with R replicates the genomic arm can only reach 314,981 // R per
    replicate. At R=3 that is 104,993, so the genomic curve stops at +100k rather
    than +300k. This is a property of the data, not of the code -- there are only
    so many real CREs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Strategies that synthesise sequence: a new seed yields new sequences (measured 0% overlap).
RESEEDABLE = {
    "random",
    "gc_matched",
    "dinuc_shuffle",
    "mutagenesis",
    "evoaug",
    "evoaug_ours_default",
    "evoaug_paper2023",
    "motif_shared_core",
    "motif_ct_enriched",
    "motif_syntax_core",
    "zoonomia",
    "zoonomia_orthologs",
}
# Strategies drawing from a fixed finite set: replicates must partition it.
PARTITIONED = {"genomic"}

INCREMENTS = [10_000, 30_000, 100_000, 300_000, 1_000_000]


def plan(strategy: str, n_replicates: int, pool_size: int) -> dict:
    """Return the per-replicate index plan and the honest ceiling for this strategy."""
    if strategy in PARTITIONED:
        per_rep = pool_size // n_replicates
        ceiling = max([i for i in INCREMENTS if i <= per_rep], default=0)
        return {
            "mode": "partition",
            "pool_size": pool_size,
            "per_replicate": per_rep,
            "ceiling": ceiling,
            "note": (
                f"fixed pool of {pool_size:,}; {n_replicates} disjoint blocks of "
                f"{per_rep:,}. Curve stops at +{ceiling // 1000}k."
            ),
        }
    ceiling = max(INCREMENTS)
    return {
        "mode": "reseed",
        "pool_size": pool_size,
        "per_replicate": pool_size,
        "ceiling": ceiling,
        "note": f"seed 42+r gives disjoint sequences (measured 0% overlap); ceiling +{ceiling // 1000}k.",
    }


def replicate_indices(strategy: str, rep: int, n_replicates: int, pool_size: int, seed: int = 42):
    """Indices for one replicate. Partitioned strategies get a disjoint block."""
    if strategy in PARTITIONED:
        perm = np.random.default_rng(seed).permutation(pool_size)
        per_rep = pool_size // n_replicates
        return perm[rep * per_rep : (rep + 1) * per_rep]
    # reseeded pools are generated independently; the whole pool belongs to this replicate
    return np.arange(pool_size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = {"replicates": args.replicates, "increments": INCREMENTS, "strategies": {}}
    for sub in sorted(
        p for p in args.pool_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        strategy = sub.name
        npz = next(iter(sub.glob("pool_*.npz")), None)
        if npz is None:
            continue
        with np.load(npz, allow_pickle=True) as d:
            key = "sequences" if "sequences" in d else list(d.keys())[0]
            pool_size = len(d[key])
        p = plan(strategy, args.replicates, pool_size)
        p["source"] = str(npz)
        # label-once bookkeeping: each replicate labels only the NEW slice of the prefix
        p["label_segments"] = [
            {"from": lo, "to": hi}
            for lo, hi in zip([0] + INCREMENTS[:-1], INCREMENTS)
            if hi <= p["ceiling"]
        ]
        manifest["strategies"][strategy] = p
        print(f"  {strategy:24s} {p['mode']:10s} ceiling +{p['ceiling'] // 1000}k   {p['note']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
