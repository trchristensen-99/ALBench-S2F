"""Generate one reservoir-cache npz (sequences + oracle_labels) for the HP search.

The scaling HP search consumes pools via `--reservoir_cache <path>` where the npz
holds keys `sequences` (object array) + `oracle_labels` (float32) — a different
schema/location than scripts/generate_labeled_pools.py's pool.npz (`labels`). This
is the missing generator that scripts/slurm/relabel_reservoir_caches_chrsplit.sh
calls. It reuses generate_labeled_pools' building blocks so sequence generation +
oracle labeling stay in one place, and stamps `oracle_id` so a cache produced by a
different oracle can be told apart (the hashfrag-era caches carry no stamp).

Usage (one strategy, GPU job):
    uv run --no-sync python scripts/generate_reservoir_cache.py \
        --task k562 --reservoir random --D 1000000 --seed 42 \
        --oracle ag_s2 --out outputs/reservoir_cache/k562_random_d1000000_seed42.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.generate_labeled_pools import (  # noqa: E402
    _NEEDS_POOL,
    _generate_sequences,
    _label_sequences,
    _load_pool_sequences,
)

# Reservoirs that derive from / mutate the genomic pool need pool_seqs loaded even
# when not in generate_labeled_pools._NEEDS_POOL (loading is harmless if unused).
_POOL_PREFIXES = ("prm", "motif_clustering", "recombination", "evoaug", "phylogenetic")


def _needs_pool(reservoir: str) -> bool:
    return reservoir in _NEEDS_POOL or reservoir.startswith(_POOL_PREFIXES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["k562", "yeast"])
    ap.add_argument("--reservoir", required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oracle", default="ag_s2", choices=["ag", "ag_s2", "dream_rnn", "default"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from experiments.exp1_1_scaling import _load_oracle

    oracle_type = args.oracle
    if oracle_type == "default":
        oracle_type = "ag" if args.task == "k562" else "dream_rnn"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    pool_seqs, pool_labels = None, None
    if _needs_pool(args.reservoir):
        print(f"Loading genomic pool for {args.task}...", flush=True)
        pool_seqs, pool_labels = _load_pool_sequences(args.task)
        print(f"  pool: {len(pool_seqs):,} sequences", flush=True)

    print(f"Loading oracle {oracle_type}...", flush=True)
    oracle = _load_oracle(args.task, oracle_type=oracle_type)

    t0 = time.perf_counter()
    print(f"Generating {args.D:,} sequences with reservoir={args.reservoir}...", flush=True)
    seqs = _generate_sequences(
        args.reservoir, args.D, args.task, args.seed, pool_seqs, pool_labels, oracle=oracle
    )
    print(f"  generated {len(seqs):,} in {time.perf_counter() - t0:.1f}s", flush=True)

    print("Labeling with oracle...", flush=True)
    labels = _label_sequences(oracle, seqs)

    metadata = {
        "task": args.task,
        "oracle_type": oracle_type,
        "reservoir": args.reservoir,
        "D": int(args.D),
        "seed": int(args.seed),
        "label_mean": float(np.mean(labels)),
        "label_std": float(np.std(labels)),
    }
    np.savez_compressed(
        args.out,
        sequences=np.array(seqs, dtype=object),
        oracle_labels=labels,
        oracle_id=np.array(oracle_type),
        metadata=json.dumps(metadata),
    )
    print(
        f"Wrote {args.out}  n={len(seqs):,}  mean={metadata['label_mean']:.3f} "
        f"std={metadata['label_std']:.3f}  oracle_id={oracle_type}",
        flush=True,
    )


if __name__ == "__main__":
    main()
