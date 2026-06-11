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
import os
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


def _load_background_cache(path: Path) -> tuple[list[str], np.ndarray | None]:
    """Load background sequences (+ optional oracle labels) from an npz, for
    generating a transform-matched *held-out* val cache from a disjoint pool
    (e.g. chr19/21/X via outputs/chr_split_cache/chr_val_ref_only.npz)."""
    z = np.load(path, allow_pickle=True)
    if "sequences" not in z.files:
        raise KeyError(f"{path} has no 'sequences' key (files={z.files})")
    seqs = [str(s) for s in z["sequences"]]
    labels = None
    for k in ("oracle_labels", "oracle_mean", "labels"):
        if k in z.files:
            labels = np.asarray(z[k], dtype=np.float32)
            break
    return seqs, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["k562", "yeast"])
    ap.add_argument("--reservoir", required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oracle", default="ag_s2", choices=["ag", "ag_s2", "dream_rnn", "default"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--background_cache",
        type=Path,
        default=None,
        help="npz of background sequences to transform instead of the genomic train "
        "pool (e.g. chr_val_ref_only.npz) — for held-out, transform-matched val sets. "
        "Also exported as RESERVOIR_BG_CACHE so self-loading samplers honor it.",
    )
    ap.add_argument(
        "--oracle_id_stamp",
        default=None,
        help="oracle_id string written to the cache (default: the oracle type). Set "
        "'full856k_clean' so the cache passes the HP-search contamination guard.",
    )
    args = ap.parse_args()

    # Self-loading samplers (e.g. MotifPlantedV2Sampler) read their backgrounds from
    # a fixed path; point them at the held-out pool via env so val == same transform.
    if args.background_cache is not None:
        os.environ["RESERVOIR_BG_CACHE"] = str(args.background_cache)

    from experiments.exp1_1_scaling import _load_oracle

    oracle_type = args.oracle
    if oracle_type == "default":
        oracle_type = "ag" if args.task == "k562" else "dream_rnn"
    oracle_id = args.oracle_id_stamp or oracle_type

    args.out.parent.mkdir(parents=True, exist_ok=True)

    pool_seqs, pool_labels = None, None
    if _needs_pool(args.reservoir):
        if args.background_cache is not None:
            print(f"Loading background pool from {args.background_cache}...", flush=True)
            pool_seqs, pool_labels = _load_background_cache(args.background_cache)
        else:
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
        "background_cache": str(args.background_cache) if args.background_cache else None,
        "label_mean": float(np.mean(labels)),
        "label_std": float(np.std(labels)),
    }
    np.savez_compressed(
        args.out,
        sequences=np.array(seqs, dtype=object),
        oracle_labels=labels,
        oracle_id=np.array(oracle_id),
        metadata=json.dumps(metadata),
    )
    print(
        f"Wrote {args.out}  n={len(seqs):,}  mean={metadata['label_mean']:.3f} "
        f"std={metadata['label_std']:.3f}  oracle_id={oracle_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
