"""Generate + label ONE shard of a strategy's master pool (#54 relabel+expansion).

Reuses generate_labeled_pools' building blocks so sequence generation + oracle
labeling stay in one place, and always labels with the canonical full856k_clean
AG_S2 oracle (via exp1_1._load_oracle("k562","ag_s2")). Idempotent: skips if the
shard npz already exists, so a SLURM array can be resubmitted to fill gaps.

One array task = one (strategy, shard). See scripts/master_pool_io.build_manifest_rows.

Usage::
    uv run --no-sync python scripts/generate_master_pool.py \
        --task k562 --reservoir prm_5pct \
        --target 5000000 --n-shards 10 --shard 3 --mode seed --seed 42
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
    _generate_sequences,
    _label_sequences,
    _load_pool_sequences,
)
from scripts.generate_reservoir_cache import _needs_pool  # noqa: E402
from scripts.master_pool_io import shard_path, shard_seed, shard_size_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="k562", choices=["k562", "yeast"])
    ap.add_argument("--reservoir", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--mode", choices=["seed", "index"], default="seed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()

    out = shard_path(args.task, args.reservoir, args.shard, args.out_root)
    if out.exists():
        print(f"[skip] shard already exists: {out}", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    from experiments.exp1_1_scaling import _load_oracle

    pool_seqs, pool_labels = None, None
    if _needs_pool(args.reservoir):
        print(f"Loading genomic pool for {args.task}...", flush=True)
        pool_seqs, pool_labels = _load_pool_sequences(args.task)
        print(f"  pool: {len(pool_seqs):,} sequences", flush=True)

    # Resolve this shard's slice/seed.
    if args.mode == "index":
        # Raw genomic: disjoint slice of the finite pool, no expansion past pool size.
        if pool_seqs is None:
            raise ValueError(f"index mode needs a pool but {args.reservoir} loaded none")
        n_pool = len(pool_seqs)
        per = -(-n_pool // args.n_shards)  # ceil
        lo, hi = args.shard * per, min((args.shard + 1) * per, n_pool)
        if lo >= hi:
            print(f"[skip] empty index slice [{lo}:{hi}] of pool size {n_pool}", flush=True)
            return
        pool_seqs = pool_seqs[lo:hi]
        pool_labels = None if pool_labels is None else pool_labels[lo:hi]
        n_gen = hi - lo
        gen_seed = args.seed
        print(f"  index-mode slice [{lo}:{hi}] -> {n_gen:,} seqs", flush=True)
    else:
        n_gen = shard_size_for(args.target, args.shard)
        gen_seed = shard_seed(args.seed, args.shard)
        print(f"  seed-mode shard {args.shard}: n={n_gen:,} sub-seed={gen_seed}", flush=True)

    print(f"Loading canonical AG_S2 oracle (full856k_clean)...", flush=True)
    oracle = _load_oracle(args.task, oracle_type="ag_s2")

    t0 = time.perf_counter()
    print(f"Generating {n_gen:,} seqs reservoir={args.reservoir}...", flush=True)
    seqs = _generate_sequences(
        args.reservoir, n_gen, args.task, gen_seed, pool_seqs, pool_labels, oracle=oracle
    )
    print(f"  generated {len(seqs):,} in {time.perf_counter() - t0:.1f}s", flush=True)

    print("Labeling with oracle...", flush=True)
    t1 = time.perf_counter()
    labels = _label_sequences(oracle, seqs)
    dt = time.perf_counter() - t1
    print(
        f"  labeled {len(labels):,} in {dt:.1f}s ({len(labels) / max(dt, 1e-9):.1f} seq/s)",
        flush=True,
    )

    meta = {
        "task": args.task,
        "oracle_id": "full856k_clean",
        "reservoir": args.reservoir,
        "mode": args.mode,
        "shard": int(args.shard),
        "n_shards": int(args.n_shards),
        "gen_seed": int(gen_seed),
        "n": int(len(seqs)),
        "label_mean": float(np.mean(labels)),
        "label_std": float(np.std(labels)),
    }
    tmp = out.with_suffix(".npz.tmp")
    np.savez_compressed(
        tmp,
        sequences=np.array(seqs, dtype=object),
        oracle_labels=labels.astype(np.float32),
        oracle_id=np.array("full856k_clean"),
        metadata=json.dumps(meta),
    )
    tmp.rename(out)  # atomic publish so a killed job never leaves a half-written shard
    print(
        f"Wrote {out}  n={len(seqs):,}  mean={meta['label_mean']:.3f} std={meta['label_std']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
