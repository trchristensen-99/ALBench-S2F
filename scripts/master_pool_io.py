"""Shared IO + design spec for the #54 master-pool relabel + expansion.

A *master pool* is, per reservoir strategy, a single large set of sequences
generated ONCE and labeled ONCE with the canonical full856k_clean AG_S2 oracle.
Scaling-curve subsets of size D are drawn later via *seeded random sampling*
(`draw_subset`) keyed on the dataset seed, so a curve is never biased by a fixed
prefix of the pool (the failure mode of the legacy per-(strategy,D,seed) caches).

Layout on disk::

    outputs/master_pools/{task}/{strategy}/
        shard_0000.npz   # sequences (object) + oracle_labels (float32) + meta
        shard_0001.npz
        ...
        MANIFEST.json    # written after all shards land; totals + provenance

Two shard modes (see scripts/generate_master_pool.py):
  - "seed":  strategy draws parents WITH replacement or generates fresh seqs, so
             shard k just uses a distinct sub-seed. Expandable past the genomic cap.
  - "index": raw `genomic` only — finite real-K562 pool, NOT prefix-stable, so each
             shard takes a disjoint slice of the genomic pool (no expansion).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Sequences per shard. 500k @ ~300 seq/s on a V100 ~= 28 min/shard, comfortably
# inside a slow_nice slot and small enough to checkpoint/resume per shard.
SHARD_SIZE = 500_000

# Default expansion target for derived/synthetic strategies. genomic is overridden
# to its true pool size at generation time (it cannot expand past the real seqs).
DEFAULT_TARGET = 5_000_000

# Per-strategy overrides: (mode, target). Anything not listed defaults to
# ("seed", DEFAULT_TARGET) — valid only for strategies that draw parents WITH
# replacement or generate fresh seqs (random, gc_matched, prm_*, recombination_*,
# evoaug_*, motif_*), which can genuinely expand past the genomic cap.
#
# Finite-pool strategies CANNOT expand past the real K562 pool (~618k chr-split+alt):
#   - genomic: "index" mode, target clamped to pool size by the generator.
#   - activity_stratified_oracle: stratified draw from the finite labeled pool; cap
#     at the pool size (final number confirmed by a capacity probe before launch).
# dinuc_shuffle is seed-mode but capped lower (strict dinuc-preserving permutations
# are the soft ceiling the user flagged: ~618k-1.25M).
GENOMIC_POOL_CAP = 618_000
STRATEGY_SPECS: dict[str, tuple[str, int]] = {
    "genomic": ("index", DEFAULT_TARGET),  # clamped to actual pool size at gen time
    "activity_stratified_oracle": ("seed", GENOMIC_POOL_CAP),
    "dinuc_shuffle": ("seed", 1_250_000),
}

# The full strategy menu to relabel. Mirrors generate_labeled_pools.ALL_RESERVOIRS;
# kept explicit here so a launcher can iterate without importing the heavy module.
MASTER_STRATEGIES: list[str] = [
    "random",
    "genomic",
    "gc_matched",
    "dinuc_shuffle",
    "snv",
    "prm_1pct",
    "prm_5pct",
    "prm_10pct",
    "prm_20pct",
    "prm_50pct",
    "prm_uniform_1_10",
    "recombination_uniform",
    "recombination_2pt",
    "evoaug_structural",
    "evoaug_heavy",
    "evoaug_prior",
    "motif_density_2",
    "motif_density_3",
    "motif_density_5",
    "motif_planted",
    "motif_grammar",
    "activity_stratified_oracle",
]


def spec_for(strategy: str) -> tuple[str, int]:
    return STRATEGY_SPECS.get(strategy, ("seed", DEFAULT_TARGET))


def master_dir(task: str, strategy: str, out_root: Path | None = None) -> Path:
    root = out_root or (REPO / "outputs" / "master_pools")
    return root / task / strategy


def shard_path(task: str, strategy: str, shard: int, out_root: Path | None = None) -> Path:
    return master_dir(task, strategy, out_root) / f"shard_{shard:04d}.npz"


def n_shards_for(target: int) -> int:
    return max(1, math.ceil(target / SHARD_SIZE))


def shard_size_for(target: int, shard: int) -> int:
    """Size of shard `shard` so the shards sum exactly to `target`."""
    n = n_shards_for(target)
    base = target // n
    rem = target - base * n
    return base + (1 if shard < rem else 0)


def shard_seed(base_seed: int, shard: int) -> int:
    """Distinct, collision-free sub-seed per shard for seed-mode generation."""
    return base_seed * 100_000 + shard


def build_manifest_rows(strategies: list[str] | None = None) -> list[dict]:
    """Flat (strategy, shard) task list for a SLURM array.

    Each row carries everything generate_master_pool.py needs for one array task.
    Row index == SLURM_ARRAY_TASK_ID.
    """
    rows: list[dict] = []
    for strat in strategies or MASTER_STRATEGIES:
        mode, target = spec_for(strat)
        n = n_shards_for(target)
        for k in range(n):
            rows.append(
                {
                    "strategy": strat,
                    "mode": mode,
                    "target": target,
                    "n_shards": n,
                    "shard": k,
                    "shard_size": shard_size_for(target, k),
                }
            )
    return rows


def load_master(
    task: str, strategy: str, out_root: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all shards of a strategy's master pool into (seqs, labels)."""
    d = master_dir(task, strategy, out_root)
    shards = sorted(d.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No master-pool shards under {d}")
    seqs_parts, lab_parts = [], []
    for sp in shards:
        with np.load(sp, allow_pickle=True) as z:
            seqs_parts.append(z["sequences"])
            lab_parts.append(z["oracle_labels"])
    return np.concatenate(seqs_parts), np.concatenate(lab_parts).astype(np.float32)


def draw_subset(
    seqs: np.ndarray, labels: np.ndarray, D: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded random subset of size D (without replacement) from a master pool.

    Keyed on the dataset `seed` so two seeds give genuinely different subsets and a
    scaling curve is never the fixed prefix of the pool. If D >= pool size, returns
    the whole pool (shuffled by `seed`).
    """
    n = len(seqs)
    rng = np.random.default_rng(seed)
    if D >= n:
        idx = rng.permutation(n)
    else:
        idx = rng.choice(n, size=D, replace=False)
    return seqs[idx], labels[idx].astype(np.float32)


def write_manifest_summary(
    task: str, strategy: str, oracle_id: str, out_root: Path | None = None
) -> dict:
    """After all shards land, summarize totals + provenance into MANIFEST.json."""
    seqs, labels = load_master(task, strategy, out_root)
    summary = {
        "task": task,
        "strategy": strategy,
        "oracle_id": oracle_id,
        "n_total": int(len(seqs)),
        "label_mean": float(np.mean(labels)),
        "label_std": float(np.std(labels)),
        "n_shards": len(sorted(master_dir(task, strategy, out_root).glob("shard_*.npz"))),
    }
    out = master_dir(task, strategy, out_root) / "MANIFEST.json"
    out.write_text(json.dumps(summary, indent=2))
    return summary
