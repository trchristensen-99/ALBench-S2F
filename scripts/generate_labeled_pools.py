#!/usr/bin/env python
"""Pre-generate large labeled pools for Experiment 1.1.

Generates 500K sequences per (task, oracle, reservoir) combo using the
reservoir strategy, labels them with the oracle ensemble, and saves to a
single NPZ file.  Training jobs then load and subsample for each n_train
value, ensuring strict subsets (n=5K is a subset of n=10K, etc.).

Output layout::

    outputs/labeled_pools/{task}/{oracle}/{reservoir}/pool.npz
        sequences: (500_000,) object array of strings
        labels:    (500_000,) float32 array
        metadata:  JSON string with generation info

Usage::

    # Single combo
    uv run --no-sync python scripts/generate_labeled_pools.py \
        --task k562 --oracle ag --reservoir random --pool-size 500000

    # Multiple reservoirs
    uv run --no-sync python scripts/generate_labeled_pools.py \
        --task k562 --oracle ag \
        --reservoir random genomic snv recombination_uniform

    # All Phase 1 reservoirs for K562
    uv run --no-sync python scripts/generate_labeled_pools.py \
        --task k562 --oracle ag --reservoir all

    # Dry run (show what would be generated)
    uv run --no-sync python scripts/generate_labeled_pools.py \
        --task k562 --oracle ag --reservoir all --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# All Phase 1 reservoirs (order matches the paper)
ALL_RESERVOIRS = [
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
    "motif_density_2",
    "motif_density_3",
    "motif_density_5",
    "motif_planted",
    "motif_grammar",
    "motif_clustering",
    "motif_clustering_mutant",
    "activity_stratified_oracle",
]

# Per-strategy pool size caps (genomic-derived limited to pool size).
# With --chr-split --include-alt-alleles, K562 pool grows to ~618K.
# These caps are for hashfrag (296K); chr-split+alt auto-expands in generate_pool.
_MAX_POOL_SIZE_HASHFRAG = {
    "genomic": 296_000,
    "gc_matched": 296_000,
    "dinuc_shuffle": 296_000,
    "snv": 296_000,
    "activity_stratified": 296_000,
    "activity_stratified_oracle": 296_000,
    "motif_clustering": 296_000,
    "motif_clustering_mutant": 296_000,
}
_MAX_POOL_SIZE_CHRSPLIT_ALT = {
    "genomic": 618_000,
    "gc_matched": 618_000,
    "dinuc_shuffle": 618_000,
    "snv": 618_000,
    "activity_stratified": 618_000,
    "activity_stratified_oracle": 618_000,
    "motif_clustering": 618_000,
    "motif_clustering_mutant": 618_000,
}

# Reservoirs that need pool (genomic) sequences loaded
_NEEDS_POOL = {
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
    "activity_stratified",
    "activity_stratified_oracle",
    "motif_density_2",
    "motif_density_3",
    "motif_density_5",
    "motif_clustering",
    "motif_clustering_mutant",
}

DEFAULT_POOL_SIZE = 500_000
ORACLE_BATCH_SIZE = 2048  # sequences per oracle.predict() call


def _load_reservoir(name: str, seed: int):
    """Load a reservoir sampler from its YAML config."""
    import yaml

    cfg_path = REPO / "configs" / "reservoir" / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No reservoir config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    target = cfg.pop("_target_")
    for k, v in cfg.items():
        if v == "${seed}":
            cfg[k] = seed

    parts = target.rsplit(".", 1)
    mod = __import__(parts[0], fromlist=[parts[1]])
    cls = getattr(mod, parts[1])
    return cls(**cfg)


def _load_pool_sequences(task: str, chr_split: bool = False, include_alt_alleles: bool = False):
    """Load genomic pool sequences for the task."""
    if task == "k562":
        from data.k562 import K562Dataset

        kwargs = {"data_path": str(REPO / "data" / "k562"), "split": "train"}
        if chr_split:
            kwargs["use_hashfrag"] = False
            kwargs["use_chromosome_fallback"] = True
        if include_alt_alleles:
            kwargs["include_alt_alleles"] = True
        ds = K562Dataset(**kwargs)
        return list(ds.sequences), ds.labels.astype(np.float32)
    else:
        from data.yeast import YeastDataset

        ds = YeastDataset(
            data_path=str(REPO / "data" / "yeast"),
            split="train",
            context_mode="dream150",
        )
        return list(ds.sequences), ds.labels.astype(np.float32)


def _generate_sequences(
    reservoir_name: str,
    n: int,
    task: str,
    seed: int,
    pool_seqs: list[str] | None,
    pool_labels: np.ndarray | None,
    oracle=None,
) -> list[str]:
    """Generate n sequences using the given reservoir strategy."""
    res = _load_reservoir(reservoir_name, seed=seed)

    if reservoir_name == "random":
        seqs, _ = res.generate(n, task=task)
    elif reservoir_name == "dinuc_shuffle":
        seqs, _ = res.generate(n, task=task, method="dinuc_shuffle", reference_sequences=pool_seqs)
    elif reservoir_name == "genomic":
        seqs, _ = res.generate(n, pool_sequences=pool_seqs, pool_labels=pool_labels)
    elif reservoir_name == "gc_matched":
        seqs, _ = res.generate(n, pool_sequences=pool_seqs, task=task)
    elif reservoir_name.startswith("prm") or reservoir_name == "snv":
        seqs, _ = res.generate(n, base_sequences=pool_seqs, task=task)
    elif reservoir_name in (
        "recombination_uniform",
        "recombination_2pt",
        "evoaug_structural",
        "evoaug_heavy",
    ):
        seqs, _ = res.generate(n, base_sequences=pool_seqs, task=task)
    elif reservoir_name == "activity_stratified":
        if oracle is None:
            raise ValueError("activity_stratified requires oracle for scoring")
        seqs, _ = res.generate(n, pool_sequences=pool_seqs, student_model=oracle)
    elif reservoir_name == "activity_stratified_oracle":
        seqs, _ = res.generate(n, pool_sequences=pool_seqs, pool_labels=pool_labels)
    elif reservoir_name.startswith("motif_density") or reservoir_name == "motif_planted":
        seqs, _ = res.generate(n, task=task)
    elif reservoir_name == "motif_grammar":
        seqs, _ = res.generate(n, task=task)
    elif reservoir_name.startswith("motif_clustering"):
        seqs, _ = res.generate(n, pool_sequences=pool_seqs, task=task)
    else:
        # Fallback: try task-only
        seqs, _ = res.generate(n, task=task)

    return seqs


def _label_sequences(
    oracle, sequences: list[str], batch_size: int = ORACLE_BATCH_SIZE
) -> np.ndarray:
    """Label sequences with oracle in batches (memory-safe)."""
    n = len(sequences)
    all_labels = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = sequences[start:end]
        labels = oracle.predict(batch)
        all_labels.append(labels)
        if (start // batch_size) % 10 == 0:
            logger.info(f"  Labeled {end:,}/{n:,} sequences ({100 * end / n:.1f}%)")
    return np.concatenate(all_labels).astype(np.float32)


def generate_pool(
    task: str,
    oracle_type: str,
    reservoir_name: str,
    pool_size: int,
    seed: int,
    output_dir: Path,
    dry_run: bool = False,
    chr_split: bool = False,
    include_alt_alleles: bool = False,
) -> Path:
    """Generate and label one pool, saving to NPZ.

    Returns the path to the saved NPZ file.
    """
    # Import oracle loader from exp1_1
    from experiments.exp1_1_scaling import _load_oracle

    # Cap pool size for genomic-derived strategies
    caps = (
        _MAX_POOL_SIZE_CHRSPLIT_ALT
        if (chr_split and include_alt_alleles)
        else _MAX_POOL_SIZE_HASHFRAG
    )
    max_pool = caps.get(reservoir_name, pool_size)
    if pool_size > max_pool:
        logger.info(f"Capping pool size from {pool_size:,} to {max_pool:,} for {reservoir_name}")
        pool_size = max_pool

    out_path = output_dir / "pool.npz"
    if out_path.exists():
        logger.info(f"SKIP (already exists): {out_path}")
        return out_path

    if dry_run:
        logger.info(f"DRY RUN: would generate {pool_size:,} seqs -> {out_path}")
        return out_path

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pool sequences if needed
    pool_seqs, pool_labels = None, None
    if reservoir_name in _NEEDS_POOL:
        logger.info(f"Loading genomic pool for {task}...")
        pool_seqs, pool_labels = _load_pool_sequences(
            task, chr_split=chr_split, include_alt_alleles=include_alt_alleles
        )
        logger.info(f"Pool loaded: {len(pool_seqs):,} sequences")

    # Load oracle
    logger.info(f"Loading oracle: task={task}, oracle_type={oracle_type}...")
    t0 = time.perf_counter()
    oracle = _load_oracle(task, oracle_type=oracle_type)
    logger.info(f"Oracle loaded in {time.perf_counter() - t0:.1f}s")

    # Generate sequences
    logger.info(f"Generating {pool_size:,} sequences with reservoir={reservoir_name}...")
    t0 = time.perf_counter()
    sequences = _generate_sequences(
        reservoir_name=reservoir_name,
        n=pool_size,
        task=task,
        seed=seed,
        pool_seqs=pool_seqs,
        pool_labels=pool_labels,
        oracle=oracle,
    )
    gen_time = time.perf_counter() - t0
    logger.info(f"Generated {len(sequences):,} sequences in {gen_time:.1f}s")

    # Label with oracle
    logger.info(f"Labeling {len(sequences):,} sequences with oracle...")
    t0 = time.perf_counter()
    labels = _label_sequences(oracle, sequences)
    label_time = time.perf_counter() - t0
    logger.info(f"Labeling took {label_time:.1f}s")

    # Save
    metadata = {
        "task": task,
        "oracle_type": oracle_type,
        "reservoir": reservoir_name,
        "pool_size": len(sequences),
        "seed": seed,
        "generation_time_s": round(gen_time, 1),
        "labeling_time_s": round(label_time, 1),
        "label_mean": float(np.mean(labels)),
        "label_std": float(np.std(labels)),
        "label_min": float(np.min(labels)),
        "label_max": float(np.max(labels)),
    }

    np.savez_compressed(
        out_path,
        sequences=np.array(sequences, dtype=object),
        labels=labels,
        metadata=json.dumps(metadata),
    )
    file_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved pool to {out_path} ({file_mb:.1f} MB)")
    logger.info(f"  Labels: mean={metadata['label_mean']:.3f}, std={metadata['label_std']:.3f}")

    # Also save human-readable metadata
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    return out_path


def load_pool_subset(
    pool_path: Path,
    n_train: int,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Load a strict subset of n_train sequences from a pre-generated pool.

    The subset is deterministic: for a given seed, n=5K is always a prefix
    of n=10K, which is a prefix of n=50K, etc.  This is achieved by using
    the seed to generate a single permutation of the full pool, then taking
    the first n_train elements.

    Args:
        pool_path: Path to pool.npz file.
        n_train: Number of training sequences to select.
        seed: Random seed for the permutation (same seed = strict subsets).

    Returns:
        (sequences, labels) tuple.
    """
    data = np.load(pool_path, allow_pickle=True)
    all_sequences = data["sequences"]
    all_labels = data["labels"]
    pool_size = len(all_sequences)

    if n_train > pool_size:
        raise ValueError(
            f"Requested n_train={n_train:,} but pool only has "
            f"{pool_size:,} sequences. "
            f"Regenerate with --pool-size >= {n_train}"
        )

    # Generate a deterministic permutation of the full pool
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool_size)

    # Take the first n_train indices -> strict subsets
    idx = perm[:n_train]
    sequences = all_sequences[idx].tolist()
    labels = all_labels[idx].astype(np.float32)

    return sequences, labels


def main():
    parser = argparse.ArgumentParser(description="Pre-generate large labeled pools for exp1.1")
    parser.add_argument(
        "--task",
        required=True,
        choices=["k562", "yeast"],
        help="Task/dataset",
    )
    parser.add_argument(
        "--oracle",
        required=True,
        choices=["ag", "ag_s2", "dream_rnn", "legnet", "default"],
        help="Oracle type for labeling",
    )
    parser.add_argument(
        "--reservoir",
        nargs="+",
        required=True,
        help="Reservoir strategy names, or 'all' for all Phase 1 reservoirs",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help="Number of sequences per pool (default: 500000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=REPO / "outputs" / "labeled_pools",
        help="Base output directory",
    )
    parser.add_argument(
        "--chr-split",
        action="store_true",
        help="Use chromosome-based splits (train=non-chr7/13, test=chr7+13). "
        "Increases K562 genomic pool to ~316K (ref) or ~618K (ref+alt).",
    )
    parser.add_argument(
        "--include-alt-alleles",
        action="store_true",
        help="Include ref+alt alleles in pool (K562 only). "
        "Combined with --chr-split gives ~618K genomic training sequences.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running",
    )
    args = parser.parse_args()

    # Expand "all" to all Phase 1 reservoirs
    reservoirs = args.reservoir
    if "all" in reservoirs:
        reservoirs = ALL_RESERVOIRS
        logger.info(f"Expanded 'all' to {len(reservoirs)} reservoirs")

    # Resolve oracle type
    oracle_type = args.oracle
    if oracle_type == "default":
        oracle_type = "ag" if args.task == "k562" else "dream_rnn"

    logger.info(f"Task: {args.task}")
    logger.info(f"Oracle: {oracle_type}")
    logger.info(f"Reservoirs: {reservoirs}")
    logger.info(f"Pool size: {args.pool_size:,}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output base: {args.output_base}")
    logger.info("")

    total_start = time.perf_counter()
    for i, res_name in enumerate(reservoirs):
        sep = "=" * 60
        logger.info(f"\n{sep}")
        logger.info(f"[{i + 1}/{len(reservoirs)}] {args.task}/{oracle_type}/{res_name}")
        logger.info(sep)

        output_dir = args.output_base / args.task / oracle_type / res_name
        try:
            generate_pool(
                task=args.task,
                oracle_type=oracle_type,
                reservoir_name=res_name,
                pool_size=args.pool_size,
                seed=args.seed,
                output_dir=output_dir,
                dry_run=args.dry_run,
                chr_split=args.chr_split,
                include_alt_alleles=args.include_alt_alleles,
            )
        except Exception:
            logger.exception(f"FAILED: {res_name}")
            continue

    total_time = time.perf_counter() - total_start
    logger.info(f"\nAll done in {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
