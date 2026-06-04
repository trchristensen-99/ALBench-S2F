#!/usr/bin/env python
"""
Pre-compute HashFrag splits for K562 dataset.

Creates homology-aware train/validation/test splits and caches them as .npy files.
This must be run once before K562 experiments.

Usage:
    python scripts/create_hashfrag_splits.py
    python scripts/create_hashfrag_splits.py --threshold 70 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.k562 import K562Dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute HashFrag splits for K562 dataset")
    parser.add_argument("--data-dir", type=str, default="./data/k562")
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "hashfrag_splits"

    cache_files = [output_dir / f"{s}_indices.npy" for s in ("train", "pool", "val", "test")]

    if all(f.exists() for f in cache_files) and not args.force:
        logger.info("HashFrag splits already exist at %s. Use --force to recompute.", output_dir)
        return

    logger.info("=" * 70)
    logger.info("HashFrag Split Creation")
    logger.info("Data: %s | Output: %s | Threshold: %s", data_dir, output_dir, args.threshold)
    logger.info("Split: 80%% train+pool / 10%% val / 10%% test")
    logger.info("  Train pool → 100K train + remainder pool")
    logger.info("=" * 70)
    logger.info("⚠️  This will take several hours (BLAST on ~367K sequences)")
    logger.info("Consider: sbatch scripts/slurm/create_hashfrag_splits.sh")

    start = time.time()

    try:
        _dataset = K562Dataset(
            data_path=str(data_dir),
            split="train",
            use_hashfrag=True,
            hashfrag_threshold=args.threshold,
            hashfrag_cache_dir=str(output_dir),
        )

        elapsed = time.time() - start
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

        logger.info("✓ HashFrag splits created in %dh %dm %ds", h, m, s)
        logger.info("Saved to: %s", output_dir)

        import numpy as np

        total = 0
        for name in ("train", "pool", "val", "test"):
            f = output_dir / f"{name}_indices.npy"
            if f.exists():
                n = len(np.load(f))
                total += n
                logger.info("  %s: %d sequences", name.capitalize(), n)
        logger.info("  Total: %d sequences", total)

    except Exception as e:
        logger.error("Failed: %s", e)
        logger.error("Check: BLAST+ installed? HashFrag in PATH?")
        sys.exit(1)


if __name__ == "__main__":
    main()
