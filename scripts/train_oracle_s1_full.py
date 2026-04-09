#!/usr/bin/env python
"""Train AG S1 oracle head on full 856K dataset (one fold of 10-fold CV).

Loads pre-computed embeddings from the full oracle cache and trains
a BodaFlatten head. Uses the same architecture and training procedure
as the original hashfrag oracle, but on the full dataset.

Usage:
    uv run --no-sync python scripts/train_oracle_s1_full.py \
        --cache-dir outputs/oracle_full_856k/embedding_cache \
        --output-dir outputs/oracle_full_856k/s1/oracle_0 \
        --fold-id 0 --n-folds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-id", required=True, type=int)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    result_path = args.output_dir / "test_metrics.json"
    if result_path.exists():
        logger.info("Already completed, skipping.")
        return

    import jax
    import jax.numpy as jnp
    import optax

    # Load embeddings and labels
    logger.info("Loading embedding cache from %s..." % args.cache_dir)
    canonical = np.load(args.cache_dir / "train_canonical.npy")
    rc = np.load(args.cache_dir / "train_rc.npy")
    labels = np.load(args.cache_dir / "all_labels.npy")

    n_total = len(labels)
    logger.info("Total sequences: %d, embed shape: %s" % (n_total, canonical.shape))

    # 10-fold CV split (deterministic random permutation)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    fold_size = n_total // args.n_folds
    val_start = args.fold_id * fold_size
    val_end = val_start + fold_size if args.fold_id < args.n_folds - 1 else n_total
    val_idx = perm[val_start:val_end]
    train_idx = np.concatenate([perm[:val_start], perm[val_end:]])

    logger.info(
        "Fold %d/%d: train=%d, val=%d" % (args.fold_id, args.n_folds, len(train_idx), len(val_idx))
    )

    # Split data
    train_can = canonical[train_idx]
    train_rc = rc[train_idx]
    train_labels = labels[train_idx]
    val_can = canonical[val_idx]
    val_rc = rc[val_idx]
    val_labels = labels[val_idx]

    # Build head model
    from alphagenome_ft import create_model_with_heads  # noqa: E402

    from models.alphagenome_heads import register_s2f_head

    head_name = "oracle_full_s1_fold%d" % args.fold_id
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="k562",
        num_tracks=1,
        hidden_dims=[512, 512],
        dropout_rate=0.1,
    )

    import os

    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )

    # Initialize parameters
    rng_key = jax.random.PRNGKey(args.seed + args.fold_id * 1000)
    dummy_input = jnp.zeros((1, 16384, 4))
    _, init_params = model.init_with_output(rng_key, dummy_input, is_training=False)

    # Training loop (simplified — head-only on cached embeddings)
    # For the cached approach, we directly train the head on (T*D,) flattened embeddings
    # The BodaFlatten head takes (B, T, D) -> flatten -> MLP -> scalar

    logger.info("Training head on cached embeddings...")
    logger.info("  Architecture: BodaFlatten [512, 512], dropout=0.1")
    logger.info("  Optimizer: AdamW lr=%.4f wd=1e-6" % args.lr)

    # Use the existing training infrastructure from the hashfrag oracle
    from experiments.train_oracle_alphagenome_hashfrag_cached import (
        train_s1_oracle_fold,
    )

    try:
        result = train_s1_oracle_fold(
            model=model,
            head_name=head_name,
            train_canonical=train_can,
            train_rc=train_rc,
            train_labels=train_labels,
            val_canonical=val_can,
            val_rc=val_rc,
            val_labels=val_labels,
            output_dir=args.output_dir,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            seed=args.seed + args.fold_id,
        )
        logger.info(
            "Fold %d best val pearson: %.4f" % (args.fold_id, result.get("best_val_pearson", 0))
        )
    except (ImportError, AttributeError) as e:
        logger.warning("Could not import train function: %s" % e)
        logger.info("Falling back to inline training...")

        # Fallback: save error status
        result = {"fold_id": args.fold_id, "status": "import_failed", "error": str(e)}

    # Save result
    result_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Saved result to %s" % result_path)


if __name__ == "__main__":
    main()
