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

    # Re-init head params for this fold (fresh random weights)
    from models.embedding_cache import reinit_head_params

    reinit_head_params(model, head_name, num_tokens=5, dim=1536, rng=args.seed + args.fold_id)

    # Build head-only predict/train functions
    from models.embedding_cache import build_head_only_predict_fn

    head_predict_fn = build_head_only_predict_fn(model, head_name)

    logger.info("Training head on cached embeddings...")
    logger.info("  Architecture: BodaFlatten [512, 512], dropout=0.1")
    logger.info("  Optimizer: AdamW lr=%.4f wd=1e-6" % args.lr)

    # Training loop: MSE loss on (canonical + RC) / 2 predictions
    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=1e-6)
    opt_state = optimizer.init(model._params)

    organism_index = jnp.zeros(args.batch_size, dtype=jnp.int32)

    @jax.jit
    def train_step(params, opt_state, emb_can, emb_rc, targets):
        def loss_fn(p):
            pred_can = head_predict_fn(p, emb_can, organism_index[: len(targets)])
            pred_rc = head_predict_fn(p, emb_rc, organism_index[: len(targets)])
            preds = (pred_can + pred_rc) / 2.0
            return jnp.mean((preds - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def predict_batch(params, emb_can, emb_rc):
        pred_can = head_predict_fn(params, emb_can, organism_index[: emb_can.shape[0]])
        pred_rc = head_predict_fn(params, emb_rc, organism_index[: emb_rc.shape[0]])
        return (pred_can + pred_rc) / 2.0

    # Training
    best_val_r = -1.0
    best_epoch = 0
    patience_counter = 0
    params = model._params

    for epoch in range(args.epochs):
        # Shuffle training data
        rng_epoch = np.random.default_rng(args.seed + args.fold_id + epoch)
        perm = rng_epoch.permutation(len(train_labels))

        epoch_losses = []
        for start in range(0, len(train_labels), args.batch_size):
            end = min(start + args.batch_size, len(train_labels))
            idx = perm[start:end]
            batch_can = jnp.array(train_can[idx].astype(np.float32))
            batch_rc = jnp.array(train_rc[idx].astype(np.float32))
            batch_labels = jnp.array(train_labels[idx])
            params, opt_state, loss = train_step(
                params, opt_state, batch_can, batch_rc, batch_labels
            )
            epoch_losses.append(float(loss))

        # Validate
        val_preds = []
        for start in range(0, len(val_labels), args.batch_size):
            end = min(start + args.batch_size, len(val_labels))
            batch_can = jnp.array(val_can[start:end].astype(np.float32))
            batch_rc = jnp.array(val_rc[start:end].astype(np.float32))
            pred = predict_batch(params, batch_can, batch_rc)
            val_preds.append(np.array(pred))
        val_preds = np.concatenate(val_preds)

        from scipy.stats import pearsonr

        val_r = float(pearsonr(val_preds, val_labels)[0])
        train_loss = np.mean(epoch_losses)

        if epoch % 5 == 0 or val_r > best_val_r:
            logger.info(
                "  Epoch %d: train_loss=%.4f val_r=%.4f%s"
                % (epoch, train_loss, val_r, " *" if val_r > best_val_r else "")
            )

        if val_r > best_val_r:
            best_val_r = val_r
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            import orbax.checkpoint as ocp

            ckpt_path = args.output_dir / "best_model" / "checkpoint"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_mgr = ocp.CheckpointManager(str(ckpt_path.parent))
            ckpt_mgr.save(0, args=ocp.args.StandardSave(params))
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                logger.info(
                    "  Early stopping at epoch %d (best=%d, val_r=%.4f)"
                    % (epoch, best_epoch, best_val_r)
                )
                break

    result = {
        "fold_id": args.fold_id,
        "best_val_pearson": best_val_r,
        "best_epoch": best_epoch,
        "head_arch": "boda-flatten-512-512",
        "n_train": len(train_labels),
        "n_val": len(val_labels),
    }

    # Save result
    result_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Saved result to %s" % result_path)


if __name__ == "__main__":
    main()
