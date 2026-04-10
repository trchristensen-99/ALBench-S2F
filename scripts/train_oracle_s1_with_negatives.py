#!/usr/bin/env python
"""Train AG S1 oracle head with synthetic negative augmentation.

Same as train_oracle_s1_full.py but adds synthetic negative sequences
(random, dinuc-shuffled, GC-matched) to the training data. The negatives
get embeddings computed on-the-fly or cached, then the head learns to
predict low activity for them.

Since we can't cache embeddings for 150K synthetic sequences easily,
we use a simpler approach: we add the negatives DIRECTLY to the head
training by computing their AG embeddings and caching them first.

Usage:
    uv run --no-sync python scripts/train_oracle_s1_with_negatives.py \
        --cache-dir outputs/oracle_full_856k/embedding_cache \
        --negatives-dir data/synthetic_negatives \
        --output-dir outputs/oracle_with_negatives/s1/oracle_0 \
        --fold-id 0 --neg-type random --neg-fraction 0.1
"""

from __future__ import annotations

import argparse
import csv
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


def load_negatives(neg_dir: Path, neg_type: str, max_n: int = 50000):
    """Load synthetic negative sequences and labels."""
    type_to_file = {
        "random": "random_negatives.tsv",
        "dinuc_shuffled": "dinuc_shuffled_negatives.tsv",
        "gc_matched": "gc_matched_negatives.tsv",
        "all": "all_negatives.tsv",
    }
    filepath = neg_dir / type_to_file[neg_type]
    seqs, labels = [], []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seqs.append(row["sequence"])
            labels.append(float(row["K562_log2FC"]))
            if len(seqs) >= max_n:
                break
    return seqs, np.array(labels, dtype=np.float32)


def encode_sequences_ag(seqs, batch_size=64):
    """Compute AG embeddings for sequences (canonical + RC)."""
    from experiments.exp1_1_scaling import _encode_sequences_for_ag

    logger.info("Computing AG embeddings for %d sequences...", len(seqs))
    canonical, rc = [], []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        can_batch, rc_batch = _encode_sequences_for_ag(batch)
        canonical.append(can_batch)
        rc.append(rc_batch)
        if (i // batch_size) % 50 == 0:
            logger.info("  Encoded %d/%d", min(i + batch_size, len(seqs)), len(seqs))
    return np.concatenate(canonical), np.concatenate(rc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--negatives-dir", required=True, type=Path)
    parser.add_argument(
        "--neg-cache-dir",
        type=Path,
        default=None,
        help="Pre-cached negative embeddings (skip encoding)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-id", required=True, type=int)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument(
        "--neg-type",
        type=str,
        default="random",
        choices=["random", "dinuc_shuffled", "gc_matched", "all"],
    )
    parser.add_argument(
        "--neg-fraction",
        type=float,
        default=0.1,
        help="Fraction of negatives relative to training size",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.output_dir / "test_metrics.json"
    if result_path.exists():
        logger.info("Already completed, skipping.")
        return

    import jax
    import jax.numpy as jnp
    import optax

    # Load original embeddings and labels
    logger.info("Loading original embedding cache from %s...", args.cache_dir)
    canonical = np.load(args.cache_dir / "train_canonical.npy")
    rc = np.load(args.cache_dir / "train_rc.npy")
    labels = np.load(args.cache_dir / "all_labels.npy")
    n_original = len(labels)
    logger.info("Original: %d sequences, embed shape: %s", n_original, canonical.shape)

    # Load or compute negative embeddings
    neg_seqs, neg_labels = load_negatives(args.negatives_dir, args.neg_type)
    n_neg_target = int(n_original * args.neg_fraction)
    n_neg = min(n_neg_target, len(neg_seqs))
    neg_seqs = neg_seqs[:n_neg]
    neg_labels = neg_labels[:n_neg]
    logger.info(
        "Negatives: %d sequences (type=%s, fraction=%.2f)", n_neg, args.neg_type, args.neg_fraction
    )

    neg_cache_dir = args.neg_cache_dir
    if neg_cache_dir is None:
        neg_cache_dir = args.output_dir / "neg_embed_cache"
    neg_cache_dir.mkdir(parents=True, exist_ok=True)

    neg_can_path = neg_cache_dir / f"neg_{args.neg_type}_canonical.npy"
    neg_rc_path = neg_cache_dir / f"neg_{args.neg_type}_rc.npy"

    if neg_can_path.exists() and neg_rc_path.exists():
        logger.info("Loading cached negative embeddings...")
        neg_can = np.load(neg_can_path)[:n_neg]
        neg_rc = np.load(neg_rc_path)[:n_neg]
    else:
        neg_can, neg_rc = encode_sequences_ag(neg_seqs)
        np.save(neg_can_path, neg_can)
        np.save(neg_rc_path, neg_rc)
        logger.info("Cached negative embeddings to %s", neg_cache_dir)

    # Combine original + negatives
    all_canonical = np.concatenate([canonical, neg_can])
    all_rc = np.concatenate([rc, neg_rc])
    all_labels = np.concatenate([labels, neg_labels])
    n_total = len(all_labels)
    logger.info("Combined: %d total (%d original + %d negatives)", n_total, n_original, n_neg)

    # 10-fold CV split (same permutation as original, negatives go to all folds' train)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_original)  # Only permute original sequences
    fold_size = n_original // args.n_folds
    val_start = args.fold_id * fold_size
    val_end = val_start + fold_size if args.fold_id < args.n_folds - 1 else n_original
    val_idx = perm[val_start:val_end]
    train_idx_orig = np.concatenate([perm[:val_start], perm[val_end:]])

    # Add all negatives to training (not validation - we want val to be pure)
    neg_idx = np.arange(n_original, n_total)
    train_idx = np.concatenate([train_idx_orig, neg_idx])

    logger.info(
        "Fold %d/%d: train=%d (orig=%d + neg=%d), val=%d",
        args.fold_id,
        args.n_folds,
        len(train_idx),
        len(train_idx_orig),
        len(neg_idx),
        len(val_idx),
    )

    # Split data
    train_can = all_canonical[train_idx]
    train_rc = all_rc[train_idx]
    train_labels = all_labels[train_idx]
    val_can = all_canonical[val_idx]
    val_rc = all_rc[val_idx]
    val_labels = all_labels[val_idx]

    # Build head model (same as original)
    import os

    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head

    head_name = "oracle_neg_s1_fold%d" % args.fold_id
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="k562",
        num_tracks=1,
        hidden_dims=[512, 512],
        dropout_rate=0.1,
    )

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

    from models.embedding_cache import build_head_only_predict_fn, reinit_head_params

    reinit_head_params(model, head_name, num_tokens=5, dim=1536, rng=args.seed + args.fold_id)
    head_predict_fn = build_head_only_predict_fn(model, head_name)

    # Training loop (identical to original)
    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=1e-6)
    opt_state = optimizer.init(model._params)

    @jax.jit
    def train_step(params, opt_state, emb_can, emb_rc, targets):
        def loss_fn(p):
            org_idx = jnp.zeros(targets.shape[0], dtype=jnp.int32)
            pred_can = head_predict_fn(p, emb_can, org_idx)
            pred_rc = head_predict_fn(p, emb_rc, org_idx)
            preds = (pred_can + pred_rc) / 2.0
            return jnp.mean((preds - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def predict_batch(params, emb_can, emb_rc):
        pred_can = head_predict_fn(params, emb_can, jnp.zeros(emb_can.shape[0], dtype=jnp.int32))
        pred_rc = head_predict_fn(params, emb_rc, jnp.zeros(emb_rc.shape[0], dtype=jnp.int32))
        return (pred_can + pred_rc) / 2.0

    best_val_r = -1.0
    best_epoch = 0
    patience_counter = 0
    params = model._params

    for epoch in range(args.epochs):
        rng_epoch = np.random.default_rng(args.seed + args.fold_id + epoch)
        epoch_perm = rng_epoch.permutation(len(train_labels))
        epoch_losses = []
        for start in range(0, len(train_labels), args.batch_size):
            end = min(start + args.batch_size, len(train_labels))
            idx = epoch_perm[start:end]
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
            val_preds.append(np.array(pred).reshape(-1)[: end - start])
        val_preds = np.concatenate(val_preds)

        from scipy.stats import pearsonr

        val_r = float(pearsonr(val_preds, val_labels)[0])
        train_loss = np.mean(epoch_losses)

        if epoch % 5 == 0 or val_r > best_val_r:
            logger.info(
                "  Epoch %d: train_loss=%.4f val_r=%.4f%s",
                epoch,
                train_loss,
                val_r,
                " *" if val_r > best_val_r else "",
            )

        if val_r > best_val_r:
            best_val_r = val_r
            best_epoch = epoch
            patience_counter = 0
            import orbax.checkpoint as ocp

            ckpt_dir = (args.output_dir / "best_model").resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_mgr = ocp.CheckpointManager(str(ckpt_dir))
            ckpt_mgr.save(0, args=ocp.args.StandardSave(params))
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                logger.info(
                    "  Early stopping at epoch %d (best=%d, val_r=%.4f)",
                    epoch,
                    best_epoch,
                    best_val_r,
                )
                break

    result = {
        "fold_id": args.fold_id,
        "best_val_pearson": best_val_r,
        "best_epoch": best_epoch,
        "neg_type": args.neg_type,
        "neg_fraction": args.neg_fraction,
        "n_original": n_original,
        "n_negatives": n_neg,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    result_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Saved result to %s", result_path)


if __name__ == "__main__":
    main()
