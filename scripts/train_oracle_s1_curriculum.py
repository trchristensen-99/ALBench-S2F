#!/usr/bin/env python
"""Train AG S1 oracle with curriculum-based negative augmentation.

Avoids the bimodal label instability by:
  1. Phase 1: Train normally on original data for N epochs (warmup)
  2. Phase 2: Gradually mix in negatives with increasing fraction
  3. Phase 3: Train on full mixed dataset at lower LR

Also tests alternative approaches:
  - "finetune": Train to convergence on original, then fine-tune with negatives at low LR
  - "label_shift": Shift ALL labels so training mean is ~0 (negatives become less extreme)
  - "weighted": Use weighted MSE with lower weight on negatives

Usage:
    uv run --no-sync python scripts/train_oracle_s1_curriculum.py \
        --cache-dir outputs/oracle_full_856k/embedding_cache \
        --negatives-dir data/synthetic_negatives \
        --output-dir outputs/oracle_neg_curriculum/finetune/oracle_0 \
        --fold-id 0 --approach finetune
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_negatives(neg_dir, neg_type="dinuc_shuffled", max_n=50000):
    filepath = neg_dir / ("%s_negatives.tsv" % neg_type)
    seqs, labels = [], []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seqs.append(row["sequence"])
            labels.append(float(row["K562_log2FC"]))
            if len(seqs) >= max_n:
                break
    return seqs, np.array(labels, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--negatives-dir", required=True, type=Path)
    parser.add_argument("--neg-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-id", required=True, type=int)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument(
        "--approach",
        choices=["finetune", "label_shift", "weighted", "curriculum"],
        required=True,
    )
    parser.add_argument("--neg-fraction", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
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

    # Load embeddings
    logger.info("Loading embedding cache...")
    canonical = np.load(args.cache_dir / "train_canonical.npy")
    rc = np.load(args.cache_dir / "train_rc.npy")
    labels = np.load(args.cache_dir / "all_labels.npy")
    n_original = len(labels)

    # Load negative embeddings
    neg_cache = args.neg_cache_dir or Path("outputs/oracle_neg_augmentation/neg_embed_cache")
    neg_can = np.load(neg_cache / "neg_dinuc_shuffled_canonical.npy")
    neg_rc = np.load(neg_cache / "neg_dinuc_shuffled_rc.npy")
    neg_seqs, neg_labels_raw = load_negatives(
        args.negatives_dir, "dinuc_shuffled", int(n_original * args.neg_fraction)
    )
    n_neg = min(len(neg_labels_raw), neg_can.shape[0])
    neg_can = neg_can[:n_neg]
    neg_rc = neg_rc[:n_neg]
    neg_labels_raw = neg_labels_raw[:n_neg]
    logger.info("Loaded %d negatives", n_neg)

    # Apply approach-specific label transformations
    if args.approach == "label_shift":
        # Shift ALL labels so training mean is ~0
        # This makes negatives less extreme relative to positives
        shift = np.mean(labels)
        labels = labels - shift
        neg_labels = neg_labels_raw - shift
        logger.info("Label shift: subtracted %.4f from all labels", shift)
        logger.info("  Original labels: mean=%.4f → %.4f", shift, np.mean(labels))
        logger.info(
            "  Negative labels: mean=%.4f → %.4f", np.mean(neg_labels_raw), np.mean(neg_labels)
        )
    elif args.approach == "weighted":
        neg_labels = neg_labels_raw
    elif args.approach in ("finetune", "curriculum"):
        neg_labels = neg_labels_raw
    else:
        neg_labels = neg_labels_raw

    # CV split (same as original - only original sequences)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_original)
    fold_size = n_original // args.n_folds
    val_start = args.fold_id * fold_size
    val_end = val_start + fold_size if args.fold_id < args.n_folds - 1 else n_original
    val_idx = perm[val_start:val_end]
    train_idx = np.concatenate([perm[:val_start], perm[val_end:]])

    train_can = canonical[train_idx]
    train_rc = rc[train_idx]
    train_labels = labels[train_idx]
    val_can = canonical[val_idx]
    val_rc = rc[val_idx]
    val_labels_orig = np.load(args.cache_dir / "all_labels.npy")[
        val_idx
    ]  # always eval on original scale

    if args.approach == "label_shift":
        val_labels = val_labels_orig - shift
    else:
        val_labels = val_labels_orig

    n_train_orig = len(train_labels)
    logger.info(
        "Fold %d: train=%d, val=%d, neg=%d", args.fold_id, n_train_orig, len(val_idx), n_neg
    )

    # Build model
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head

    head_name = "oracle_curr_fold%d" % args.fold_id
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

    # Training functions
    def make_optimizer(lr):
        return optax.adamw(learning_rate=lr, weight_decay=1e-6)

    @jax.jit
    def train_step(params, opt_state, optimizer, emb_can, emb_rc, targets):
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
    def train_step_weighted(params, opt_state, optimizer, emb_can, emb_rc, targets, weights):
        def loss_fn(p):
            org_idx = jnp.zeros(targets.shape[0], dtype=jnp.int32)
            pred_can = head_predict_fn(p, emb_can, org_idx)
            pred_rc = head_predict_fn(p, emb_rc, org_idx)
            preds = (pred_can + pred_rc) / 2.0
            return jnp.mean(weights * (preds - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def predict_batch(params, emb_can, emb_rc):
        pred_can = head_predict_fn(params, emb_can, jnp.zeros(emb_can.shape[0], dtype=jnp.int32))
        pred_rc = head_predict_fn(params, emb_rc, jnp.zeros(emb_rc.shape[0], dtype=jnp.int32))
        return (pred_can + pred_rc) / 2.0

    def validate(params):
        preds = []
        for start in range(0, len(val_labels), args.batch_size):
            end = min(start + args.batch_size, len(val_labels))
            batch_can = jnp.array(val_can[start:end].astype(np.float32))
            batch_rc = jnp.array(val_rc[start:end].astype(np.float32))
            pred = predict_batch(params, batch_can, batch_rc)
            preds.append(np.array(pred).reshape(-1)[: end - start])
        preds = np.concatenate(preds)
        from scipy.stats import pearsonr

        return float(pearsonr(preds, val_labels)[0])

    def train_epoch(params, opt_state, optimizer, t_can, t_rc, t_labels, t_weights=None):
        rng_e = np.random.default_rng(args.seed + hash(str(params)[:20]) % 10000)
        epoch_perm = rng_e.permutation(len(t_labels))
        losses = []
        for start in range(0, len(t_labels), args.batch_size):
            end = min(start + args.batch_size, len(t_labels))
            idx = epoch_perm[start:end]
            b_can = jnp.array(t_can[idx].astype(np.float32))
            b_rc = jnp.array(t_rc[idx].astype(np.float32))
            b_lab = jnp.array(t_labels[idx])
            if t_weights is not None:
                b_w = jnp.array(t_weights[idx])
                params, opt_state, loss = train_step_weighted(
                    params, opt_state, optimizer, b_can, b_rc, b_lab, b_w
                )
            else:
                params, opt_state, loss = train_step(
                    params, opt_state, optimizer, b_can, b_rc, b_lab
                )
            losses.append(float(loss))
        return params, opt_state, np.mean(losses)

    params = model._params
    best_val_r = -1.0
    best_epoch = 0
    patience_counter = 0
    patience = 7

    # ═══════════════════════════════════════════
    # APPROACH-SPECIFIC TRAINING
    # ═══════════════════════════════════════════

    if args.approach == "finetune":
        # Phase 1: Train on original data to convergence
        logger.info("=== FINETUNE Phase 1: Original data only ===")
        optimizer = make_optimizer(args.lr)
        opt_state = optimizer.init(params)

        for epoch in range(args.epochs):
            params, opt_state, loss = train_epoch(
                params, opt_state, optimizer, train_can, train_rc, train_labels
            )
            val_r = validate(params)
            if epoch % 5 == 0 or val_r > best_val_r:
                logger.info(
                    "  P1 Epoch %d: loss=%.4f val_r=%.4f%s",
                    epoch,
                    loss,
                    val_r,
                    " *" if val_r > best_val_r else "",
                )
            if val_r > best_val_r:
                best_val_r = val_r
                best_epoch = epoch
                best_params = jax.tree.map(lambda x: x.copy(), params)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "  P1 early stop at epoch %d (best=%d, val_r=%.4f)",
                        epoch,
                        best_epoch,
                        best_val_r,
                    )
                    break

        # Phase 2: Fine-tune with negatives at 10x lower LR
        logger.info("=== FINETUNE Phase 2: Mixed data at low LR ===")
        params = best_params
        mixed_can = np.concatenate([train_can, neg_can])
        mixed_rc = np.concatenate([train_rc, neg_rc])
        mixed_labels = np.concatenate([train_labels, neg_labels])

        optimizer2 = make_optimizer(args.lr * 0.1)
        opt_state2 = optimizer2.init(params)
        patience_counter = 0
        p2_best_val_r = best_val_r

        for epoch in range(20):  # fewer epochs for fine-tuning
            params, opt_state2, loss = train_epoch(
                params, opt_state2, optimizer2, mixed_can, mixed_rc, mixed_labels
            )
            val_r = validate(params)
            if epoch % 2 == 0 or val_r > p2_best_val_r:
                logger.info(
                    "  P2 Epoch %d: loss=%.4f val_r=%.4f%s",
                    epoch,
                    loss,
                    val_r,
                    " *" if val_r > p2_best_val_r else "",
                )
            if val_r > p2_best_val_r:
                p2_best_val_r = val_r
                best_params = jax.tree.map(lambda x: x.copy(), params)
                best_epoch = epoch + args.epochs
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break

        best_val_r = max(best_val_r, p2_best_val_r)
        params = best_params

    elif args.approach == "label_shift":
        # Same as original but with shifted labels (mean=0)
        mixed_can = np.concatenate([train_can, neg_can])
        mixed_rc = np.concatenate([train_rc, neg_rc])
        mixed_labels = np.concatenate([train_labels, neg_labels])

        optimizer = make_optimizer(args.lr)
        opt_state = optimizer.init(params)

        for epoch in range(args.epochs):
            params, opt_state, loss = train_epoch(
                params, opt_state, optimizer, mixed_can, mixed_rc, mixed_labels
            )
            val_r = validate(params)
            if epoch % 5 == 0 or val_r > best_val_r:
                logger.info(
                    "  Epoch %d: loss=%.4f val_r=%.4f%s",
                    epoch,
                    loss,
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
                ocp.CheckpointManager(str(ckpt_dir)).save(0, args=ocp.args.StandardSave(params))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    elif args.approach == "weighted":
        # Mixed data with lower weight on negatives
        mixed_can = np.concatenate([train_can, neg_can])
        mixed_rc = np.concatenate([train_rc, neg_rc])
        mixed_labels = np.concatenate([train_labels, neg_labels])
        weights = np.concatenate(
            [
                np.ones(n_train_orig, dtype=np.float32),
                np.full(n_neg, 0.1, dtype=np.float32),  # 10% weight on negatives
            ]
        )

        optimizer = make_optimizer(args.lr)
        opt_state = optimizer.init(params)

        for epoch in range(args.epochs):
            params, opt_state, loss = train_epoch(
                params, opt_state, optimizer, mixed_can, mixed_rc, mixed_labels, weights
            )
            val_r = validate(params)
            if epoch % 5 == 0 or val_r > best_val_r:
                logger.info(
                    "  Epoch %d: loss=%.4f val_r=%.4f%s",
                    epoch,
                    loss,
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
                ocp.CheckpointManager(str(ckpt_dir)).save(0, args=ocp.args.StandardSave(params))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    elif args.approach == "curriculum":
        # Gradual mixing: start with 0% negatives, increase to full
        optimizer = make_optimizer(args.lr)
        opt_state = optimizer.init(params)

        for epoch in range(args.epochs):
            # Linearly increase negative fraction from 0 to 100% over first 20 epochs
            neg_frac = min(1.0, epoch / 20.0)
            n_neg_this_epoch = int(n_neg * neg_frac)

            if n_neg_this_epoch > 0:
                epoch_can = np.concatenate([train_can, neg_can[:n_neg_this_epoch]])
                epoch_rc = np.concatenate([train_rc, neg_rc[:n_neg_this_epoch]])
                epoch_labels = np.concatenate([train_labels, neg_labels[:n_neg_this_epoch]])
            else:
                epoch_can, epoch_rc, epoch_labels = train_can, train_rc, train_labels

            params, opt_state, loss = train_epoch(
                params, opt_state, optimizer, epoch_can, epoch_rc, epoch_labels
            )
            val_r = validate(params)
            if epoch % 5 == 0 or val_r > best_val_r:
                logger.info(
                    "  Epoch %d (neg_frac=%.0f%%): loss=%.4f val_r=%.4f%s",
                    epoch,
                    100 * neg_frac,
                    loss,
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
                ocp.CheckpointManager(str(ckpt_dir)).save(0, args=ocp.args.StandardSave(params))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    # Save result
    result = {
        "fold_id": args.fold_id,
        "best_val_pearson": best_val_r,
        "best_epoch": best_epoch,
        "approach": args.approach,
        "n_negatives": n_neg,
        "neg_fraction": args.neg_fraction,
    }
    result_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Done: approach=%s val_r=%.4f", args.approach, best_val_r)


if __name__ == "__main__":
    main()
