#!/usr/bin/env python
"""Clean Stage-1 AG oracle head, full-dataset random 10-fold CV.

Trains the BodaFlatten head on the frozen AlphaGenome encoder using the
pre-built full embedding cache (ref + alt + OOD designed, 856,252 rows).
Reuses the proven head-only training internals from
``train_oracle_alphagenome_hashfrag_cached.py`` (``freeze_except_head`` +
``build_head_only_train_fn`` + separate canonical / RC gradient passes per
batch), which converge to val Pearson ~0.90 — unlike the earlier
``train_oracle_s1_full.py`` (hand-rolled loss, can+RC averaged) which
degenerated at epoch 0.

No hashfrag, no chromosome holdout: each fold is a deterministic random
90/10 partition (seed=42 permutation), so the 10-fold ensemble collectively
covers the entire dataset.

  --max-idx restricts the training pool to rows [0:max_idx). Default (None)
  uses all 856,252 (the standard "default" oracle). Pass 833290 to exclude the
  22,962 OOD designed high-activity sequences (the "comparison" oracle).

Usage (one fold)::

    uv run --no-sync python experiments/train_oracle_s1_fullcv.py \
        --cache-dir outputs/oracle_full856k_clean/embedding_cache \
        --output-dir outputs/oracle_full856k_clean/s1/oracle_0 \
        --fold-id 0 --n-folds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HEAD_NAME = "oracle_k562_fullcv_boda_flatten_512_512_v4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-id", required=True, type=int)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument(
        "--max-idx",
        type=int,
        default=None,
        help="Restrict training pool to rows [0:max_idx). None=all rows.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--head-arch", default="boda-flatten-512-512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights-path",
        default=os.environ.get(
            "ALPHAGENOME_WEIGHTS",
            "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "test_metrics.json"
    if result_path.exists():
        logger.info("Already completed (%s exists), skipping.", result_path)
        return

    import jax
    import jax.numpy as jnp
    import optax
    from alphagenome_ft import create_model_with_heads
    from scipy.stats import pearsonr, spearmanr

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import (
        build_head_only_predict_fn,
        build_head_only_train_fn,
        load_embedding_cache,
        reinit_head_params,
    )

    def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn) -> float:
        if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
            return 0.0
        return float(fn(y_true, y_pred)[0])

    np.random.seed(args.seed)

    # ── Model: frozen encoder + fresh head ────────────────────────────────────
    register_s2f_head(
        head_name=HEAD_NAME,
        arch=args.head_arch,
        task_mode="human",
        num_tracks=1,
        dropout_rate=args.dropout_rate,
    )
    if not Path(args.weights_path).exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {args.weights_path}")
    model = create_model_with_heads(
        "all_folds",
        heads=[HEAD_NAME],
        checkpoint_path=args.weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )
    reinit_head_params(model, HEAD_NAME, num_tokens=5, dim=1536, rng=args.seed)
    model.freeze_except_head(HEAD_NAME)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    logger.info("Total parameters: %s", f"{param_count:,}")

    # ── Labels + embedding cache (memmap) ─────────────────────────────────────
    all_labels = np.load(args.cache_dir / "all_labels.npy").astype(np.float32)
    all_canonical, all_rc = load_embedding_cache(args.cache_dir, "train")
    logger.info(
        "Cache: labels=%s canonical=%s rc=%s",
        all_labels.shape,
        all_canonical.shape,
        all_rc.shape,
    )
    assert len(all_labels) == len(all_canonical) == len(all_rc), "cache row mismatch"

    pool = len(all_labels) if args.max_idx is None else int(args.max_idx)
    logger.info("Training pool: rows [0:%d) of %d", pool, len(all_labels))

    # ── Deterministic random 10-fold CV over [0:pool) ─────────────────────────
    perm = np.random.default_rng(seed=42).permutation(pool)
    fold_size = pool // args.n_folds
    val_start = args.fold_id * fold_size
    val_end = val_start + fold_size if args.fold_id < args.n_folds - 1 else pool
    val_idx = np.sort(perm[val_start:val_end])
    train_idx = np.sort(np.concatenate([perm[:val_start], perm[val_end:]]))
    N_train, N_val = len(train_idx), len(val_idx)
    logger.info(
        "Fold %d/%d — train=%s val=%s",
        args.fold_id,
        args.n_folds,
        f"{N_train:,}",
        f"{N_val:,}",
    )

    val_labels = all_labels[val_idx]

    # ── Head-only JIT functions ───────────────────────────────────────────────
    head_predict_fn = build_head_only_predict_fn(model, HEAD_NAME)
    head_train_fn = build_head_only_train_fn(model, HEAD_NAME) if args.dropout_rate > 0.0 else None

    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(model._params)

    @jax.jit
    def train_step(params, current_opt_state, rng, encoder_output, targets, org_idx):
        def loss_fn(p):
            if head_train_fn is not None:
                preds = head_train_fn(p, rng, encoder_output, org_idx)
            else:
                preds = head_predict_fn(p, encoder_output, org_idx)
            pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
            return jnp.mean((pred - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    @jax.jit
    def eval_step(params, encoder_output, org_idx):
        preds = head_predict_fn(params, encoder_output, org_idx)
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    # ── Training loop (canonical + RC separate passes) ────────────────────────
    best_val_pearson = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    rng = jax.random.PRNGKey(args.seed)

    for epoch in range(args.epochs):
        order = np.random.permutation(N_train)
        train_losses: list[float] = []
        for start in range(0, N_train, args.batch_size):
            bidx = np.sort(train_idx[order[start : start + args.batch_size]])
            targets = jnp.array(all_labels[bidx])
            org_idx = jnp.zeros(len(bidx), dtype=jnp.int32)

            rng, step_rng = jax.random.split(rng)
            emb_can = jnp.array(np.asarray(all_canonical[bidx]).astype(np.float32))
            model._params, opt_state, loss = train_step(
                model._params, opt_state, step_rng, emb_can, targets, org_idx
            )
            train_losses.append(float(loss))

            rng, step_rng = jax.random.split(rng)
            emb_rc = jnp.array(np.asarray(all_rc[bidx]).astype(np.float32))
            model._params, opt_state, loss = train_step(
                model._params, opt_state, step_rng, emb_rc, targets, org_idx
            )
            train_losses.append(float(loss))

        # Validation (RC-averaged)
        y_pred_all: list[np.ndarray] = []
        for start in range(0, N_val, args.batch_size):
            bidx = val_idx[start : start + args.batch_size]
            org_idx = jnp.zeros(len(bidx), dtype=jnp.int32)
            emb_can = jnp.array(np.asarray(all_canonical[bidx]).astype(np.float32))
            emb_rc = jnp.array(np.asarray(all_rc[bidx]).astype(np.float32))
            preds_can = eval_step(model._params, emb_can, org_idx)
            preds_rc = eval_step(model._params, emb_rc, org_idx)
            y_pred_all.append(
                (np.array(preds_can).reshape(-1) + np.array(preds_rc).reshape(-1)) / 2.0
            )
        y_pred = np.concatenate(y_pred_all)
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        pear = _safe_corr(val_labels, y_pred, pearsonr)
        spear = _safe_corr(val_labels, y_pred, spearmanr)
        logger.info(
            "Epoch %d: train_loss=%.4f val_pearson=%.4f val_spearman=%.4f%s",
            epoch + 1,
            avg_train,
            pear,
            spear,
            " *" if pear > best_val_pearson else "",
        )

        if pear > best_val_pearson:
            best_val_pearson = pear
            best_epoch = epoch
            epochs_no_improve = 0
            model.save_checkpoint(str(args.output_dir / "best_model"), save_full_model=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (best=%d, val_pearson=%.4f)",
                    epoch + 1,
                    best_epoch + 1,
                    best_val_pearson,
                )
                break

    result = {
        "fold_id": args.fold_id,
        "n_folds": args.n_folds,
        "max_idx": args.max_idx,
        "pool": pool,
        "best_val_pearson": best_val_pearson,
        "best_epoch": best_epoch,
        "head_arch": args.head_arch,
        "head_name": HEAD_NAME,
        "n_train": N_train,
        "n_val": N_val,
        "seed": args.seed,
    }
    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved %s (best_val_pearson=%.4f)", result_path, best_val_pearson)


if __name__ == "__main__":
    main()
