#!/usr/bin/env python
"""Pre-train single ISE fitness predictor models for Experiment 1.1.

These models are trained ONCE and saved as checkpoints. During ISE reservoir
generation in ``exp1_1_scaling.py``, they are loaded to score candidate
sequences without re-training.

Models per task:
  - ``dream_rnn_10pct``  — DREAM-RNN on 10% of oracle-labeled train data
  - ``dream_rnn_100pct`` — DREAM-RNN on 100% of oracle-labeled train data
  - ``ag_s1_10pct``      — AG-S1 head on 10% of oracle-labeled train data
  - ``ag_s1_100pct``     — AG-S1 head on 100% of oracle-labeled train data

Usage::

    # Single model
    uv run python experiments/train_ise_fitness_models.py \\
        --task k562 --model-type dream_rnn --train-fraction 0.1

    # Full dataset AG-S1
    uv run python experiments/train_ise_fitness_models.py \\
        --task yeast --model-type ag_s1 --train-fraction 1.0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.exp1_1_scaling import (  # noqa: E402
    TASK_CONFIGS,
    _encode_sequences_for_ag,
    _get_ag_model_and_encoder,
    _load_oracle,
    _load_pool_sequences,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

VAL_FRACTION = 0.1  # hold out 10% of the subsample for validation


def _split_train_val(
    sequences: list[str],
    labels: np.ndarray,
    val_frac: float,
    seed: int,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Split sequences/labels into train and val subsets."""
    rng = np.random.default_rng(seed)
    n = len(sequences)
    n_val = max(1, int(n * val_frac))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_seqs = [sequences[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    val_labels = labels[val_idx]
    return train_seqs, train_labels, val_seqs, val_labels


def _compute_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(pred) < 2:
        return 0.0
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)
    pred_mean = pred - pred.mean()
    true_mean = true - true.mean()
    num = np.dot(pred_mean, true_mean)
    denom = np.sqrt(np.dot(pred_mean, pred_mean) * np.dot(true_mean, true_mean))
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


# ---------------------------------------------------------------------------
# DREAM-RNN training
# ---------------------------------------------------------------------------


def train_dream_rnn(
    task: str,
    sequences: list[str],
    labels: np.ndarray,
    output_dir: Path,
    seed: int = 42,
    epochs: int = 80,
    lr: float = 0.005,
    batch_size: int = 1024,
) -> dict:
    """Train a single DREAM-RNN model and save checkpoint + metrics."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from models.dream_rnn import create_dream_rnn
    from models.loss_utils import YeastKLLoss
    from models.training import train_model_optimized
    from models.training_base import create_optimizer_and_scheduler

    cfg = TASK_CONFIGS[task]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into train/val
    train_seqs, train_labels, val_seqs, val_labels = _split_train_val(
        sequences, labels, VAL_FRACTION, seed
    )
    logger.info(f"DREAM-RNN: {len(train_seqs):,} train, {len(val_seqs):,} val sequences")

    # Encode sequences
    from models.dream_rnn_student import DREAMRNNStudent

    # Use student's encoding logic for consistency
    encoder = DREAMRNNStudent(
        input_channels=cfg["input_channels"],
        sequence_length=cfg["sequence_length"],
        task_mode=cfg["task_mode"],
        ensemble_size=1,
    )
    x_train = encoder._encode_sequences(train_seqs)
    y_train = torch.from_numpy(train_labels.astype(np.float32))
    x_val = encoder._encode_sequences(val_seqs)
    y_val = torch.from_numpy(val_labels.astype(np.float32))

    class _InMemDS(Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    train_ds = _InMemDS(x_train, y_train)
    val_ds = _InMemDS(x_val, y_val)

    nw = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        persistent_workers=nw > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        persistent_workers=nw > 0,
    )

    # Create model
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = create_dream_rnn(
        input_channels=cfg["input_channels"],
        sequence_length=cfg["sequence_length"],
        task_mode=cfg["task_mode"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=epochs,
        lr=lr,
        lr_lstm=lr,
        weight_decay=0.01,
        pct_start=0.3,
    )
    criterion: nn.Module = YeastKLLoss() if cfg["task_mode"] == "yeast" else nn.MSELoss()

    t0 = time.time()
    history = train_model_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=epochs,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=output_dir,
        use_reverse_complement=True,
        early_stopping_patience=10,
        metric_for_best="pearson_r",
        use_amp=True,
        use_compile=False,
    )
    wall_time = time.time() - t0

    # Load best model checkpoint if saved, otherwise use current
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    # Save final checkpoint as model.pt
    model_path = output_dir / "model.pt"
    torch.save(
        {"model_state_dict": model.state_dict()},
        model_path,
    )
    logger.info(f"Saved DREAM-RNN checkpoint: {model_path}")

    # Compute final val Pearson
    model.to(device).eval()
    val_preds = []
    with torch.no_grad():
        for i in range(0, len(x_val), batch_size):
            batch = x_val[i : i + batch_size].float().to(device)
            p = model.predict(batch, use_reverse_complement=True)
            val_preds.append(p.cpu().numpy().reshape(-1))
    val_preds = np.concatenate(val_preds)
    val_pearson = _compute_pearson(val_preds, val_labels)

    # Best val Pearson from history
    best_val_pearson = max(history.get("val_pearson_r", [val_pearson]))

    metrics = {
        "model_type": "dream_rnn",
        "task": task,
        "n_train": len(train_seqs),
        "n_val": len(val_seqs),
        "n_total": len(sequences),
        "epochs_trained": len(history.get("train_loss", [])),
        "val_pearson": float(val_pearson),
        "best_val_pearson": float(best_val_pearson),
        "wall_seconds": wall_time,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
    }

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"DREAM-RNN val Pearson: {val_pearson:.4f} (best: {best_val_pearson:.4f})")

    # Clean up intermediate checkpoints
    for f in output_dir.glob("last_model*.pt"):
        f.unlink(missing_ok=True)

    return metrics


# ---------------------------------------------------------------------------
# AG-S1 training
# ---------------------------------------------------------------------------


def train_ag_s1(
    task: str,
    sequences: list[str],
    labels: np.ndarray,
    output_dir: Path,
    seed: int = 42,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> dict:
    """Train an AG-S1 head and save params as npz + metrics."""
    import jax
    import jax.numpy as jnp
    import optax

    from models.embedding_cache import reinit_head_params

    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into train/val
    train_seqs, train_labels, val_seqs, val_labels = _split_train_val(
        sequences, labels, VAL_FRACTION, seed
    )
    logger.info(f"AG-S1: {len(train_seqs):,} train, {len(val_seqs):,} val sequences")

    ag = _get_ag_model_and_encoder(task)
    model = ag["model"]
    head_train_fn = ag["head_train_fn"]
    head_predict_fn = ag["head_predict_fn"]
    encoder_fn = ag["encoder_fn"]

    # Re-init head
    reinit_head_params(model, ag["head_name"], num_tokens=ag["num_tokens"], dim=1536, rng=seed)

    # Encode all sequences
    logger.info(f"Encoding {len(train_seqs):,} train sequences with AG encoder...")
    t0 = time.time()
    train_embs = _encode_sequences_for_ag(train_seqs, task, encoder_fn)
    logger.info(f"Train embeddings: {train_embs.shape} ({time.time() - t0:.1f}s)")

    logger.info(f"Encoding {len(val_seqs):,} val sequences with AG encoder...")
    val_embs = _encode_sequences_for_ag(val_seqs, task, encoder_fn)
    logger.info(f"Val embeddings: {val_embs.shape}")

    # Setup optimizer
    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-6)
    opt_state = optimizer.init(model._params)
    jax_rng = jax.random.PRNGKey(seed)

    @jax.jit
    def train_step(params, current_opt_state, step_rng, emb, targets, org_idx):
        def loss_func(p):
            preds = head_train_fn(p, step_rng, emb, org_idx)
            pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
            return jnp.mean((pred - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    @jax.jit
    def eval_step(params, emb, org_idx):
        preds = head_predict_fn(params, emb, org_idx)
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    # Training loop
    n_train = len(train_seqs)
    rng_perm = np.random.default_rng(seed)
    best_val_pearson = -float("inf")
    best_params = None
    patience, patience_counter = 7, 0

    t0 = time.time()
    for epoch in range(epochs):
        perm = rng_perm.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            emb = jnp.array(train_embs[idx].astype(np.float32))
            targets = jnp.array(train_labels[idx])
            org_idx = jnp.zeros(len(idx), dtype=jnp.int32)
            jax_rng, step_rng = jax.random.split(jax_rng)
            model._params, opt_state, loss = train_step(
                model._params, opt_state, step_rng, emb, targets, org_idx
            )
            epoch_losses.append(float(loss))

        # Validate
        val_preds = []
        for start in range(0, len(val_seqs), batch_size):
            emb = jnp.array(val_embs[start : start + batch_size].astype(np.float32))
            org = jnp.zeros(emb.shape[0], dtype=jnp.int32)
            p = eval_step(model._params, emb, org)
            val_preds.append(np.array(p).reshape(-1))
        val_preds_arr = np.concatenate(val_preds)
        val_pearson = _compute_pearson(val_preds_arr, val_labels)

        mean_loss = float(np.mean(epoch_losses))
        logger.info(
            f"  Epoch {epoch + 1}/{epochs}: "
            f"train_loss={mean_loss:.5f}, val_pearson={val_pearson:.4f}"
        )

        if val_pearson > best_val_pearson + 1e-5:
            best_val_pearson = val_pearson
            best_params = jax.device_get(model._params)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  AG-S1 early stop at epoch {epoch + 1}")
                break

    wall_time = time.time() - t0

    # Use best params
    if best_params is None:
        best_params = jax.device_get(model._params)

    # Save via pickle for reliable loading
    # Also save via jax serialization for reliable loading
    import pickle

    pickle_path = output_dir / "head_params.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(best_params, f)
    logger.info(f"Saved AG-S1 head params: {pickle_path}")

    metrics = {
        "model_type": "ag_s1",
        "task": task,
        "n_train": len(train_seqs),
        "n_val": len(val_seqs),
        "n_total": len(sequences),
        "epochs_trained": epoch + 1,
        "val_pearson": float(best_val_pearson),
        "wall_seconds": wall_time,
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "head_name": ag["head_name"],
    }
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"AG-S1 best val Pearson: {best_val_pearson:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train ISE fitness predictor models for Experiment 1.1"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["k562", "yeast"],
        help="Task to train for",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["dream_rnn", "ag_s1"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction of training data to use (0.1 or 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated under outputs/ise_fitness_models/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--oracle-type",
        type=str,
        default="default",
        help="Oracle type for labeling (default: AG for K562, DREAM for yeast)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Build output dir
    frac_str = f"{int(args.train_fraction * 100)}pct"
    if args.output_dir is None:
        output_dir = (
            REPO / "outputs" / "ise_fitness_models" / args.task / f"{args.model_type}_{frac_str}"
        )
    else:
        output_dir = Path(args.output_dir)

    logger.info(f"Task: {args.task}, Model: {args.model_type}, Fraction: {args.train_fraction}")
    logger.info(f"Output: {output_dir}")

    # 1. Load pool sequences
    logger.info("Loading pool sequences...")
    pool_seqs, pool_labels_real = _load_pool_sequences(args.task)
    n_pool = len(pool_seqs)
    logger.info(f"Pool size: {n_pool:,}")

    # 2. Subsample
    rng = np.random.default_rng(args.seed)
    n_use = max(1, int(n_pool * args.train_fraction))
    if n_use < n_pool:
        idx = rng.choice(n_pool, size=n_use, replace=False)
        idx.sort()
        sub_seqs = [pool_seqs[i] for i in idx]
    else:
        sub_seqs = pool_seqs
        idx = np.arange(n_pool)
    logger.info(f"Using {len(sub_seqs):,} sequences ({args.train_fraction * 100:.0f}%)")

    # 3. Label with oracle
    logger.info("Loading oracle for labeling...")
    oracle = _load_oracle(args.task, oracle_type=args.oracle_type)

    logger.info(f"Labeling {len(sub_seqs):,} sequences with oracle...")
    t0 = time.time()
    oracle_labels = oracle.predict(sub_seqs)
    label_time = time.time() - t0
    logger.info(f"Labeling done in {label_time:.1f}s")

    # Free oracle GPU memory
    del oracle
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # 4. Train model
    if args.model_type == "dream_rnn":
        metrics = train_dream_rnn(
            task=args.task,
            sequences=sub_seqs,
            labels=oracle_labels,
            output_dir=output_dir,
            seed=args.seed,
        )
    elif args.model_type == "ag_s1":
        metrics = train_ag_s1(
            task=args.task,
            sequences=sub_seqs,
            labels=oracle_labels,
            output_dir=output_dir,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Add labeling info to metrics
    metrics["oracle_type"] = args.oracle_type
    metrics["label_time_seconds"] = label_time
    metrics["train_fraction"] = args.train_fraction

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("=" * 60)
    logger.info(f"DONE: {args.model_type} ({frac_str}) for {args.task}")
    logger.info(
        f"  Val Pearson: {metrics.get('val_pearson', metrics.get('best_val_pearson', 'N/A'))}"
    )
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
