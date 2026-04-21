#!/usr/bin/env python
"""Rank pool sequences by oracle uncertainty (ensemble variance).

Uses the AlphaGenome S2 10-fold ensemble: for each sequence, compute
predictions from all folds and take the variance as the uncertainty score.
High variance = high uncertainty = the oracle disagrees with itself.

For 5M sequences × 10 folds, this requires ~50M forward passes through
the AG head. With cached embeddings, each pass is fast (~0.1ms), so
the bottleneck is loading embeddings and computing head predictions.

Strategy: Load the pool, run each of the 10 oracle folds on all sequences,
compute per-sequence variance, save ranked indices.

Usage:
    python scripts/rank_pool_by_uncertainty.py \
        --strategy evoaug_heavy --pool-size 5m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["TORCHDYNAMO_DISABLE"] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_pool(strategy, pool_size="5m"):
    """Load pool sequences and existing oracle labels."""
    pool_dirs = {
        "5m": REPO / "outputs" / "labeled_pools_5m" / "k562" / "ag_s2" / strategy,
        "2m": REPO / "outputs" / "labeled_pools_2m" / "k562" / "ag_s2" / strategy,
        "618k": REPO / "outputs" / "labeled_pools" / "k562" / "ag_s2" / strategy,
    }

    pool_path = pool_dirs.get(pool_size, pool_dirs["618k"]) / "pool.npz"
    if not pool_path.exists():
        # Fallback
        for size in ["5m", "2m", "618k"]:
            p = pool_dirs[size] / "pool.npz"
            if p.exists():
                pool_path = p
                break

    print(f"Loading pool from {pool_path}")
    data = np.load(pool_path, allow_pickle=True)
    return data["sequences"], data["labels"]


def compute_uncertainty_from_labels(labels):
    """If labels are already ensemble means, we can't get variance.

    Instead, use a proxy: train a quick model on a subset and use
    its prediction error as uncertainty. But the user wants AG oracle
    uncertainty specifically.

    The real approach: load each of the 10 AG folds and predict.
    This requires the AG model + embeddings.
    """
    pass


def compute_fold_predictions(sequences, fold_id, batch_size=512):
    """Run a single AG oracle fold on all sequences.

    Uses cached embeddings if available, otherwise runs full encoder.
    Returns predictions array.
    """
    import jax
    import jax.numpy as jnp
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params

    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )

    head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
    )

    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )
    reinit_head_params(model, head_name, num_tokens=5, dim=1536, rng=42)

    # Load fold-specific checkpoint
    oracle_dir = REPO / "outputs" / "ag_hashfrag_oracle_cached" / f"oracle_{fold_id}"
    ckpt_path = oracle_dir / "best_model" / "checkpoint"
    if ckpt_path.exists():
        import orbax.checkpoint as ocp

        checkpointer = ocp.StandardCheckpointer()
        loaded_params, _ = checkpointer.restore(str(ckpt_path.resolve()))
        model._params = jax.device_put(loaded_params)
        print(f"  Loaded fold {fold_id} weights")
    else:
        print(f"  WARNING: fold {fold_id} checkpoint not found at {ckpt_path}")
        return None

    @jax.jit
    def predict_step(params, state, sequences):
        preds = model._predict(
            params,
            model._state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
            requested_outputs=[head_name],
        )[head_name]
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    # Encode and predict in batches
    _MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
    all_preds = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        ohe = np.zeros((len(batch), 600, 4), dtype=np.float32)
        for j, seq in enumerate(batch):
            seq_str = str(seq)[:200].upper()
            for k, c in enumerate(seq_str):
                if c in _MAP:
                    ohe[j, 200 + k, _MAP[c]] = 1.0

        preds = predict_step(model._params, model._state, jnp.array(ohe))
        all_preds.append(np.array(preds).flatten())

        if i % (batch_size * 100) == 0 and i > 0:
            print(f"    Fold {fold_id}: {i}/{len(sequences)} sequences")

    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--pool-size", default="5m", choices=["5m", "2m", "618k"])
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of oracle folds to use (max 10)"
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(REPO / "outputs" / "uncertainty_ranking" / args.strategy)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    ranking_file = out_dir / "uncertainty_ranking.npz"
    if ranking_file.exists():
        print(f"SKIP: {ranking_file} exists")
        return

    # Load pool
    sequences, labels = load_pool(args.strategy, args.pool_size)
    n_seqs = len(sequences)
    print(f"Pool: {n_seqs} sequences")

    # Estimate time
    # Each fold on 5M seqs at batch_size=512: ~10K batches × ~50ms = ~8 min
    # 5 folds: ~40 min total
    est_min = n_seqs / 512 * 0.05 / 60 * args.n_folds
    print(f"Estimated time: ~{est_min:.0f} min for {args.n_folds} folds")

    # Run each fold
    t0 = time.time()
    fold_preds = []
    for fold_id in range(args.n_folds):
        print(f"  Running fold {fold_id}...")
        preds = compute_fold_predictions(sequences, fold_id)
        if preds is not None:
            fold_preds.append(preds)
        else:
            print(f"  Skipping fold {fold_id} (no checkpoint)")

    elapsed = time.time() - t0
    print(f"All folds done in {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    if len(fold_preds) < 2:
        print("ERROR: need at least 2 folds for variance")
        return

    # Stack and compute variance
    fold_matrix = np.stack(fold_preds, axis=0)  # (n_folds, n_seqs)
    uncertainty = np.var(fold_matrix, axis=0)  # (n_seqs,)
    ensemble_mean = np.mean(fold_matrix, axis=0)

    # Rank by uncertainty (descending — most uncertain first)
    ranking = np.argsort(-uncertainty)

    # Save
    np.savez_compressed(
        ranking_file,
        ranking=ranking,
        uncertainty=uncertainty,
        ensemble_mean=ensemble_mean,
        fold_matrix=fold_matrix,
    )

    # Also save summary
    summary = {
        "strategy": args.strategy,
        "pool_size": n_seqs,
        "n_folds": len(fold_preds),
        "uncertainty_mean": float(np.mean(uncertainty)),
        "uncertainty_std": float(np.std(uncertainty)),
        "uncertainty_max": float(np.max(uncertainty)),
        "time_sec": elapsed,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved ranking to {ranking_file}")
    print(f"Uncertainty: mean={np.mean(uncertainty):.4f} max={np.max(uncertainty):.4f}")


if __name__ == "__main__":
    main()
