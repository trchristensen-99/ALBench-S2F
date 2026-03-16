#!/usr/bin/env python
"""Train AG S1 heads for fold_1 vs all_folds encoder comparison.

Both use the same training data (full K562 HashFrag train set) and
identical head architecture / hyperparameters. Only the pretrained
encoder weights differ.

Usage:
    uv run --no-sync python scripts/train_ag_s1_fold_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagenome_ft import create_model_with_heads

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

REPO = Path(__file__).resolve().parents[1]
WEIGHTS_BASE = "/grid/wsbs/home_norepl/christen/alphagenome_weights"

# Optimal HPs from grid search
LR = 0.0003
WD = 1e-6
DROPOUT = 0.1
SEEDS = [42, 123, 456]
EPOCHS = 50
PATIENCE = 7
BATCH_SIZE = 256

# MPRA flanks
FLANK_5 = MPRA_UPSTREAM[-200:]
FLANK_3 = MPRA_DOWNSTREAM[:200]
MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_seq(seq: str) -> np.ndarray:
    """One-hot encode a sequence with MPRA flanks to (600, 5)."""
    if len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > 200:
        start = (len(seq) - 200) // 2
        seq = seq[start : start + 200]
    full = FLANK_5 + seq + FLANK_3
    oh = np.zeros((600, 5), dtype=np.float32)
    for i, c in enumerate(full):
        if c in MAPPING:
            oh[i, MAPPING[c]] = 1.0
    oh[:, 4] = 1.0  # strand channel
    return oh


def encode_batch(model, params, seqs: list[str], bs: int = 64) -> np.ndarray:
    """Encode sequences through the AG encoder."""
    all_embs = []
    for i in range(0, len(seqs), bs):
        oh = np.stack([encode_seq(s) for s in seqs[i : i + bs]])
        emb = model.encode(params, jnp.array(oh))
        all_embs.append(np.array(emb))
        if (i // bs) % 200 == 0 and i > 0:
            print(f"    {i:,}/{len(seqs):,}", flush=True)
    return np.concatenate(all_embs, axis=0)


def train_and_evaluate(
    model_version: str,
    weights_dir: str,
    train_seqs: list[str],
    train_labels: np.ndarray,
    train_mask: np.ndarray,
    val_idx: np.ndarray,
) -> dict:
    """Train S1 head and evaluate on test sets."""
    out_dir = REPO / "outputs" / f"ag_{model_version}_k562_s1_full"
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"\nSkipping {model_version}: {result_path} exists")
        return json.loads(result_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"AG {model_version} S1 (lr={LR})")
    print(sep)

    # Register head and load model
    head_name = f"ag_{model_version}_s1"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=DROPOUT,
    )
    model = create_model_with_heads(
        model_version,
        heads=[head_name],
        checkpoint_path=weights_dir,
        detach_backbone=True,
    )

    # Encode full train set
    print("Encoding train set...", flush=True)
    t0 = time.time()
    all_embs = encode_batch(model, model._params, train_seqs)
    print(f"  Shape: {all_embs.shape}, time: {time.time() - t0:.0f}s")

    t_embs = all_embs[train_mask].astype(np.float32)
    t_labels = train_labels[train_mask]
    v_embs = all_embs[val_idx].astype(np.float32)
    v_labels = train_labels[val_idx]

    # Train with multiple seeds, keep best
    rng = np.random.default_rng(42)
    best_overall_val = -1.0
    best_overall_params = None

    for seed in SEEDS:
        print(f"\n  Seed {seed}...", flush=True)
        reinit_head_params(model, head_name, num_tokens=5, dim=1536, rng=seed)
        from models.embedding_cache import build_head_only_predict_fn, build_head_only_train_fn

        head_train_fn = build_head_only_train_fn(model, head_name)
        head_predict_fn = build_head_only_predict_fn(model, head_name)
        optimizer = optax.adamw(learning_rate=LR, weight_decay=WD)
        opt_state = optimizer.init(model._params)

        @jax.jit
        def train_step(params, opt_state, rng_key, emb, targets, org_idx):
            def loss_fn(p):
                preds = head_train_fn(p, emb, org_idx, rng_key)
                return jnp.mean((preds.squeeze() - targets) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_state, loss

        @jax.jit
        def eval_step(params, emb, org_idx):
            return head_predict_fn(params, emb, org_idx).squeeze()

        best_val = -1.0
        best_params = None
        no_improve = 0

        for epoch in range(EPOCHS):
            perm = rng.permutation(len(t_labels))
            for start in range(0, len(t_labels), BATCH_SIZE):
                idx = perm[start : start + BATCH_SIZE]
                rk = jax.random.PRNGKey(seed + epoch * 1000 + start)
                model._params, opt_state, _ = train_step(
                    model._params,
                    opt_state,
                    rk,
                    jnp.array(t_embs[idx]),
                    jnp.array(t_labels[idx]),
                    jnp.zeros(len(idx), dtype=jnp.int32),
                )

            # Validate
            vp = []
            for start in range(0, len(v_labels), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(v_labels))
                vp.append(
                    np.array(
                        eval_step(
                            model._params,
                            jnp.array(v_embs[start:end]),
                            jnp.zeros(end - start, dtype=jnp.int32),
                        )
                    )
                )
            val_r = float(np.corrcoef(np.concatenate(vp), v_labels)[0, 1])

            if val_r > best_val:
                best_val = val_r
                best_params = jax.device_get(model._params)
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: val_r={val_r:.4f}", flush=True)
            if no_improve >= PATIENCE:
                break

        print(f"    Best val: {best_val:.4f} (epoch {epoch - no_improve})")
        if best_val > best_overall_val:
            best_overall_val = best_val
            best_overall_params = best_params

    # Evaluate on test sets
    print(f"\n  Best overall val: {best_overall_val:.4f}")
    print("  Evaluating on test sets...", flush=True)

    from evaluation.exp1_eval import evaluate_on_exp1_test_panel

    class _AGS1Student:
        def predict(self, sequences: list[str]) -> np.ndarray:
            embs = encode_batch(model, best_overall_params, sequences)
            preds = []
            for i in range(0, len(embs), BATCH_SIZE):
                end = min(i + BATCH_SIZE, len(embs))
                preds.append(
                    np.array(
                        eval_step(
                            best_overall_params,
                            jnp.array(embs[i:end].astype(np.float32)),
                            jnp.zeros(end - i, dtype=jnp.int32),
                        )
                    )
                )
            return np.concatenate(preds)

    test_metrics = evaluate_on_exp1_test_panel(_AGS1Student(), "k562", Path("data/k562/test_sets"))

    for k, v in test_metrics.items():
        print(f"    {k}: pearson_r={v['pearson_r']:.4f}")

    result = {
        "model_version": model_version,
        "encoder": model_version,
        "training_data": "full_k562_train",
        "lr": LR,
        "wd": WD,
        "best_val_pearson": best_overall_val,
        "test_metrics": test_metrics,
    }
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {result_path}")
    return result


def main():
    # Load train data
    ds = K562Dataset(data_path="data/k562", split="train")
    train_seqs = list(ds.sequences)
    train_labels = ds.labels.astype(np.float32)
    print(f"Train: {len(train_seqs):,} sequences")

    # 90/10 split
    rng = np.random.default_rng(42)
    n_val = int(0.1 * len(train_seqs))
    val_idx = rng.choice(len(train_seqs), size=n_val, replace=False)
    train_mask = np.ones(len(train_seqs), dtype=bool)
    train_mask[val_idx] = False

    models = [
        ("fold_1", f"{WEIGHTS_BASE}/alphagenome-jax-fold_1"),
        ("all_folds", f"{WEIGHTS_BASE}/alphagenome-jax-all_folds-v1"),
    ]

    for model_version, weights_dir in models:
        train_and_evaluate(
            model_version, weights_dir, train_seqs, train_labels, train_mask, val_idx
        )

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
