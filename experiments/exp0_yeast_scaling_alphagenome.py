#!/usr/bin/env python
"""Experiment 0 (yeast, AlphaGenome): cached head scaling curve."""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import wandb
from alphagenome_ft import create_model_with_heads
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from data.yeast import YeastDataset
from evaluation.yeast_testsets import (
    evaluate_yeast_test_subsets,
    load_yeast_test_subsets,
)
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import (
    build_head_only_predict_fn,
    build_head_only_train_fn,
    load_embedding_cache,
    reinit_head_params,
)


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    return seed


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


def _subset_indices(n_total: int, fraction: float, seed: int) -> np.ndarray:
    n_samples = max(1, int(n_total * fraction))
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, size=n_samples, replace=False)


def _gather_cached(
    indices: np.ndarray,
    train_arr: np.ndarray,
    pool_arr: np.ndarray,
    n_train: int,
) -> jnp.ndarray:
    out = np.empty((len(indices), train_arr.shape[1], train_arr.shape[2]), dtype=np.float32)
    train_mask = indices < n_train
    if np.any(train_mask):
        out[train_mask] = train_arr[indices[train_mask]].astype(np.float32)
    if np.any(~train_mask):
        pool_idx = indices[~train_mask] - n_train
        out[~train_mask] = pool_arr[pool_idx].astype(np.float32)
    return jnp.array(out)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="exp0_yeast_scaling_alphagenome",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    fraction = float(cfg.fraction)
    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    output_dir = (
        Path(str(cfg.output_dir)).expanduser().resolve()
        / f"fraction_{fraction:.4f}"
        / f"seed_{used_seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="albench-s2f",
        name=f"exp0_ag_yeast_frac{fraction:.3f}_seed{used_seed}",
        config={**OmegaConf.to_container(cfg, resolve=True), "fraction": fraction},
        tags=["exp0", "yeast", "alphagenome", "cached", "scaling"],
        mode=str(cfg.wandb_mode),
        job_type="exp0_scaling",
    )

    arch_slug = str(cfg.head_arch).replace("-", "_")
    unique_head_name = f"{cfg.head_name}_{arch_slug}_v4"
    register_s2f_head(
        head_name=unique_head_name,
        arch=str(cfg.head_arch),
        task_mode="yeast",
        num_tracks=int(cfg.num_tracks),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    )

    weights_path = str(Path(str(cfg.weights_path)).expanduser().resolve())
    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )
    reinit_head_params(model, unique_head_name, num_tokens=3, dim=1536, rng=used_seed)
    model.freeze_except_head(unique_head_name)

    cache_dir = Path(str(cfg.cache_dir)).expanduser().resolve()
    train_can, train_rc = load_embedding_cache(cache_dir, "train")
    pool_can, pool_rc = load_embedding_cache(cache_dir, "pool")
    val_can, val_rc = load_embedding_cache(cache_dir, "val")

    # Load labels: either from oracle pseudolabels or from YeastDataset
    oracle_label_path = cfg.get("oracle_label_path", None)
    if oracle_label_path:
        oracle_dir = Path(str(oracle_label_path)).expanduser().resolve()
        train_labels = np.load(oracle_dir / "train_oracle_labels.npz")["oracle_mean"].astype(
            np.float32
        )
        pool_labels = np.load(oracle_dir / "train_pool_oracle_labels.npz")["oracle_mean"].astype(
            np.float32
        )
        val_labels = np.load(oracle_dir / "val_oracle_labels.npz")["oracle_mean"].astype(np.float32)
        print(
            f"  Using oracle labels from {oracle_dir}"
            f" (train={len(train_labels)}, pool={len(pool_labels)}, val={len(val_labels)})"
        )
    else:
        ds_train = YeastDataset(
            data_path=str(cfg.yeast_data_path),
            split="train",
            context_mode=str(cfg.context_mode),
        )
        train_labels = ds_train.labels.astype(np.float32)
        ds_pool = YeastDataset(
            data_path=str(cfg.yeast_data_path),
            split="pool",
            context_mode=str(cfg.context_mode),
        )
        pool_labels = ds_pool.labels.astype(np.float32)
        ds_val = YeastDataset(
            data_path=str(cfg.yeast_data_path),
            split="val",
            context_mode=str(cfg.context_mode),
        )
        val_labels = ds_val.labels.astype(np.float32)

    # Combined train+pool for scaling
    n_train = min(len(train_labels), len(train_can))
    n_pool = min(len(pool_labels), len(pool_can))
    all_labels = np.concatenate([train_labels[:n_train], pool_labels[:n_pool]])
    n_total = len(all_labels)
    print(f"  Train={n_train:,} + Pool={n_pool:,} = {n_total:,} total sequences")
    subset_seed = used_seed + int(fraction * 100_000)
    subset_idx = _subset_indices(n_total, fraction, subset_seed)

    head_predict_fn = build_head_only_predict_fn(model, unique_head_name)
    head_train_fn = (
        build_head_only_train_fn(model, unique_head_name)
        if float(cfg.get("dropout_rate", 0.1)) > 0
        else None
    )

    @jax.jit
    def cached_train_step(params, opt_state, rng, encoder_output, targets, org_idx):
        def loss_func(p):
            if head_train_fn is not None:
                preds = head_train_fn(p, rng, encoder_output, org_idx)
            else:
                preds = head_predict_fn(p, encoder_output, org_idx)
            bins = jnp.round(jnp.clip(targets, 0.0, 17.0)).astype(jnp.int32)
            target_probs = jax.nn.one_hot(bins, int(cfg.num_tracks))
            log_probs = jax.nn.log_softmax(preds, axis=-1)
            return -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, next_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), next_state, loss

    @jax.jit
    def cached_eval_step(params, encoder_output, org_idx):
        return head_predict_fn(params, encoder_output, org_idx)

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[unique_head_name]

    optimizer = optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    opt_state = optimizer.init(model._params)
    rng = jax.random.PRNGKey(used_seed)

    best_val_pearson = -1.0
    best_val_spearman = -1.0
    epochs_no_improve = 0
    eval_use_reverse_complement = bool(cfg.get("eval_use_reverse_complement", True))

    batch_size = int(cfg.batch_size)
    for epoch in range(int(cfg.epochs)):
        perm = np.random.permutation(subset_idx)
        losses: list[float] = []

        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            tgt = jnp.array(all_labels[idx])
            org = jnp.zeros(len(idx), dtype=jnp.int32)

            rng, step_rng = jax.random.split(rng)
            emb_can = _gather_cached(idx, train_can, pool_can, n_train)
            model._params, opt_state, loss = cached_train_step(
                model._params, opt_state, step_rng, emb_can, tgt, org
            )
            losses.append(float(loss))

            rng, step_rng = jax.random.split(rng)
            emb_rc = _gather_cached(idx, train_rc, pool_rc, n_train)
            model._params, opt_state, loss = cached_train_step(
                model._params, opt_state, step_rng, emb_rc, tgt, org
            )
            losses.append(float(loss))

        val_preds: list[np.ndarray] = []
        for start in range(0, len(val_labels), batch_size):
            emb = jnp.array(val_can[start : start + batch_size].astype(np.float32))
            org = jnp.zeros(len(emb), dtype=jnp.int32)
            logits_fwd = np.array(cached_eval_step(model._params, emb, org))
            probs_fwd = np.array(jax.nn.softmax(logits_fwd, axis=-1))
            if eval_use_reverse_complement:
                emb_rc = jnp.array(val_rc[start : start + batch_size].astype(np.float32))
                logits_rc = np.array(cached_eval_step(model._params, emb_rc, org))
                probs_rc = np.array(jax.nn.softmax(logits_rc, axis=-1))
                probs = (probs_fwd + probs_rc) / 2.0
            else:
                probs = probs_fwd
            pred = np.sum(probs * np.arange(int(cfg.num_tracks), dtype=np.float32), axis=-1)
            val_preds.append(pred)

        y_pred = np.concatenate(val_preds, axis=0)
        val_pearson = _safe_corr(val_labels, y_pred, pearsonr)
        val_spearman = _safe_corr(val_labels, y_pred, spearmanr)
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": float(np.mean(losses)) if losses else float("nan"),
                "val/pearson_r": val_pearson,
                "val/spearman_r": val_spearman,
            }
        )

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_val_spearman = val_spearman
            epochs_no_improve = 0
            model.save_checkpoint(str(output_dir / "best_model"), save_full_model=False)
        else:
            epochs_no_improve += 1
        model.save_checkpoint(str(output_dir / "last_model"), save_full_model=False)

        if epochs_no_improve >= int(cfg.early_stop_patience):
            break

    test_metrics: dict[str, dict[str, float]] = {}
    subset_dir = (
        Path(str(cfg.test_subset_dir))
        if cfg.get("test_subset_dir")
        else Path(str(cfg.yeast_data_path)) / "test_subset_ids"
    )
    if subset_dir.exists():
        test_dataset = YeastDataset(
            data_path=str(cfg.yeast_data_path),
            split="test",
            context_mode=str(cfg.context_mode),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(cfg.test_batch_size),
            shuffle=False,
            num_workers=0,
        )
        preds_all: list[np.ndarray] = []
        for xb, _ in test_loader:
            x_fwd = xb[:, :4, :].permute(0, 2, 1).cpu().numpy().astype(np.float32)
            x_rev = x_fwd[:, ::-1, ::-1].copy()
            logits_fwd = np.array(predict_step(model._params, model._state, jnp.array(x_fwd)))
            logits_rev = np.array(predict_step(model._params, model._state, jnp.array(x_rev)))
            probs_fwd = np.array(jax.nn.softmax(logits_fwd, axis=-1))
            probs_rev = np.array(jax.nn.softmax(logits_rev, axis=-1))
            pred_fwd = np.sum(probs_fwd * np.arange(int(cfg.num_tracks), dtype=np.float32), axis=-1)
            pred_rev = np.sum(probs_rev * np.arange(int(cfg.num_tracks), dtype=np.float32), axis=-1)
            preds_all.append((pred_fwd + pred_rev) / 2.0)

        test_preds = np.concatenate(preds_all, axis=0)
        test_subsets = load_yeast_test_subsets(
            subset_dir=subset_dir,
            public_dir=(
                str(cfg.public_leaderboard_dir) if cfg.get("public_leaderboard_dir") else None
            ),
            use_private_only=bool(cfg.get("private_only_test", False)),
        )
        if oracle_label_path:
            test_labels = np.load(oracle_dir / "test_oracle_labels.npz")["oracle_mean"].astype(
                np.float32
            )
        else:
            test_labels = test_dataset.labels.astype(np.float32)
        test_metrics = evaluate_yeast_test_subsets(
            predictions=test_preds,
            labels=test_labels,
            subsets=test_subsets,
        )

    result = {
        "fraction": fraction,
        "n_samples": int(len(subset_idx)),
        "n_total": int(n_total),
        "seed": int(used_seed),
        "label_source": "oracle" if oracle_label_path else "real",
        "eval_use_reverse_complement": bool(eval_use_reverse_complement),
        "best_val_pearson_r": float(best_val_pearson),
        "best_val_spearman_r": float(best_val_spearman),
        "test_metrics": test_metrics,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()
