#!/usr/bin/env python
"""Experiment 0 (K562, AlphaGenome): cached-head scaling curve.

Trains a frozen-encoder AlphaGenome head (boda-flatten-512-512) on random
subsets of the pre-computed hashFrag train+pool embedding cache (~320 K
sequences).  Because the encoder is never called during training this is
~50x faster than the full-encoder scaling experiment.

Canonical embeddings only — no RC augmentation pass, no shift.  Uses the
pre-built val_canonical.npy from the cache directory for per-epoch validation.

Run via SLURM array (one task per fraction):
  sbatch scripts/slurm/exp0_k562_scaling_alphagenome_cached.sh
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
from alphagenome_ft import create_model_with_heads
from dotenv import load_dotenv
from omegaconf import DictConfig
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import (
    build_head_only_predict_fn,
    build_head_only_train_fn,
    load_embedding_cache,
    reinit_head_params,
)

# ── MPRA flanks for 600 bp test-set evaluation ────────────────────────────────
_FLANK_5_STR: str = MPRA_UPSTREAM[-200:]
_FLANK_3_STR: str = MPRA_DOWNSTREAM[:200]
_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}

_FLANK_5_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(_FLANK_5_STR):
    if _c in _MAPPING:
        _FLANK_5_ENC[_i, _MAPPING[_c]] = 1.0

_FLANK_3_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(_FLANK_3_STR):
    if _c in _MAPPING:
        _FLANK_3_ENC[_i, _MAPPING[_c]] = 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


# ── Test-set evaluation helpers (full encoder) ────────────────────────────────


def _seq_str_to_600bp(seq_str: str) -> np.ndarray:
    seq_str = seq_str.upper()
    target_len = 200
    if len(seq_str) < target_len:
        pad = target_len - len(seq_str)
        seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
    elif len(seq_str) > target_len:
        start = (len(seq_str) - target_len) // 2
        seq_str = seq_str[start : start + target_len]
    core = np.zeros((200, 4), dtype=np.float32)
    for i, c in enumerate(seq_str):
        if c in _MAPPING:
            core[i, _MAPPING[c]] = 1.0
    return np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)  # (600, 4)


def _predict_sequences(
    predict_step_fn,
    model_params,
    model_state,
    seqs_str: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """RC-averaged predictions on raw 200 bp sequence strings via 600 bp context."""
    if not seqs_str:
        return np.array([], dtype=np.float32)
    x_fwd = np.stack([_seq_str_to_600bp(s) for s in seqs_str])
    x_rev = np.stack([_seq_str_to_600bp(s)[::-1, ::-1] for s in seqs_str])
    preds_fwd, preds_rev = [], []
    for i in range(0, len(x_fwd), batch_size):
        preds_fwd.append(
            np.array(
                predict_step_fn(model_params, model_state, jnp.array(x_fwd[i : i + batch_size]))
            ).reshape(-1)
        )
        preds_rev.append(
            np.array(
                predict_step_fn(model_params, model_state, jnp.array(x_rev[i : i + batch_size]))
            ).reshape(-1)
        )
    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def evaluate_all_test_sets(
    model,
    predict_step_fn,
    test_set_dir: Path,
) -> dict[str, dict[str, float]]:
    params, state = model._params, model._state
    metrics: dict[str, dict[str, float]] = {}

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        in_df = pd.read_csv(in_path, sep="\t")
        in_pred = _predict_sequences(predict_step_fn, params, state, in_df["sequence"].tolist())
        in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["in_distribution"] = {
            "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
            "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
            "n": int(len(in_true)),
        }

    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        ref_pred = _predict_sequences(
            predict_step_fn, params, state, snv_df["sequence_ref"].tolist()
        )
        alt_pred = _predict_sequences(
            predict_step_fn, params, state, snv_df["sequence_alt"].tolist()
        )
        alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
        metrics["snv_abs"] = {
            "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
            "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            "n": int(len(alt_true)),
        }
        delta_pred = alt_pred - ref_pred
        delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            "n": int(len(delta_true)),
        }

    ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_pred = _predict_sequences(predict_step_fn, params, state, ood_df["sequence"].tolist())
        ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["ood"] = {
            "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
            "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
            "mse": float(np.mean((ood_pred - ood_true) ** 2)),
            "n": int(len(ood_true)),
        }

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="exp0_k562_scaling_alphagenome_cached",
)
def main(cfg: DictConfig) -> None:
    """Train AlphaGenome scaling experiment using embedding cache (canonical only)."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    fraction = float(cfg.fraction)

    # Random run identity — no fixed seed
    rng_int = int.from_bytes(os.urandom(4), "big") % (2**31)
    run_id = int.from_bytes(os.urandom(4), "big") % (10**9)

    output_dir = (
        Path(str(cfg.output_dir)).expanduser().resolve()
        / f"fraction_{fraction:.4f}"
        / f"run_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dropout_rate = float(cfg.get("dropout_rate", 0.1))

    arch_slug = str(cfg.head_arch).replace("-", "_")
    unique_head_name = f"{cfg.head_name}_{arch_slug}_v4"

    register_s2f_head(
        head_name=unique_head_name,
        arch=str(cfg.head_arch),
        task_mode="human",
        num_tracks=int(cfg.num_tracks),
        dropout_rate=dropout_rate,
    )

    weights_path = str(Path(str(cfg.weights_path)).expanduser().resolve())
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {weights_path}")

    # Use explicit CPU device if no GPU is available (e.g. on Citra with old driver).
    try:
        _jax_device = jax.devices("gpu")[0]
    except RuntimeError:
        _jax_device = jax.devices("cpu")[0]
        print(f"No GPU detected — using CPU device: {_jax_device}", flush=True)

    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
        device=_jax_device,
    )
    reinit_head_params(model, unique_head_name, num_tokens=5, dim=1536, rng=rng_int)
    model.freeze_except_head(unique_head_name)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    print(f"Total parameters: {param_count:,}", flush=True)
    print(f"Run ID: {run_id}  rng_int: {rng_int}", flush=True)

    # ── Load embedding cache ──────────────────────────────────────────────────
    cache_dir = Path(str(cfg.cache_dir)).expanduser().resolve()
    print(f"Loading embedding cache from {cache_dir} …", flush=True)

    # Combine train + pool splits for training
    rc_aug: bool = bool(cfg.get("rc_aug", False))
    can_train, rc_train = load_embedding_cache(cache_dir, "train")
    can_pool, rc_pool = load_embedding_cache(cache_dir, "pool")
    all_canonical = np.concatenate([can_train, can_pool], axis=0)
    all_rc = np.concatenate([rc_train, rc_pool], axis=0) if rc_aug else None
    print(f"  Combined train+pool embeddings: {all_canonical.shape}  rc_aug={rc_aug}", flush=True)

    # Val embeddings (canonical only for validation)
    val_canonical, _ = load_embedding_cache(cache_dir, "val")
    print(f"  Val embeddings: {val_canonical.shape}", flush=True)

    # ── Load labels ───────────────────────────────────────────────────────────
    # K562Dataset(split="train") transparently merges legacy train + pool indices
    # in the same [train, pool] order as can_train / can_pool above.
    ds_all = K562Dataset(data_path=str(cfg.k562_data_path), split="train")
    all_labels = ds_all.labels.astype(np.float32)
    print(f"  Combined train+pool labels: {len(all_labels):,}", flush=True)

    ds_val = K562Dataset(data_path=str(cfg.k562_data_path), split="val")
    val_labels = ds_val.labels.astype(np.float32)
    N_val = len(val_labels)
    print(f"  Val labels: {N_val:,}", flush=True)

    # ── Fraction subsetting ───────────────────────────────────────────────────
    n_total = len(all_labels)
    n_samples = max(1, int(n_total * fraction))
    rng_subset = np.random.default_rng(rng_int)
    subset_idx = rng_subset.choice(n_total, size=n_samples, replace=False)

    train_canonical = all_canonical[subset_idx]
    train_rc = all_rc[subset_idx] if rc_aug else None
    train_labels = all_labels[subset_idx]
    N_train = n_samples

    print(
        f"Fraction {fraction:.4f}: {N_train:,}/{n_total:,} training sequences | Val: {N_val:,}",
        flush=True,
    )

    # ── Head-only JIT functions ───────────────────────────────────────────────
    head_predict_fn = build_head_only_predict_fn(model, unique_head_name)
    head_train_fn = (
        build_head_only_train_fn(model, unique_head_name) if dropout_rate > 0.0 else None
    )

    # Verify shapes
    _dummy_emb = jnp.zeros((2, 5, 1536), dtype=jnp.float32)
    _dummy_org = jnp.zeros((2,), dtype=jnp.int32)
    _ = head_predict_fn(model._params, _dummy_emb, _dummy_org)
    print("Head-only predict function compiled OK.", flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    opt_state = optimizer.init(model._params)

    # ── JIT steps ─────────────────────────────────────────────────────────────
    jax_rng = jax.random.PRNGKey(rng_int)
    batch_size = int(cfg.batch_size)

    @jax.jit
    def cached_train_step(params, current_opt_state, step_rng, encoder_output, targets, org_idx):
        def loss_func(p):
            if head_train_fn is not None:
                preds = head_train_fn(p, step_rng, encoder_output, org_idx)
            else:
                preds = head_predict_fn(p, encoder_output, org_idx)
            pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
            return jnp.mean((pred - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    @jax.jit
    def cached_eval_step(params, encoder_output, org_idx):
        preds = head_predict_fn(params, encoder_output, org_idx)
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    # Full-encoder predict for test evaluation
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

    # ── Training loop ─────────────────────────────────────────────────────────
    _aug_rng = np.random.default_rng(rng_int ^ 0xDEADBEEF)
    best_val_pearson = -1.0
    best_val_spearman = 0.0
    best_val_loss = float("inf")
    early_stop_patience = int(cfg.get("early_stop_patience", 5))
    epochs_no_improve = 0
    num_epochs_run = 0

    t_train_start = time.time()

    for epoch in range(int(cfg.epochs)):
        perm = np.random.default_rng(rng_int + epoch).permutation(N_train)
        train_losses: list[float] = []

        pbar = tqdm(
            range(0, N_train, batch_size),
            desc=f"Epoch {epoch + 1}/{int(cfg.epochs)}",
            total=(N_train + batch_size - 1) // batch_size,
        )
        for start in pbar:
            indices = perm[start : start + batch_size]
            targets_jax = jnp.array(train_labels[indices])
            org_idx = jnp.zeros(len(indices), dtype=jnp.int32)
            if rc_aug:
                use_rc = _aug_rng.random(len(indices)) > 0.5
                emb_np = np.where(
                    use_rc[:, None, None],
                    train_rc[indices].astype(np.float32),
                    train_canonical[indices].astype(np.float32),
                )
                emb_can = jnp.array(emb_np)
            else:
                emb_can = jnp.array(train_canonical[indices].astype(np.float32))

            jax_rng, step_rng = jax.random.split(jax_rng)
            model._params, opt_state, loss = cached_train_step(
                model._params, opt_state, step_rng, emb_can, targets_jax, org_idx
            )
            loss_v = float(loss)
            train_losses.append(loss_v)
            pbar.set_postfix({"loss": f"{loss_v:.4f}"})

        num_epochs_run = epoch + 1

        # ── Validation ────────────────────────────────────────────────────────
        y_pred_all: list[np.ndarray] = []
        for start in range(0, N_val, batch_size):
            end = min(start + batch_size, N_val)
            indices = np.arange(start, end, dtype=np.int64)
            emb = jnp.array(val_canonical[indices].astype(np.float32))
            org_idx = jnp.zeros(len(indices), dtype=jnp.int32)
            preds = cached_eval_step(model._params, emb, org_idx)
            y_pred_all.append(np.array(preds).reshape(-1))

        y_pred = np.concatenate(y_pred_all)
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        pear = _safe_corr(val_labels, y_pred, pearsonr)
        spear = _safe_corr(val_labels, y_pred, spearmanr)
        val_loss = float(np.mean((y_pred - val_labels) ** 2))

        print(
            f"Epoch {epoch + 1}: train_loss={avg_train:.4f}  val_pearson={pear:.4f}  "
            f"val_spearman={spear:.4f}  val_loss={val_loss:.4f}",
            flush=True,
        )

        if pear > best_val_pearson:
            best_val_pearson = pear
            best_val_spearman = spear
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save_checkpoint(str(output_dir / "best_model"), save_full_model=False)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping: no improvement for {early_stop_patience} epochs "
                    f"(best val Pearson={best_val_pearson:.4f})",
                    flush=True,
                )
                break

        model.save_checkpoint(str(output_dir / "last_model"), save_full_model=False)

    training_time_seconds = time.time() - t_train_start

    # ── Post-training evaluation on test sets (full encoder) ──────────────────
    print("\n[eval] Loading best checkpoint for test evaluation …", flush=True)

    def _merge(base, override):
        if not isinstance(override, Mapping):
            return override
        if not isinstance(base, Mapping):
            return override
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                merged[k] = _merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    ckpt_path = output_dir / "best_model" / "checkpoint"
    if ckpt_path.exists():
        checkpointer = ocp.StandardCheckpointer()
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(_merge(model._params, loaded_params))
    else:
        print("[eval] No best_model checkpoint — using final weights.", flush=True)

    test_set_dir = Path(str(cfg.k562_data_path)) / "test_sets"
    test_metrics = evaluate_all_test_sets(model, predict_step, test_set_dir)

    results = {
        "fraction": fraction,
        "n_samples": n_samples,
        "n_total": n_total,
        "training_time_seconds": training_time_seconds,
        "best_val_pearson_r": best_val_pearson,
        "best_val_spearman_r": best_val_spearman,
        "best_val_loss": best_val_loss,
        "num_epochs_run": num_epochs_run,
        "test_metrics": test_metrics,
    }

    out_json = output_dir / "result.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Wrote {out_json}", flush=True)

    for test_set, m in test_metrics.items():
        print(
            f"[eval]   {test_set}: pearson_r={m.get('pearson_r', 0.0):.4f}  "
            f"spearman_r={m.get('spearman_r', 0.0):.4f}  mse={m.get('mse', 0.0):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
