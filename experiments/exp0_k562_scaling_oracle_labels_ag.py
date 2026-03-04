#!/usr/bin/env python
"""Experiment 0 (K562, AlphaGenome): oracle-label cached-head scaling curve.

Trains a frozen-encoder AlphaGenome head (boda-flatten-512-512) on oracle
ensemble pseudolabels at various downsampling fractions.  Training uses the
pre-computed hashFrag embedding cache with RC augmentation.  Evaluation is
always against true test labels.

Run via SLURM array (one task per fraction):
  sbatch scripts/slurm/exp0_k562_scaling_oracle_labels_ag.sh
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
    return np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)


def _predict_sequences(
    predict_step_fn,
    model_params,
    model_state,
    seqs_str: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """RC-averaged predictions on raw 200 bp strings via 600 bp context."""
    if not seqs_str:
        return np.array([], dtype=np.float32)
    n = len(seqs_str)
    x_fwd = np.stack([_seq_str_to_600bp(s) for s in seqs_str])
    x_rev = x_fwd[:, ::-1, ::-1]
    preds_fwd, preds_rev = [], []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        actual = end - i
        if actual < batch_size:
            pad = batch_size - actual
            b_fwd = np.concatenate([x_fwd[i:end], np.zeros((pad, 600, 4), dtype=np.float32)])
            b_rev = np.concatenate([x_rev[i:end], np.zeros((pad, 600, 4), dtype=np.float32)])
        else:
            b_fwd = x_fwd[i:end]
            b_rev = x_rev[i:end]
        preds_fwd.append(
            np.array(predict_step_fn(model_params, model_state, jnp.array(b_fwd))).reshape(-1)[
                :actual
            ]
        )
        preds_rev.append(
            np.array(predict_step_fn(model_params, model_state, jnp.array(b_rev))).reshape(-1)[
                :actual
            ]
        )
    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="exp0_k562_scaling_oracle_labels_ag",
)
def main(cfg: DictConfig) -> None:
    """Train AlphaGenome head on oracle pseudolabels using embedding cache."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    fraction = float(cfg.fraction)
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

    try:
        _jax_device = jax.devices("gpu")[0]
    except RuntimeError:
        _jax_device = jax.devices("cpu")[0]

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

    # ── Load embedding cache ──────────────────────────────────────────────────
    cache_dir = Path(str(cfg.cache_dir)).expanduser().resolve()
    pseudolabel_dir = Path(str(cfg.pseudolabel_dir)).expanduser().resolve()
    rc_aug: bool = bool(cfg.get("rc_aug", True))

    can_train, rc_train = load_embedding_cache(cache_dir, "train")
    can_pool, rc_pool = load_embedding_cache(cache_dir, "pool")
    all_canonical = np.concatenate([can_train, can_pool], axis=0)
    all_rc = np.concatenate([rc_train, rc_pool], axis=0) if rc_aug else None

    val_canonical, val_rc = load_embedding_cache(cache_dir, "val")

    # ── Load oracle pseudolabels (for training + val early stopping) ─────────
    train_pl = np.load(pseudolabel_dir / "train_oracle_labels.npz")
    all_oracle_labels = train_pl["oracle_mean"].astype(np.float32)

    val_pl = np.load(pseudolabel_dir / "val_oracle_labels.npz")
    val_oracle_labels = val_pl["oracle_mean"].astype(np.float32)
    N_val = len(val_oracle_labels)

    # ── True labels (for test evaluation) ─────────────────────────────────────
    ds_val_true = K562Dataset(data_path=str(cfg.k562_data_path), split="val")
    val_true_labels = ds_val_true.labels.astype(np.float32)

    print(
        f"  train+pool oracle labels: {len(all_oracle_labels):,}  "
        f"val oracle labels: {N_val:,}  rc_aug={rc_aug}",
        flush=True,
    )

    # ── Fraction subsetting ───────────────────────────────────────────────────
    n_total = len(all_oracle_labels)
    n_samples = max(1, int(n_total * fraction))
    rng_subset = np.random.default_rng(rng_int)
    subset_idx = rng_subset.choice(n_total, size=n_samples, replace=False)

    train_canonical = all_canonical[subset_idx]
    train_rc = all_rc[subset_idx] if rc_aug else None
    train_labels = all_oracle_labels[subset_idx]
    N_train = n_samples

    print(
        f"Fraction {fraction:.4f}: {N_train:,}/{n_total:,} training sequences",
        flush=True,
    )

    # ── Head-only JIT functions ───────────────────────────────────────────────
    head_predict_fn = build_head_only_predict_fn(model, unique_head_name)
    head_train_fn = (
        build_head_only_train_fn(model, unique_head_name) if dropout_rate > 0.0 else None
    )

    _dummy_emb = jnp.zeros((2, 5, 1536), dtype=jnp.float32)
    _dummy_org = jnp.zeros((2,), dtype=jnp.int32)
    _ = head_predict_fn(model._params, _dummy_emb, _dummy_org)
    print("Head-only predict function compiled OK.", flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    opt_state = optimizer.init(model._params)

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
            train_losses.append(float(loss))
            pbar.set_postfix({"loss": f"{float(loss):.4f}"})

        num_epochs_run = epoch + 1

        # ── Validation (RC-averaged, using oracle labels for loss) ──────────
        y_pred_all: list[np.ndarray] = []
        for start in range(0, N_val, batch_size):
            end = min(start + batch_size, N_val)
            org_idx = jnp.zeros(end - start, dtype=jnp.int32)
            emb_can = jnp.array(val_canonical[start:end].astype(np.float32))
            emb_rc = jnp.array(val_rc[start:end].astype(np.float32))
            preds_can = cached_eval_step(model._params, emb_can, org_idx)
            preds_rc = cached_eval_step(model._params, emb_rc, org_idx)
            y_pred_all.append(
                (np.array(preds_can).reshape(-1) + np.array(preds_rc).reshape(-1)) / 2.0
            )

        y_pred = np.concatenate(y_pred_all)
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        # Early stopping on oracle val labels (what the model is trained on)
        pear_oracle = _safe_corr(val_oracle_labels, y_pred, pearsonr)
        # Also track correlation with true labels for monitoring
        pear_true = _safe_corr(val_true_labels, y_pred, pearsonr)
        spear = _safe_corr(val_oracle_labels, y_pred, spearmanr)

        print(
            f"Epoch {epoch + 1}: train_loss={avg_train:.4f}  "
            f"val_pearson_oracle={pear_oracle:.4f}  val_pearson_true={pear_true:.4f}",
            flush=True,
        )

        if pear_oracle > best_val_pearson:
            best_val_pearson = pear_oracle
            best_val_spearman = spear
            best_val_loss = float(np.mean((y_pred - val_oracle_labels) ** 2))
            epochs_no_improve = 0
            model.save_checkpoint(str(output_dir / "best_model"), save_full_model=False)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping: no improvement for {early_stop_patience} epochs",
                    flush=True,
                )
                break

        model.save_checkpoint(str(output_dir / "last_model"), save_full_model=False)

    training_time_seconds = time.time() - t_train_start

    # ── Post-training test evaluation (always on true labels) ─────────────────
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

    # Try cached test evaluation first
    _test_cache_names = ["test_in_dist", "test_snv_ref", "test_snv_alt", "test_ood"]
    _test_caches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        for name in _test_cache_names:
            can, rc = load_embedding_cache(cache_dir, name)
            _test_caches[name] = (can, rc)
        print("[eval] Using cached test embeddings (head-only inference).", flush=True)
    except FileNotFoundError:
        _test_caches = {}
        print("[eval] No test cache found — using full encoder (slow).", flush=True)

    test_set_dir = Path(str(cfg.k562_data_path)) / "test_sets"

    if _test_caches:

        def _predict_cached(cache_can, cache_rc):
            n = len(cache_can)
            preds = []
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                org_idx = jnp.zeros(end - i, dtype=jnp.int32)
                emb_can = jnp.array(cache_can[i:end].astype(np.float32))
                emb_rc = jnp.array(cache_rc[i:end].astype(np.float32))
                p_can = cached_eval_step(model._params, emb_can, org_idx)
                p_rc = cached_eval_step(model._params, emb_rc, org_idx)
                preds.append((np.array(p_can).reshape(-1) + np.array(p_rc).reshape(-1)) / 2.0)
            return np.concatenate(preds)

        test_metrics: dict[str, dict[str, float]] = {}

        in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
        if in_path.exists():
            in_true = pd.read_csv(in_path, sep="\t")["K562_log2FC"].to_numpy(dtype=np.float32)
            in_pred = _predict_cached(*_test_caches["test_in_dist"])
            test_metrics["in_distribution"] = {
                "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
                "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
                "mse": float(np.mean((in_pred - in_true) ** 2)),
                "n": int(len(in_true)),
            }

        snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
        if snv_path.exists():
            snv_df = pd.read_csv(snv_path, sep="\t")
            ref_pred = _predict_cached(*_test_caches["test_snv_ref"])
            alt_pred = _predict_cached(*_test_caches["test_snv_alt"])
            alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
            test_metrics["snv_abs"] = {
                "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
                "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
                "mse": float(np.mean((alt_pred - alt_true) ** 2)),
                "n": int(len(alt_true)),
            }
            delta_pred = alt_pred - ref_pred
            delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
            test_metrics["snv_delta"] = {
                "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
                "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
                "mse": float(np.mean((delta_pred - delta_true) ** 2)),
                "n": int(len(delta_true)),
            }

        ood_path = test_set_dir / "test_ood_designed_k562.tsv"
        if ood_path.exists():
            ood_true = pd.read_csv(ood_path, sep="\t")["K562_log2FC"].to_numpy(dtype=np.float32)
            ood_pred = _predict_cached(*_test_caches["test_ood"])
            test_metrics["ood"] = {
                "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
                "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
                "mse": float(np.mean((ood_pred - ood_true) ** 2)),
                "n": int(len(ood_true)),
            }
    else:
        from experiments.exp0_k562_scaling_alphagenome_cached import evaluate_all_test_sets

        test_metrics = evaluate_all_test_sets(model, predict_step, test_set_dir)

    results = {
        "fraction": fraction,
        "n_samples": n_samples,
        "n_total": n_total,
        "label_source": "oracle_pseudolabel",
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
