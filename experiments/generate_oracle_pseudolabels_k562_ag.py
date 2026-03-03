#!/usr/bin/env python
"""Generate AlphaGenome oracle ensemble pseudo-labels for K562 hashFrag splits.

Produces ensemble-mean predictions (+ std + out-of-fold) for:
  • train+pool  (~320 K sequences) — from precomputed embedding cache
  • val         (~36 K sequences)  — from precomputed embedding cache
  • test_in_distribution           — full encoder, RC averaged
  • test_snv_pairs (ref + alt)     — full encoder, RC averaged
  • test_ood_designed              — full encoder, RC averaged

Because all 10 oracle models share the SAME frozen encoder (only the small
head differs), we exploit the precomputed embedding cache for train/pool/val
to avoid running the expensive encoder 10 times.  For the three test sets
(which are raw 200 bp strings not in the cache) we run the full encoder once
per oracle — acceptable since the test sets are small (~10-60 K sequences).

Output directory structure::

    outputs/oracle_pseudolabels_k562_ag/
        train_oracle_labels.npz   — oracle_mean, oracle_std, oof_oracle, true_label
        val_oracle_labels.npz     — oracle_mean, oracle_std, true_label
        test_in_dist_oracle_labels.npz  — oracle_mean, oracle_std, true_label
        test_snv_oracle_labels.npz      — ref_mean, alt_mean, delta_mean,
                                          true_alt_label, true_delta
        test_ood_oracle_labels.npz      — oracle_mean, oracle_std, true_label
        summary.json

Run via SLURM::

    sbatch scripts/slurm/generate_oracle_pseudolabels_k562_ag.sh
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
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
    load_embedding_cache,
    reinit_head_params,
)

# ── MPRA flanks for 600 bp test-set sequences ─────────────────────────────────
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


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


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


def _oracle_fold_val_indices(n_total: int, n_folds: int = 10) -> dict[int, np.ndarray]:
    """Reconstruct the val-fold index sets used during oracle training.

    Matches exactly the logic in train_oracle_alphagenome_hashfrag_cached.py
    (seed=42, equal-size folds, last fold absorbs remainder).
    """
    perm = np.random.default_rng(seed=42).permutation(n_total)
    fold_size = n_total // n_folds
    fold_val_idx: dict[int, np.ndarray] = {}
    for fold_id in range(n_folds):
        val_start = fold_id * fold_size
        val_end = val_start + fold_size if fold_id < n_folds - 1 else n_total
        fold_val_idx[fold_id] = perm[val_start:val_end]
    return fold_val_idx


# ── Test-set sequence helpers ─────────────────────────────────────────────────


def _seq_str_to_600bp(seq_str: str) -> np.ndarray:
    """Convert a 200 bp string to a 600 bp one-hot array with MPRA flanks."""
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


def _predict_strings(
    predict_step_fn,
    model_params,
    model_state,
    seqs_str: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """RC-averaged predictions on raw 200 bp strings via 600 bp context.

    Pads every batch to exactly ``batch_size`` so JAX JIT only ever sees one
    static shape, avoiding repeated costly recompilations for the last batch.
    """
    if not seqs_str:
        return np.array([], dtype=np.float32)
    n = len(seqs_str)
    x_fwd = np.stack([_seq_str_to_600bp(s) for s in seqs_str])
    x_rev = x_fwd[:, ::-1, ::-1]
    preds_fwd, preds_rev = [], []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        actual = end - i
        # Pad to batch_size so JIT sees a single static shape every call.
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


def _predict_cache(
    head_predict_fn,
    params,
    canonical: np.ndarray,
    rc: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """RC-averaged head-only predictions from precomputed embeddings."""
    N = len(canonical)
    preds_can, preds_rc = [], []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        org_idx = jnp.zeros(end - start, dtype=jnp.int32)
        emb_c = jnp.array(canonical[start:end].astype(np.float32))
        emb_r = jnp.array(rc[start:end].astype(np.float32))
        preds_can.append(np.array(head_predict_fn(params, emb_c, org_idx)).reshape(-1))
        preds_rc.append(np.array(head_predict_fn(params, emb_r, org_idx)).reshape(-1))
    return (np.concatenate(preds_can) + np.concatenate(preds_rc)) / 2.0


# ── Main ───────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="generate_oracle_pseudolabels_k562_ag",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    oracle_dir = Path(str(cfg.oracle_dir)).expanduser().resolve()
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    cache_dir = Path(str(cfg.cache_dir)).expanduser().resolve()
    k562_data_path = str(cfg.k562_data_path)
    weights_path = str(Path(str(cfg.weights_path)).expanduser().resolve())
    n_folds = int(cfg.get("n_folds", 10))
    batch_size_cache = int(cfg.get("batch_size_cache", 512))
    batch_size_encoder = int(cfg.get("batch_size_encoder", 256))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover oracle runs ───────────────────────────────────────────────────
    oracle_runs: list[tuple[int, Path]] = []
    for run_dir in sorted(oracle_dir.glob("oracle_*")):
        ckpt = run_dir / "best_model" / "checkpoint"
        if not ckpt.exists():
            print(f"[WARN] Skipping {run_dir.name}: no checkpoint at {ckpt}")
            continue
        fold_id = int(run_dir.name.split("_")[-1])
        oracle_runs.append((fold_id, run_dir))
    if not oracle_runs:
        raise FileNotFoundError(f"No oracle checkpoints found in {oracle_dir}")
    print(f"Found {len(oracle_runs)} oracle folds: {[f for f, _ in oracle_runs]}")

    # ── Head registration & model setup ───────────────────────────────────────
    arch_slug = str(cfg.head_arch).replace("-", "_")
    unique_head_name = f"{cfg.head_name}_{arch_slug}_v4"

    register_s2f_head(
        head_name=unique_head_name,
        arch=str(cfg.head_arch),
        task_mode="human",
        num_tracks=int(cfg.num_tracks),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
    )

    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )
    reinit_head_params(model, unique_head_name, num_tokens=5, dim=1536)
    model.freeze_except_head(unique_head_name)

    head_predict_fn = build_head_only_predict_fn(model, unique_head_name)

    # Warm-up compile
    _dummy_emb = jnp.zeros((2, 5, 1536), dtype=jnp.float32)
    _dummy_org = jnp.zeros(2, dtype=jnp.int32)
    _ = head_predict_fn(model._params, _dummy_emb, _dummy_org)
    print("Head-only predict compiled OK.", flush=True)

    # Full-encoder predict (for test set strings)
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

    # NOTE: Full-encoder warm-up is deferred until after checking for test caches.
    # If test caches exist, the encoder is never called → skip the 2h JIT compilation.

    # ── Load embedding cache (train + pool + val) ──────────────────────────────
    print(f"Loading embedding cache from {cache_dir} …", flush=True)
    can_train, rc_train = load_embedding_cache(cache_dir, "train")
    can_pool, rc_pool = load_embedding_cache(cache_dir, "pool")
    can_val, rc_val = load_embedding_cache(cache_dir, "val")
    all_canonical = np.concatenate([can_train, can_pool], axis=0)
    all_rc = np.concatenate([rc_train, rc_pool], axis=0)
    print(
        f"  train+pool: {all_canonical.shape}  val: {can_val.shape}",
        flush=True,
    )

    # ── Labels ────────────────────────────────────────────────────────────────
    ds_train = K562Dataset(data_path=k562_data_path, split="train")
    ds_val = K562Dataset(data_path=k562_data_path, split="val")
    train_labels = ds_train.labels.astype(np.float32)
    val_labels = ds_val.labels.astype(np.float32)
    n_train = len(train_labels)
    n_val = len(val_labels)
    print(f"  train+pool labels: {n_train:,}  val labels: {n_val:,}", flush=True)

    # ── Fold split (matches oracle training exactly) ───────────────────────────
    fold_val_idx = _oracle_fold_val_indices(n_train, n_folds)

    # ── Test sets ─────────────────────────────────────────────────────────────
    test_dir = Path(k562_data_path) / "test_sets"

    in_dist_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_dist_seqs = in_dist_df["sequence"].tolist()
    in_dist_labels = in_dist_df["K562_log2FC"].to_numpy(dtype=np.float32)

    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    snv_ref_seqs = snv_df["sequence_ref"].tolist()
    snv_alt_seqs = snv_df["sequence_alt"].tolist()
    snv_alt_labels = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    snv_delta_labels = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)

    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
    ood_seqs = ood_df["sequence"].tolist()
    ood_labels = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    print(
        f"  test_in_dist: {len(in_dist_seqs):,}  "
        f"test_snv: {len(snv_ref_seqs):,} pairs  "
        f"test_ood: {len(ood_seqs):,}",
        flush=True,
    )

    # ── Try loading test set embedding caches (built by build_test_embedding_cache.py)
    # If available, use head-only inference for test sets too (eliminates encoder JIT).
    _test_cache_available = False
    _test_caches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    _test_cache_names = ["test_in_dist", "test_snv_ref", "test_snv_alt", "test_ood"]
    try:
        for name in _test_cache_names:
            can, rc = load_embedding_cache(cache_dir, name)
            _test_caches[name] = (can, rc)
        _test_cache_available = True
        print(
            f"  Test set embedding caches found! Using head-only inference for test sets.",
            flush=True,
        )
    except FileNotFoundError:
        print(
            f"  No test set embedding caches — using full encoder for test sets.",
            flush=True,
        )

    # Warm-up compile for full encoder (only needed when test caches are missing)
    if not _test_cache_available:
        print(
            "Compiling full-encoder predict_step (may take several minutes) …",
            flush=True,
        )
        _dummy_seq = jnp.zeros((batch_size_encoder, 600, 4), dtype=jnp.float32)
        _ = predict_step(model._params, model._state, _dummy_seq)
        _.block_until_ready()
        print("Full-encoder predict compiled OK.", flush=True)

    # ── Accumulators ──────────────────────────────────────────────────────────
    def _init(n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    train_sum, train_sumsq = _init(n_train)
    val_sum, val_sumsq = _init(n_val)
    in_dist_sum, in_dist_sumsq = _init(len(in_dist_labels))
    snv_ref_sum, snv_ref_sumsq = _init(len(snv_alt_labels))
    snv_alt_sum, snv_alt_sumsq = _init(len(snv_alt_labels))
    ood_sum, ood_sumsq = _init(len(ood_labels))
    train_oof = np.full(n_train, np.nan, dtype=np.float32)

    # ── Main loop over oracle models ───────────────────────────────────────────
    checkpointer = ocp.StandardCheckpointer()

    for fold_id, run_dir in tqdm(oracle_runs, desc="Oracle folds"):
        ckpt_path = run_dir / "best_model" / "checkpoint"
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(_merge(model._params, loaded_params))

        # ── train+pool predictions (head-only, from cache) ────────────────────
        train_preds = _predict_cache(
            head_predict_fn, model._params, all_canonical, all_rc, batch_size_cache
        )
        train_sum += train_preds
        train_sumsq += train_preds.astype(np.float64) ** 2

        # OOF: each oracle predicts its held-out fold
        if fold_id in fold_val_idx:
            oof_idx = fold_val_idx[fold_id]
            train_oof[oof_idx] = train_preds[oof_idx]

        # ── val predictions (head-only, from cache) ───────────────────────────
        val_preds = _predict_cache(
            head_predict_fn, model._params, can_val, rc_val, batch_size_cache
        )
        val_sum += val_preds
        val_sumsq += val_preds.astype(np.float64) ** 2

        # ── test set predictions ──────────────────────────────────────────────
        if _test_cache_available:
            # Head-only inference from cached embeddings (fast path)
            in_dist_preds = _predict_cache(
                head_predict_fn, model._params, *_test_caches["test_in_dist"], batch_size_cache
            )
            snv_ref_preds = _predict_cache(
                head_predict_fn, model._params, *_test_caches["test_snv_ref"], batch_size_cache
            )
            snv_alt_preds = _predict_cache(
                head_predict_fn, model._params, *_test_caches["test_snv_alt"], batch_size_cache
            )
            ood_preds = _predict_cache(
                head_predict_fn, model._params, *_test_caches["test_ood"], batch_size_cache
            )
        else:
            # Full encoder inference (slow path — ~30 min/fold)
            in_dist_preds = _predict_strings(
                predict_step, model._params, model._state, in_dist_seqs, batch_size_encoder
            )
            snv_ref_preds = _predict_strings(
                predict_step, model._params, model._state, snv_ref_seqs, batch_size_encoder
            )
            snv_alt_preds = _predict_strings(
                predict_step, model._params, model._state, snv_alt_seqs, batch_size_encoder
            )
            ood_preds = _predict_strings(
                predict_step, model._params, model._state, ood_seqs, batch_size_encoder
            )

        in_dist_sum += in_dist_preds
        in_dist_sumsq += in_dist_preds.astype(np.float64) ** 2
        snv_ref_sum += snv_ref_preds
        snv_ref_sumsq += snv_ref_preds.astype(np.float64) ** 2
        snv_alt_sum += snv_alt_preds
        snv_alt_sumsq += snv_alt_preds.astype(np.float64) ** 2
        ood_sum += ood_preds
        ood_sumsq += ood_preds.astype(np.float64) ** 2

        print(
            f"  fold {fold_id}: train_mean={float(np.mean(train_preds)):.4f}  "
            f"val_pearson={_safe_corr(val_labels, val_preds, pearsonr):.4f}  "
            f"in_dist_pearson={_safe_corr(in_dist_labels, in_dist_preds, pearsonr):.4f}",
            flush=True,
        )

    # ── Finalize ensemble statistics ──────────────────────────────────────────
    n_models = len(oracle_runs)

    def _finalize(s: np.ndarray, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = (s / n_models).astype(np.float32)
        var = np.maximum((s2 / n_models) - mean.astype(np.float64) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        return mean, std

    train_mean, train_std = _finalize(train_sum, train_sumsq)
    val_mean, val_std = _finalize(val_sum, val_sumsq)
    in_dist_mean, in_dist_std = _finalize(in_dist_sum, in_dist_sumsq)
    snv_ref_mean, _ = _finalize(snv_ref_sum, snv_ref_sumsq)
    snv_alt_mean, snv_alt_std = _finalize(snv_alt_sum, snv_alt_sumsq)
    snv_delta_mean = snv_alt_mean - snv_ref_mean
    ood_mean, ood_std = _finalize(ood_sum, ood_sumsq)

    # ── Save .npz files ───────────────────────────────────────────────────────
    np.savez_compressed(
        output_dir / "train_oracle_labels.npz",
        oracle_mean=train_mean,
        oracle_std=train_std,
        oof_oracle=train_oof,
        true_label=train_labels,
    )
    np.savez_compressed(
        output_dir / "val_oracle_labels.npz",
        oracle_mean=val_mean,
        oracle_std=val_std,
        true_label=val_labels,
    )
    np.savez_compressed(
        output_dir / "test_in_dist_oracle_labels.npz",
        oracle_mean=in_dist_mean,
        oracle_std=in_dist_std,
        true_label=in_dist_labels,
    )
    np.savez_compressed(
        output_dir / "test_snv_oracle_labels.npz",
        ref_oracle_mean=snv_ref_mean,
        alt_oracle_mean=snv_alt_mean,
        delta_oracle_mean=snv_delta_mean,
        alt_oracle_std=snv_alt_std,
        true_alt_label=snv_alt_labels,
        true_delta=snv_delta_labels,
    )
    np.savez_compressed(
        output_dir / "test_ood_oracle_labels.npz",
        oracle_mean=ood_mean,
        oracle_std=ood_std,
        true_label=ood_labels,
    )
    print(f"Wrote .npz files to {output_dir}", flush=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    oof_mask = np.isfinite(train_oof)
    summary = {
        "n_oracle_models": n_models,
        "oracle_folds": [f for f, _ in oracle_runs],
        "train_pool": {
            "n": int(n_train),
            "ensemble_pearson_r": _safe_corr(train_labels, train_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(train_labels, train_mean, spearmanr),
            "oof_covered": int(np.sum(oof_mask)),
            "oof_pearson_r": _safe_corr(train_labels[oof_mask], train_oof[oof_mask], pearsonr),
            "oof_spearman_r": _safe_corr(train_labels[oof_mask], train_oof[oof_mask], spearmanr),
        },
        "val": {
            "n": int(n_val),
            "ensemble_pearson_r": _safe_corr(val_labels, val_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(val_labels, val_mean, spearmanr),
        },
        "test_in_distribution": {
            "n": int(len(in_dist_labels)),
            "ensemble_pearson_r": _safe_corr(in_dist_labels, in_dist_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(in_dist_labels, in_dist_mean, spearmanr),
        },
        "test_snv_alt": {
            "n": int(len(snv_alt_labels)),
            "ensemble_pearson_r": _safe_corr(snv_alt_labels, snv_alt_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(snv_alt_labels, snv_alt_mean, spearmanr),
        },
        "test_snv_delta": {
            "n": int(len(snv_delta_labels)),
            "ensemble_pearson_r": _safe_corr(snv_delta_labels, snv_delta_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(snv_delta_labels, snv_delta_mean, spearmanr),
        },
        "test_ood": {
            "n": int(len(ood_labels)),
            "ensemble_pearson_r": _safe_corr(ood_labels, ood_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(ood_labels, ood_mean, spearmanr),
        },
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote summary to {output_dir / 'summary.json'}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
