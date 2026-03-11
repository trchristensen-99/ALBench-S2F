#!/usr/bin/env python
"""Generate Stage 2 pseudolabel predictions for a SINGLE oracle fold.

Parallelized version of generate_oracle_pseudolabels_stage2_k562_ag.py.
Each fold runs as an independent SLURM array task, then an aggregation
script combines the per-fold predictions into the final NPZ files.

Saves to: output_dir/fold_preds/fold_{fold_id}.npz
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import torch
from alphagenome_ft import create_model_with_heads
from dotenv import load_dotenv
from omegaconf import DictConfig
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

# ── MPRA flanks for 600 bp sequences ─────────────────────────────────────────
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


def _safe_corr(y_true, y_pred):
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


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


def _predict_strings(predict_step_fn, params, state, seqs_str, batch_size=256):
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
            np.array(predict_step_fn(params, state, jnp.array(b_fwd))).reshape(-1)[:actual]
        )
        preds_rev.append(
            np.array(predict_step_fn(params, state, jnp.array(b_rev))).reshape(-1)[:actual]
        )
    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def _collate_to_600bp(batch):
    bsz = len(batch)
    x = np.zeros((bsz, 600, 4), dtype=np.float32)
    y = np.zeros(bsz, dtype=np.float32)
    for i, (seq_5ch, label) in enumerate(batch):
        core = np.asarray(seq_5ch)[:4, :].T
        x[i] = np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)
        y[i] = float(label.numpy()) if hasattr(label, "numpy") else float(label)
    return {"sequences": x, "targets": y}


def _predict_k562_dataset(predict_step_fn, params, state, dataset, batch_size=256):
    import time as _time

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_to_600bp,
        pin_memory=False,
    )
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    preds_all = []
    t_start = _time.time()
    for batch_idx, batch in enumerate(loader):
        seqs = batch["sequences"]
        actual = seqs.shape[0]
        if actual < batch_size:
            pad = batch_size - actual
            seqs = np.concatenate([seqs, np.zeros((pad, 600, 4), dtype=np.float32)])
        seqs_rev = seqs[:, ::-1, ::-1]
        p_fwd = np.array(predict_step_fn(params, state, jnp.array(seqs))).reshape(-1)[:actual]
        p_rev = np.array(predict_step_fn(params, state, jnp.array(seqs_rev))).reshape(-1)[:actual]
        preds_all.append((p_fwd + p_rev) / 2.0)
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            elapsed = _time.time() - t_start
            rate = (batch_idx + 1) / elapsed
            eta = (n_batches - batch_idx - 1) / rate if rate > 0 else 0
            print(
                f"    batch {batch_idx + 1}/{n_batches} ({elapsed:.0f}s elapsed, {eta:.0f}s ETA)",
                flush=True,
            )
    return np.concatenate(preds_all, axis=0)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="generate_oracle_pseudolabels_stage2_k562_ag",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    fold_id = int(cfg.fold_id)
    oracle_dir = Path(str(cfg.oracle_dir)).expanduser().resolve()
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    k562_data_path = str(cfg.k562_data_path)
    weights_path = str(Path(str(cfg.weights_path)).expanduser().resolve())
    batch_size = int(cfg.get("batch_size", 256))

    fold_preds_dir = output_dir / "fold_preds"
    fold_preds_dir.mkdir(parents=True, exist_ok=True)

    out_path = fold_preds_dir / f"fold_{fold_id}.npz"
    if out_path.exists():
        print(f"SKIP: {out_path} already exists")
        return

    # ── Verify checkpoint ─────────────────────────────────────────────────────
    fold_dir = oracle_dir / f"fold_{fold_id}"
    ckpt_path = fold_dir / "best_model" / "checkpoint"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    print(f"Processing fold {fold_id}: {ckpt_path}", flush=True)

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
        detach_backbone=False,
    )
    reinit_head_params(model, unique_head_name, num_tokens=5, dim=1536)

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

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpointer = ocp.StandardCheckpointer()
    loaded_params, _ = checkpointer.restore(ckpt_path)
    model._params = jax.device_put(loaded_params)

    # ── JIT warm-up ───────────────────────────────────────────────────────────
    print("Compiling predict_step …", flush=True)
    _dummy = jnp.zeros((batch_size, 600, 4), dtype=jnp.float32)
    _ = predict_step(model._params, model._state, _dummy)
    _.block_until_ready()
    print("Compiled OK.", flush=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    ds_train = K562Dataset(data_path=k562_data_path, split="train")
    ds_val = K562Dataset(data_path=k562_data_path, split="val")

    test_dir = Path(k562_data_path) / "test_sets"
    in_dist_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")

    # ── Predict all splits (incremental saving for preemption resilience) ────
    import time

    partial_dir = fold_preds_dir / f"fold_{fold_id}_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_predict_dataset(name, dataset):
        p = partial_dir / f"{name}.npy"
        if p.exists():
            arr = np.load(p)
            print(f"  [RESUME] Loaded {name} from cache ({len(arr):,} preds)", flush=True)
            return arr
        t0 = time.time()
        print(f"  Predicting {name} ({len(dataset):,} seqs) …", flush=True)
        preds = _predict_k562_dataset(
            predict_step, model._params, model._state, dataset, batch_size
        ).astype(np.float32)
        np.save(p, preds)
        print(f"    done in {time.time() - t0:.0f}s — saved to {p.name}", flush=True)
        return preds

    def _load_or_predict_strings(name, seqs_str):
        p = partial_dir / f"{name}.npy"
        if p.exists():
            arr = np.load(p)
            print(f"  [RESUME] Loaded {name} from cache ({len(arr):,} preds)", flush=True)
            return arr
        t0 = time.time()
        print(f"  Predicting {name} ({len(seqs_str):,} seqs) …", flush=True)
        preds = _predict_strings(predict_step, model._params, model._state, seqs_str, batch_size)
        np.save(p, preds)
        print(f"    done in {time.time() - t0:.0f}s — saved to {p.name}", flush=True)
        return preds

    train_preds = _load_or_predict_dataset("train_preds", ds_train)
    val_preds = _load_or_predict_dataset("val_preds", ds_val)
    in_dist_preds = _load_or_predict_strings("in_dist_preds", in_dist_df["sequence"].tolist())
    snv_ref_preds = _load_or_predict_strings("snv_ref_preds", snv_df["sequence_ref"].tolist())
    snv_alt_preds = _load_or_predict_strings("snv_alt_preds", snv_df["sequence_alt"].tolist())
    ood_preds = _load_or_predict_strings("ood_preds", ood_df["sequence"].tolist())

    # ── Quick sanity check ────────────────────────────────────────────────────
    val_labels = ds_val.labels.astype(np.float32)
    in_dist_labels = in_dist_df["K562_log2FC"].to_numpy(dtype=np.float32)
    print(
        f"  fold {fold_id}: val_pearson={_safe_corr(val_labels, val_preds):.4f}  "
        f"in_dist_pearson={_safe_corr(in_dist_labels, in_dist_preds):.4f}",
        flush=True,
    )

    # ── Save per-fold predictions ─────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        fold_id=fold_id,
        train_preds=train_preds,
        val_preds=val_preds,
        in_dist_preds=in_dist_preds,
        snv_ref_preds=snv_ref_preds,
        snv_alt_preds=snv_alt_preds,
        ood_preds=ood_preds,
    )
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Clean up partial files after successful final save
    import shutil

    shutil.rmtree(partial_dir, ignore_errors=True)
    print(f"Cleaned up {partial_dir}", flush=True)


if __name__ == "__main__":
    main()
