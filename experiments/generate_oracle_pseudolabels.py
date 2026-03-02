#!/usr/bin/env python
"""Generate oracle pseudo-labels for all K562 hashFrag sequences.

Loads all oracle checkpoints in ``oracle_dir`` (one per fold), runs RC-averaged
ensemble inference on every sequence in the train+pool, val, and test splits,
and writes a single TSV with oracle mean/std predictions alongside ground-truth
labels.

Usage (standalone):
  uv run python experiments/generate_oracle_pseudolabels.py

Usage (override oracle dir):
  uv run python experiments/generate_oracle_pseudolabels.py \
      ++oracle_dir=outputs/ag_hashfrag_oracle \
      ++output_dir=outputs/oracle_pseudolabels

K-fold OOF note:
  When oracle_dir contains models trained with k-fold CV (fold_id 0–9), each
  model's val sequences are out-of-fold.  Set ``oof_mode=true`` to use
  fold-specific predictions for train+pool sequences (requires fold metadata in
  training checkpoint directories).  Default: ensemble average for all splits.
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
from tqdm import tqdm

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head

# ── MPRA flanks ───────────────────────────────────────────────────────────────
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


def _seq_to_600bp(seq_str: str) -> np.ndarray:
    seq_str = seq_str.upper()
    if len(seq_str) < 200:
        pad = 200 - len(seq_str)
        seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
    elif len(seq_str) > 200:
        start = (len(seq_str) - 200) // 2
        seq_str = seq_str[start : start + 200]
    core = np.zeros((200, 4), dtype=np.float32)
    for i, c in enumerate(seq_str):
        if c in _MAPPING:
            core[i, _MAPPING[c]] = 1.0
    return np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)  # (600, 4)


def _load_checkpoint(model, ckpt_path: Path) -> None:
    """Restore model._params from an orbax checkpoint directory."""

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

    checkpointer = ocp.StandardCheckpointer()
    loaded_params, _ = checkpointer.restore(ckpt_path)
    model._params = jax.device_put(_merge(model._params, loaded_params))


def _predict_ensemble(
    predict_fns: list,
    model_params_list: list,
    model_state,
    seqs_str: list[str],
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """RC-averaged ensemble predictions. Returns (mean, std) arrays."""
    n = len(seqs_str)
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    x_fwd = np.stack([_seq_to_600bp(s) for s in seqs_str])  # (N, 600, 4)
    x_rev = x_fwd[:, ::-1, ::-1]  # RC

    all_preds = []  # shape: (n_models, N)
    for predict_fn, params in zip(predict_fns, model_params_list):
        preds_fwd, preds_rev = [], []
        for i in range(0, n, batch_size):
            chunk_f = jnp.array(x_fwd[i : i + batch_size])
            chunk_r = jnp.array(x_rev[i : i + batch_size])
            preds_fwd.append(np.array(predict_fn(params, model_state, chunk_f)).reshape(-1))
            preds_rev.append(np.array(predict_fn(params, model_state, chunk_r)).reshape(-1))
        p = (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0
        all_preds.append(p)

    arr = np.stack(all_preds, axis=0)  # (n_models, N)
    return arr.mean(axis=0), arr.std(axis=0)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="generate_oracle_pseudolabels",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    oracle_dir = Path(str(cfg.oracle_dir)).expanduser().resolve()
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = str(Path(str(cfg.weights_path)).expanduser().resolve())
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {weights_path}")

    # ── Discover oracle checkpoints ───────────────────────────────────────────
    oracle_ckpt_paths = sorted(
        [
            p / "best_model" / "checkpoint"
            for p in sorted(oracle_dir.glob("oracle_*"))
            if (p / "best_model" / "checkpoint").exists()
        ]
    )
    if not oracle_ckpt_paths:
        raise FileNotFoundError(f"No oracle checkpoints found in {oracle_dir}")
    print(f"Found {len(oracle_ckpt_paths)} oracle checkpoints:", flush=True)
    for p in oracle_ckpt_paths:
        print(f"  {p}", flush=True)

    # ── Build model (shared structure; params swapped per oracle) ─────────────
    head_name = str(cfg.head_name)
    arch = str(cfg.head_arch)
    register_s2f_head(
        head_name=head_name,
        arch=arch,
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.0,  # inference mode — no dropout
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    # ── Load all oracle params ────────────────────────────────────────────────
    print("Loading oracle checkpoints …", flush=True)
    params_list = []
    for ckpt_path in tqdm(oracle_ckpt_paths):
        _load_checkpoint(model, ckpt_path)
        params_list.append(jax.device_put(model._params))
    model_state = model._state

    predict_fns = [predict_step] * len(params_list)

    # ── Predict on each split ─────────────────────────────────────────────────
    batch_size = int(cfg.get("batch_size", 128))
    k562_data_path = str(cfg.k562_data_path)
    rows = []

    for split in ("train", "val"):
        ds = K562Dataset(data_path=k562_data_path, split=split)
        seqs = [str(s).upper() for s in ds.sequences]
        labels = ds.labels.astype(np.float32)
        print(f"\n[{split}] Predicting {len(seqs):,} sequences …", flush=True)
        means, stds = _predict_ensemble(
            predict_fns, params_list, model_state, seqs, batch_size=batch_size
        )
        for i, (seq, true_val, mean_pred, std_pred) in enumerate(zip(seqs, labels, means, stds)):
            rows.append(
                {
                    "split": split,
                    "index": i,
                    "sequence": seq,
                    "true_log2fc": float(true_val),
                    "oracle_mean": float(mean_pred),
                    "oracle_std": float(std_pred),
                }
            )

    # ── Test sets (from TSV files) ────────────────────────────────────────────
    test_set_dir = Path(k562_data_path) / "test_sets"

    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        for col_seq, col_label, split_label in [
            ("sequence_ref", "K562_log2FC_ref", "test_snv_ref"),
            ("sequence_alt", "K562_log2FC_alt", "test_snv_alt"),
        ]:
            if col_seq not in snv_df.columns:
                continue
            seqs = snv_df[col_seq].tolist()
            labels = (
                snv_df[col_label].to_numpy(dtype=np.float32)
                if col_label in snv_df.columns
                else np.full(len(seqs), float("nan"))
            )
            print(f"\n[{split_label}] Predicting {len(seqs):,} sequences …", flush=True)
            means, stds = _predict_ensemble(
                predict_fns, params_list, model_state, seqs, batch_size=batch_size
            )
            for i, (seq, true_val, mean_pred, std_pred) in enumerate(
                zip(seqs, labels, means, stds)
            ):
                rows.append(
                    {
                        "split": split_label,
                        "index": i,
                        "sequence": seq,
                        "true_log2fc": float(true_val),
                        "oracle_mean": float(mean_pred),
                        "oracle_std": float(std_pred),
                    }
                )

    for split_label, fname in [
        ("test_in_dist", "test_in_distribution_hashfrag.tsv"),
        ("test_ood", "test_ood_designed_k562.tsv"),
    ]:
        fpath = test_set_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, sep="\t")
        seqs = df["sequence"].tolist()
        labels = df["K562_log2FC"].to_numpy(dtype=np.float32)
        print(f"\n[{split_label}] Predicting {len(seqs):,} sequences …", flush=True)
        means, stds = _predict_ensemble(
            predict_fns, params_list, model_state, seqs, batch_size=batch_size
        )
        for i, (seq, true_val, mean_pred, std_pred) in enumerate(zip(seqs, labels, means, stds)):
            rows.append(
                {
                    "split": split_label,
                    "index": i,
                    "sequence": seq,
                    "true_log2fc": float(true_val),
                    "oracle_mean": float(mean_pred),
                    "oracle_std": float(std_pred),
                }
            )

    # ── Write output ──────────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows)
    out_tsv = output_dir / "k562_oracle_pseudolabels.tsv"
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nWrote {len(out_df):,} rows to {out_tsv}", flush=True)

    # Summary stats
    summary = {}
    for split in out_df["split"].unique():
        sub = out_df[out_df["split"] == split]
        valid = sub.dropna(subset=["true_log2fc", "oracle_mean"])
        if len(valid) > 1:
            from scipy.stats import pearsonr, spearmanr

            r = pearsonr(valid["true_log2fc"].values, valid["oracle_mean"].values)[0]
            rho = spearmanr(valid["true_log2fc"].values, valid["oracle_mean"].values)[0]
            summary[split] = {"n": len(valid), "pearson_r": float(r), "spearman_r": float(rho)}
            print(f"  {split}: n={len(valid):,}  pearson={r:.4f}  spearman={rho:.4f}", flush=True)

    with open(output_dir / "pseudolabel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {output_dir / 'pseudolabel_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
