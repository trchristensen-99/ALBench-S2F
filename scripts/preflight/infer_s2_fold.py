"""Run AG-S2 fold inference on the ref+alt + boda2 K562 pool.

For one fold (passed as --fold), loads the corresponding S2 ckpt at
outputs/stage2_k562_oracle/fold_{k}/best_model/, runs full-encoder
inference on every sequence in the pool (train + val + test + snv pairs),
and saves per-fold predictions to:
    outputs/oracle_pseudolabels_k562_ag_s2_refalt/fold_preds/fold_{k}.npz

Resilient to preemption: per-split predictions saved incrementally to
fold_{k}_partial/, and existing partials are loaded and skipped if the
job restarts.

Usage:
    uv run --no-sync python scripts/preflight/infer_s2_fold.py --fold 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from alphagenome_ft import create_model_with_heads
from scipy.stats import pearsonr

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

REPO = Path(__file__).resolve().parents[2]

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


def _predict_strings(
    predict_step_fn,
    params,
    state,
    seqs_str,
    batch_size=256,
    rc_average: bool = False,
):
    """Pre-encode all seqs, then run inference. RC averaging is OFF by
    default for this preflight cache regeneration (~2× speedup vs the
    fwd+rev double-pass): AG-S2 was trained with RC augmentation so the
    head's single-strand prediction is approximately RC-equivariant, and
    the downstream students apply their own RC augmentation. Pseudolabel
    quality penalty is small.

    Set ``rc_average=True`` to fall back to fwd+rev averaging if needed.
    """
    if not seqs_str:
        return np.array([], dtype=np.float32)
    n = len(seqs_str)
    print(f"    Pre-encoding {n:,} sequences to one-hot …", flush=True)
    t0 = time.time()
    x_fwd = np.stack([_seq_str_to_600bp(s) for s in seqs_str])
    if rc_average:
        x_rev = x_fwd[:, ::-1, ::-1]
    print(f"    Pre-encoding done in {time.time() - t0:.1f}s.", flush=True)
    preds_fwd: list[np.ndarray] = []
    preds_rev: list[np.ndarray] = []
    t_loop = time.time()
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        actual = end - i
        if actual < batch_size:
            pad = batch_size - actual
            b_fwd = np.concatenate([x_fwd[i:end], np.zeros((pad, 600, 4), dtype=np.float32)])
        else:
            b_fwd = x_fwd[i:end]
        preds_fwd.append(
            np.array(predict_step_fn(params, state, jnp.array(b_fwd))).reshape(-1)[:actual]
        )
        if rc_average:
            if actual < batch_size:
                b_rev = np.concatenate([x_rev[i:end], np.zeros((pad, 600, 4), dtype=np.float32)])
            else:
                b_rev = x_rev[i:end]
            preds_rev.append(
                np.array(predict_step_fn(params, state, jnp.array(b_rev))).reshape(-1)[:actual]
            )
        if (i // batch_size) % 25 == 0:
            done_pct = 100.0 * (i + actual) / n
            elapsed = time.time() - t_loop
            rate_bps = (i // batch_size + 1) / max(0.001, elapsed)
            eta_s = (n - i - actual) / max(1, batch_size) / max(0.001, rate_bps)
            print(
                f"    {i + actual:,}/{n:,} ({done_pct:.1f}%) "
                f"rate={rate_bps:.2f} batches/s  eta={eta_s / 60:.1f}min",
                flush=True,
            )
    if rc_average:
        return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0
    return np.concatenate(preds_fwd)


def _safe_corr(y, p):
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return float(pearsonr(y, p)[0])


def _build_predict_step(fold_id: int, batch_size: int):
    """Construct AG model + load fold-k checkpoint + JIT-compile predict_step.

    The head name MUST match what was used to train the S2 ckpts —
    ``alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4`` — same
    across all 10 folds. The folds differ only in which trained
    checkpoint is loaded (per-fold weights), not in head structure.
    Verified against ``outputs/stage2_k562_oracle/fold_0/best_model/config.json``
    on 2026-05-04.
    """
    base_head_name = "alphagenome_k562_head_hashfrag"
    head_arch = "boda-flatten-512-512"
    arch_slug = head_arch.replace("-", "_")
    head_name = f"{base_head_name}_{arch_slug}_v4"
    register_s2f_head(
        head_name=head_name,
        arch=head_arch,
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.1,
    )
    weights = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights,
        use_encoder_output=True,
        detach_backbone=False,
    )
    reinit_head_params(model, head_name, num_tokens=5, dim=1536)

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            requested_outputs=[head_name],
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    ckpt_path = (
        REPO / "outputs" / "stage2_k562_oracle" / f"fold_{fold_id}" / "best_model" / "checkpoint"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing S2 ckpt: {ckpt_path}")
    print(f"Loading S2 fold {fold_id} checkpoint from {ckpt_path} …", flush=True)
    checkpointer = ocp.StandardCheckpointer()
    loaded_params, _ = checkpointer.restore(ckpt_path)
    model._params = jax.device_put(loaded_params)

    print("JIT compiling predict_step …", flush=True)
    _dummy = jnp.zeros((batch_size, 600, 4), dtype=jnp.float32)
    _ = predict_step(model._params, model._state, _dummy)
    _.block_until_ready()
    print("Compiled OK.", flush=True)
    return predict_step, model._params, model._state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    pool_dir = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt" / "pool"
    out_dir = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt" / "fold_preds"
    partial_dir = out_dir / f"fold_{args.fold}_partial"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold_{args.fold}.npz"
    if out_path.exists():
        print(f"Already done: {out_path}")
        return

    print(f"Loading pool from {pool_dir} …", flush=True)
    train = pd.read_parquet(pool_dir / "train.parquet")
    val = pd.read_parquet(pool_dir / "val.parquet")
    test = pd.read_parquet(pool_dir / "test.parquet")
    snv = pd.read_parquet(pool_dir / "snv_pairs.parquet")
    print(
        f"  train={len(train):,}  val={len(val):,}  test={len(test):,}  snv_pairs={len(snv):,}",
        flush=True,
    )

    predict_step, params, state = _build_predict_step(args.fold, args.batch_size)

    def _load_or_predict(name, seqs_list):
        p = partial_dir / f"{name}.npy"
        if p.exists():
            arr = np.load(p)
            if len(arr) == len(seqs_list):
                print(f"  [RESUME] {name}: {len(arr):,} cached", flush=True)
                return arr
        t0 = time.time()
        print(f"  Predicting {name} ({len(seqs_list):,}) …", flush=True)
        preds = _predict_strings(predict_step, params, state, seqs_list, args.batch_size).astype(
            np.float32
        )
        np.save(p, preds)
        print(f"    {name} done in {time.time() - t0:.0f}s", flush=True)
        return preds

    train_preds = _load_or_predict("train_preds", train["sequence"].astype(str).tolist())
    val_preds = _load_or_predict("val_preds", val["sequence"].astype(str).tolist())
    test_preds = _load_or_predict("test_preds", test["sequence"].astype(str).tolist())
    snv_ref_preds = _load_or_predict("snv_ref_preds", snv["sequence_ref"].astype(str).tolist())
    snv_alt_preds = _load_or_predict("snv_alt_preds", snv["sequence_alt"].astype(str).tolist())

    print(
        f"  fold {args.fold}: "
        f"train_pearson={_safe_corr(train['K562_log2FC'].to_numpy(np.float32), train_preds):.4f}  "
        f"val_pearson={_safe_corr(val['K562_log2FC'].to_numpy(np.float32), val_preds):.4f}  "
        f"test_pearson={_safe_corr(test['K562_log2FC'].to_numpy(np.float32), test_preds):.4f}",
        flush=True,
    )

    np.savez_compressed(
        out_path,
        fold_id=args.fold,
        train_preds=train_preds,
        val_preds=val_preds,
        test_preds=test_preds,
        snv_ref_preds=snv_ref_preds,
        snv_alt_preds=snv_alt_preds,
    )
    print(f"Saved {out_path}", flush=True)
    import shutil

    shutil.rmtree(partial_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
