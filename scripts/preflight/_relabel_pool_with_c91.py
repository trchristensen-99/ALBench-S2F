"""Relabel an existing pool npz file with the c91 10-fold ensemble predictions.

Usage:
    python _relabel_pool_with_c91.py \\
        --pool-path outputs/labeled_pools/k562/ag_s2/random/pool.npz \\
        --c91-base outputs/oracle_neg_sweep/debias_c91_10fold \\
        --output-path outputs/labeled_pools/k562/ag_s2_c91_10fold/random/pool.npz

The pool.npz must contain 'sequences' (str array or one-hot ndarray).
For each c91 fold, predict forward + RC and average. Then average across folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from alphagenome_ft import create_model_with_heads

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

_FLANK_5 = MPRA_UPSTREAM[-200:]
_FLANK_3 = MPRA_DOWNSTREAM[:200]
_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _seq_to_600(seq: str) -> np.ndarray:
    seq = seq.upper()
    if len(seq) > 200:
        s = (len(seq) - 200) // 2
        seq = seq[s : s + 200]
    elif len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    full = _FLANK_5 + seq + _FLANK_3
    out = np.zeros((600, 4), dtype=np.float32)
    for i, c in enumerate(full):
        if c in _MAP:
            out[i, _MAP[c]] = 1.0
    return out


def _build_predict(ckpt: Path, batch: int = 256):
    head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
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
            params, state, sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            requested_outputs=[head_name],
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    loaded_params, _ = ocp.StandardCheckpointer().restore(ckpt)
    model._params = jax.device_put(loaded_params)
    _ = predict_step(model._params, model._state, jnp.zeros((batch, 600, 4), dtype=jnp.float32))
    _.block_until_ready()
    return predict_step, model._params, model._state


def _predict_batched(predict_step, params, state, x, batch=256):
    n = len(x)
    preds = []
    for i in range(0, n, batch):
        end = min(i + batch, n)
        actual = end - i
        b = x[i:end]
        if actual < batch:
            pad = batch - actual
            b = np.concatenate([b, np.zeros((pad, 600, 4), dtype=np.float32)])
        preds.append(np.array(predict_step(params, state, jnp.array(b))).reshape(-1)[:actual])
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-path", required=True, type=Path)
    ap.add_argument("--c91-base", required=True, type=Path)
    ap.add_argument("--output-path", required=True, type=Path)
    ap.add_argument("--n-folds", default=10, type=int)
    ap.add_argument("--batch-size", default=256, type=int)
    args = ap.parse_args()

    print(f"Loading pool from {args.pool_path}")
    d = np.load(args.pool_path, allow_pickle=True)
    print(f"  keys: {list(d.keys())}")

    # Try to extract sequences (could be 'sequences' str array or one-hot)
    if "sequences" in d.files:
        seqs_or_oh = d["sequences"]
    elif "x" in d.files:
        seqs_or_oh = d["x"]
    else:
        raise ValueError(f"No 'sequences' or 'x' key in {args.pool_path}")

    print(f"  shape: {seqs_or_oh.shape}")
    if seqs_or_oh.ndim == 1:
        # String array — convert each to 600bp one-hot
        print("  detected string seqs; converting to 600bp one-hot")
        canonical = np.stack([_seq_to_600(str(s)) for s in seqs_or_oh])
    elif seqs_or_oh.ndim == 3 and seqs_or_oh.shape[-1] == 4 and seqs_or_oh.shape[1] == 600:
        canonical = seqs_or_oh.astype(np.float32)
    elif seqs_or_oh.ndim == 3 and seqs_or_oh.shape[-1] == 4 and seqs_or_oh.shape[1] == 200:
        # 200bp insert — need to wrap with flanks. Convert one-hot back to str then re-encode.
        print("  detected 200bp insert one-hot; wrapping with adapters")
        rev_map = "ACGT"
        seqs_str = []
        for arr in seqs_or_oh:
            idx = arr.argmax(axis=-1)
            seqs_str.append("".join(rev_map[i] for i in idx))
        canonical = np.stack([_seq_to_600(s) for s in seqs_str])
    else:
        raise ValueError(f"Unknown sequence format: shape={seqs_or_oh.shape}")

    n = len(canonical)
    print(f"\nPredicting {n:,} sequences with {args.n_folds}-fold c91 ensemble (RC-averaged)")

    fold_preds = []
    for fold in range(args.n_folds):
        ckpt = args.c91_base / f"fold_{fold}" / "best_model" / "checkpoint"
        if not ckpt.exists():
            print(f"  WARN: skip fold {fold} — no checkpoint at {ckpt}")
            continue
        print(f"\n  Fold {fold}: predicting...")
        ps, p, s = _build_predict(ckpt, batch=args.batch_size)
        fwd = _predict_batched(ps, p, s, canonical, batch=args.batch_size)
        rev = _predict_batched(ps, p, s, canonical[:, ::-1, ::-1], batch=args.batch_size)
        avg = (fwd + rev) / 2
        fold_preds.append(avg)
        print(f"    pred range: [{avg.min():.3f}, {avg.max():.3f}], mean={avg.mean():.3f}")
        del ps, p, s
        jax.clear_caches()

    if not fold_preds:
        raise RuntimeError("No fold predictions made!")

    ensemble = np.mean(np.stack(fold_preds), axis=0).astype(np.float32)
    print(f"\nEnsemble: mean={ensemble.mean():.3f}, range=[{ensemble.min():.3f}, {ensemble.max():.3f}]")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve original pool format but replace labels
    out_data = {}
    for key in d.files:
        if key in ("labels", "targets", "y"):
            continue  # replace
        out_data[key] = d[key]
    out_data["labels"] = ensemble
    np.savez_compressed(args.output_path, **out_data)
    print(f"\nSaved {args.output_path}")


if __name__ == "__main__":
    main()
