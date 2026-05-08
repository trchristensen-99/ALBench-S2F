"""Test-time augmentation (TTA) eval: predict each sequence with multiple
shift offsets + RC, average predictions. Pure inference-time technique
that may reduce variance and improve SNV correlation without retraining.

For each ref/alt sequence:
  - Generate 5 shifted variants (offsets -10, -5, 0, +5, +10 bp via shifted flanks)
  - For each shift, predict forward + reverse-complement
  - Average all 10 predictions per sequence

Compares baseline (RC only) vs TTA-shift (5 shifts × 2 strands) on SNV pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from alphagenome_ft import create_model_with_heads
from scipy.stats import pearsonr, spearmanr

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

REPO = Path(__file__).resolve().parents[2]
SNV_TSV = REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv"

_FLANK_5 = MPRA_UPSTREAM[-220:]
_FLANK_3 = MPRA_DOWNSTREAM[:220]
_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _seq_to_600_with_shift(seq: str, shift: int = 0) -> np.ndarray:
    """600bp window around 200bp insert with `shift` bp offset.
    Positive shift = move insert right (more upstream flank visible)."""
    seq = seq.upper()
    if len(seq) != 200:
        if len(seq) > 200:
            s = (len(seq) - 200) // 2
            seq = seq[s : s + 200]
        else:
            pad = 200 - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    # Use 600bp window with insert centered at 200..400 (default), shift slides it
    # Default: 200 left flank + 200 insert + 200 right flank (we use 200bp flanks here)
    # With shift +s: take left flank ending s bp later, right flank starting s bp later
    left = (
        _FLANK_5[200 - shift - 200 : 200 - shift]
        if shift >= 0
        else _FLANK_5[200 - shift - 200 : 200 - shift]
    )
    right = _FLANK_3[-shift : 200 - shift] if shift <= 0 else _FLANK_3[-shift : 200 - shift]
    if len(left) != 200 or len(right) != 200:
        # Fallback: standard alignment, ignore shift
        left = _FLANK_5[-200:]
        right = _FLANK_3[:200]
    full = left + seq + right
    full = full[:600]
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
            params,
            state,
            sequences,
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


def _predict_batched(predict_step, params, state, x: np.ndarray, batch: int = 256):
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


def _tta_predict(
    predict_step, params, state, seqs: list[str], shifts=(-10, -5, 0, 5, 10), batch: int = 256
):
    """Predict each sequence with all (shift, strand) combos, average."""
    all_preds = []
    for sh in shifts:
        x = np.stack([_seq_to_600_with_shift(s, shift=sh) for s in seqs])
        x_rc = x[:, ::-1, ::-1]
        pf = _predict_batched(predict_step, params, state, x, batch)
        pr = _predict_batched(predict_step, params, state, x_rc, batch)
        all_preds.append(pf)
        all_preds.append(pr)
    return np.mean(np.stack(all_preds), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--shifts", type=int, nargs="+", default=[-10, -5, 0, 5, 10])
    args = ap.parse_args()

    oracle_dir = Path(args.oracle_dir)
    df = pd.read_csv(SNV_TSV, sep="\t")
    print(f"Loaded {len(df):,} SNV pairs; using shifts={args.shifts}")
    ref_seqs = df["sequence_ref"].astype(str).tolist()
    alt_seqs = df["sequence_alt"].astype(str).tolist()
    ref_true = df["K562_log2FC_ref"].to_numpy(np.float32)
    alt_true = df["K562_log2FC_alt"].to_numpy(np.float32)
    delta_true = df["delta_log2FC"].to_numpy(np.float32)

    fold_ref, fold_alt = [], []
    for f in range(args.n_folds):
        ckpt = oracle_dir / f"fold_{f}" / "best_model" / "checkpoint"
        if not ckpt.exists():
            continue
        print(f"\nFold {f}: TTA-predicting...")
        ps, p, s = _build_predict(ckpt)
        rp = _tta_predict(ps, p, s, ref_seqs, shifts=args.shifts)
        ap_ = _tta_predict(ps, p, s, alt_seqs, shifts=args.shifts)
        fold_ref.append(rp)
        fold_alt.append(ap_)
        r_d = pearsonr(ap_ - rp, delta_true)[0]
        print(f"  fold {f}: TTA delta_R={r_d:.3f}")
        del ps, p, s
        jax.clear_caches()

    ref_ens = np.mean(np.stack(fold_ref), axis=0)
    alt_ens = np.mean(np.stack(fold_alt), axis=0)
    delta_ens = alt_ens - ref_ens
    summary = {
        "oracle_dir": str(oracle_dir),
        "n_folds": len(fold_ref),
        "n_pairs": len(ref_seqs),
        "shifts": list(args.shifts),
        "n_tta_views_per_seq": len(args.shifts) * 2,
        "ensemble_ref_R": float(pearsonr(ref_ens, ref_true)[0]),
        "ensemble_alt_R": float(pearsonr(alt_ens, alt_true)[0]),
        "ensemble_delta_R_pearson": float(pearsonr(delta_ens, delta_true)[0]),
        "ensemble_delta_rho_spearman": float(spearmanr(delta_ens, delta_true)[0]),
        "ensemble_delta_mse": float(np.mean((delta_ens - delta_true) ** 2)),
        "per_fold_delta_R": [
            float(pearsonr(fold_alt[i] - fold_ref[i], delta_true)[0]) for i in range(len(fold_ref))
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== SUMMARY ({len(args.shifts)} shifts × 2 strands × {len(fold_ref)} folds) ===")
    print(f"  ENSEMBLE-AVG TTA delta_R: {summary['ensemble_delta_R_pearson']:.3f}")
    print(f"  ENSEMBLE-AVG TTA delta_rho: {summary['ensemble_delta_rho_spearman']:.3f}")
    print(f"  ENSEMBLE-AVG TTA delta_MSE: {summary['ensemble_delta_mse']:.3f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
