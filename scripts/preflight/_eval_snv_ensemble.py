"""Tier 1 SNV evaluation: ensemble-averaged predictions across all 10
folds of the AG-S2 oracle.

Computes the SNV delta correlation when predictions are averaged across
folds (rather than per-fold). This is analogous to MPAC's chromosome-
aware ensemble — averaging multiple independently-trained models reduces
prediction variance and improves the SNV correlation for small-effect
variants.

No genomic context needed (uses existing 200bp ref/alt pairs).

Outputs: results/preflight/snv_ensemble_eval/{baseline,c28_10fold}_summary.json
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

_FLANK_5 = MPRA_UPSTREAM[-200:]
_FLANK_3 = MPRA_DOWNSTREAM[:200]
_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _seq_to_600(seq: str) -> np.ndarray:
    seq = seq.upper()
    if len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > 200:
        s = (len(seq) - 200) // 2
        seq = seq[s : s + 200]
    full = _FLANK_5 + seq + _FLANK_3
    out = np.zeros((600, 4), dtype=np.float32)
    for i, c in enumerate(full):
        if c in _MAP:
            out[i, _MAP[c]] = 1.0
    return out


def _build_predict_step(ckpt_dir: Path, batch_size: int = 256):
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

    loaded_params, _ = ocp.StandardCheckpointer().restore(ckpt_dir)
    model._params = jax.device_put(loaded_params)
    _dummy = jnp.zeros((batch_size, 600, 4), dtype=jnp.float32)
    _ = predict_step(model._params, model._state, _dummy)
    _.block_until_ready()
    return predict_step, model._params, model._state


def _predict_seqs(predict_step, params, state, seqs: list[str], batch_size=256):
    if not seqs:
        return np.array([], dtype=np.float32)
    n = len(seqs)
    x = np.stack([_seq_to_600(s) for s in seqs])
    x_rev = x[:, ::-1, ::-1]
    pf, pr = [], []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        actual = end - i
        bf = x[i:end]
        br = x_rev[i:end]
        if actual < batch_size:
            pad = batch_size - actual
            bf = np.concatenate([bf, np.zeros((pad, 600, 4), dtype=np.float32)])
            br = np.concatenate([br, np.zeros((pad, 600, 4), dtype=np.float32)])
        pf.append(np.array(predict_step(params, state, jnp.array(bf))).reshape(-1)[:actual])
        pr.append(np.array(predict_step(params, state, jnp.array(br))).reshape(-1)[:actual])
    return (np.concatenate(pf) + np.concatenate(pr)) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--oracle_dir",
        type=str,
        required=True,
        help="Path to dir with fold_0/, fold_1/, ... each containing best_model/checkpoint",
    )
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_folds", type=int, default=10)
    args = ap.parse_args()

    oracle_dir = Path(args.oracle_dir)
    print(f"Loading SNV pairs from {SNV_TSV}")
    df = pd.read_csv(SNV_TSV, sep="\t")
    print(f"  loaded {len(df):,} pairs")
    ref_seqs = df["sequence_ref"].astype(str).tolist()
    alt_seqs = df["sequence_alt"].astype(str).tolist()
    ref_true = df["K562_log2FC_ref"].to_numpy(np.float32)
    alt_true = df["K562_log2FC_alt"].to_numpy(np.float32)
    delta_true = df["delta_log2FC"].to_numpy(np.float32)

    fold_ref_preds = []
    fold_alt_preds = []
    for fold in range(args.n_folds):
        ckpt = oracle_dir / f"fold_{fold}" / "best_model" / "checkpoint"
        if not ckpt.exists():
            print(f"  skip fold {fold}: no ckpt at {ckpt}")
            continue
        print(f"\nFold {fold}: predicting...")
        predict_step, params, state = _build_predict_step(ckpt)
        ref_p = _predict_seqs(predict_step, params, state, ref_seqs)
        alt_p = _predict_seqs(predict_step, params, state, alt_seqs)
        fold_ref_preds.append(ref_p)
        fold_alt_preds.append(alt_p)
        # Per-fold metrics
        r_ref = pearsonr(ref_p, ref_true)[0]
        r_alt = pearsonr(alt_p, alt_true)[0]
        r_delta = pearsonr(alt_p - ref_p, delta_true)[0]
        print(f"  fold {fold}: ref_R={r_ref:.3f} alt_R={r_alt:.3f} delta_R={r_delta:.3f}")
        del params, state, predict_step
        jax.clear_caches()

    # Ensemble-averaged predictions
    ref_ens = np.mean(np.stack(fold_ref_preds), axis=0)
    alt_ens = np.mean(np.stack(fold_alt_preds), axis=0)
    delta_ens = alt_ens - ref_ens
    r_ref = pearsonr(ref_ens, ref_true)[0]
    r_alt = pearsonr(alt_ens, alt_true)[0]
    r_delta_pearson = pearsonr(delta_ens, delta_true)[0]
    rho_delta = spearmanr(delta_ens, delta_true)[0]
    mse_delta = float(np.mean((delta_ens - delta_true) ** 2))

    summary = {
        "oracle_dir": str(oracle_dir),
        "n_folds": len(fold_ref_preds),
        "n_pairs": len(ref_seqs),
        "per_fold_delta_R": [
            float(pearsonr(fold_alt_preds[i] - fold_ref_preds[i], delta_true)[0])
            for i in range(len(fold_ref_preds))
        ],
        "ensemble_ref_R": float(r_ref),
        "ensemble_alt_R": float(r_alt),
        "ensemble_delta_R_pearson": float(r_delta_pearson),
        "ensemble_delta_rho_spearman": float(rho_delta),
        "ensemble_delta_mse": mse_delta,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n===== SUMMARY =====")
    print(f"  n_folds: {summary['n_folds']}")
    print(
        f"  per-fold delta_R range: [{min(summary['per_fold_delta_R']):.3f}, {max(summary['per_fold_delta_R']):.3f}]"
    )
    print(f"  per-fold delta_R mean: {np.mean(summary['per_fold_delta_R']):.3f}")
    print(f"  ENSEMBLE-AVG delta_R (Pearson): {r_delta_pearson:.3f}")
    print(f"  ENSEMBLE-AVG delta_R (Spearman): {rho_delta:.3f}")
    print(f"  ENSEMBLE-AVG delta_MSE: {mse_delta:.3f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
