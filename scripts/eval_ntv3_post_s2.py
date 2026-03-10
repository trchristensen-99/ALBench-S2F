#!/usr/bin/env python
"""Evaluate NTv3-Borzoi S2 checkpoints (read-only, no training).

Loads best_encoder_state.pkl + best_head.pt from each sweep config and
runs JIT-compiled test evaluation. Safe to run while training is still
running — only reads checkpoint files.

Writes result_eval.json next to each checkpoint dir to avoid overwriting
any result.json that training may write.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from scipy.stats import pearsonr, spearmanr

# ── JAX 0.9 / Flax 0.10 compat fix ──────────────────────────────────────────
_orig_jax_jit = jax.jit


def _patched_jax_jit(fun, *args, **kwargs):
    kwargs.pop("abstracted_axes", None)
    return _orig_jax_jit(fun, *args, **kwargs)


jax.jit = _patched_jax_jit


def evaluate_checkpoint(sweep_dir: Path, encoder, tokenizer, ntv3_config, species_token):
    """Load checkpoint from sweep_dir and run test eval."""
    # Deferred import: must come after JAX monkey-patch above
    from experiments.train_ntv3_stage2 import (
        _FLANK_3_STR,
        _FLANK_5_STR,
        _load_s1_head_into_jax,
        _pad_to_divisible,
        _rc_str,
        _safe_corr,
    )

    enc_path = sweep_dir / "best_encoder_state.pkl"
    head_path = sweep_dir / "best_head.pt"

    if not enc_path.exists() or not head_path.exists():
        print(f"  SKIP {sweep_dir.name}: missing checkpoint files")
        return None

    # Load encoder state
    print(f"  Loading encoder from {enc_path.name} ...", flush=True)
    with open(enc_path, "rb") as f:
        enc_state = pickle.load(f)
    enc_jax = jax.tree_util.tree_map(jnp.asarray, enc_state)
    nnx.update(encoder, enc_jax)

    # Load head
    rngs = nnx.Rngs(0)
    head = _load_s1_head_into_jax(sweep_dir, 1536, 512, 0.1, rngs)

    # Combine for JIT
    class _Combined(nnx.Module):
        def __init__(self, enc, hd):
            self.encoder = enc
            self.mlp_head = hd

    combined = _Combined(encoder, head)
    pad_token_id = tokenizer.pad_token_id
    seq_divisor = 2**ntv3_config.num_downsamples

    @nnx.jit
    def jit_predict(tokens, sp_tokens):
        if sp_tokens is not None:
            outs = combined.encoder(tokens, species_tokens=sp_tokens)
        else:
            outs = combined.encoder(tokens)
        embeddings = outs["embedding"]
        mask = jnp.expand_dims(tokens != pad_token_id, axis=-1)
        masked = embeddings * mask
        lens = jnp.sum(mask, axis=1)
        pooled = jnp.sum(masked, axis=1) / jnp.maximum(lens, 1)
        return combined.mlp_head(pooled, deterministic=True)

    def predict_sequences(seqs_str, batch_size=16):
        """RC-averaged predictions."""
        full_fwd, full_rev = [], []
        for s in seqs_str:
            s = s.upper()
            if len(s) < 200:
                pad = 200 - len(s)
                s = "N" * (pad // 2) + s + "N" * (pad - pad // 2)
            elif len(s) > 200:
                start = (len(s) - 200) // 2
                s = s[start : start + 200]
            fwd = _FLANK_5_STR + s + _FLANK_3_STR
            full_fwd.append(fwd)
            full_rev.append(_rc_str(fwd))

        def _tok(seqs):
            padded = [_pad_to_divisible(s, seq_divisor) for s in seqs]
            return jnp.asarray(tokenizer.batch_np_tokenize(padded))

        def _sp(n):
            return jnp.tile(species_token, (n,)) if species_token is not None else None

        preds_fwd, preds_rev = [], []
        for i in range(0, len(full_fwd), batch_size):
            tf = _tok(full_fwd[i : i + batch_size])
            preds_fwd.append(np.asarray(jit_predict(tf, _sp(tf.shape[0]))).reshape(-1))
            tr = _tok(full_rev[i : i + batch_size])
            preds_rev.append(np.asarray(jit_predict(tr, _sp(tr.shape[0]))).reshape(-1))

        return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0

    # Run test eval
    test_dir = Path("data/k562/test_sets")
    metrics = {}

    for name, path, seq_col, label_col in [
        (
            "in_distribution",
            test_dir / "test_in_distribution_hashfrag.tsv",
            "sequence",
            "K562_log2FC",
        ),
        ("ood", test_dir / "test_ood_designed_k562.tsv", "sequence", "K562_log2FC"),
    ]:
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            pred = predict_sequences(df[seq_col].tolist())
            true = df[label_col].to_numpy(dtype=np.float32)
            metrics[name] = {
                "pearson_r": _safe_corr(pred, true, pearsonr),
                "spearman_r": _safe_corr(pred, true, spearmanr),
                "mse": float(np.mean((pred - true) ** 2)),
                "n": int(len(true)),
            }

    snv_path = test_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        df = pd.read_csv(snv_path, sep="\t")
        ref_pred = predict_sequences(df["sequence_ref"].tolist())
        alt_pred = predict_sequences(df["sequence_alt"].tolist())
        alt_true = df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
        metrics["snv_abs"] = {
            "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
            "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            "n": int(len(alt_true)),
        }
        delta_pred = alt_pred - ref_pred
        delta_true = df["delta_log2FC"].to_numpy(dtype=np.float32)
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            "n": int(len(delta_true)),
        }

    return metrics


def main():
    from nucleotide_transformer_v3.pretrained import get_posttrained_ntv3_model

    print("Loading NTv3 650M post-trained model ...", flush=True)
    t0 = time.time()
    encoder, tokenizer, ntv3_config = get_posttrained_ntv3_model(
        model_name="NTv3_650M_post", use_bfloat16=True
    )
    species_token = encoder.encode_species("human")
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    base = Path("outputs/ntv3_post_k562_stage2")
    sweep_dirs = sorted(base.glob("sweep_*"))
    if not sweep_dirs:
        print("No sweep directories found!")
        return

    all_results = {}
    for sweep_dir in sweep_dirs:
        name = sweep_dir.name
        # Skip if full result.json already exists (training completed)
        if (sweep_dir / "result.json").exists():
            print(f"\n{name}: result.json exists, skipping (training already finished)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 60}")

        metrics = evaluate_checkpoint(sweep_dir, encoder, tokenizer, ntv3_config, species_token)
        if metrics is None:
            continue

        # Save as result_eval.json (not result.json, to avoid conflicts)
        out = {"config": name, "test_metrics": metrics}
        out_path = sweep_dir / "result_eval.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        all_results[name] = metrics
        print(f"  Saved: {out_path}")
        for k, v in metrics.items():
            print(f"    {k}: pearson_r={v['pearson_r']:.4f}  spearman_r={v['spearman_r']:.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':<30} {'InDist':>8} {'SNV':>8} {'Delta':>8} {'OOD':>8}")
    print("-" * 70)
    for name, m in sorted(
        all_results.items(),
        key=lambda x: x[1].get("in_distribution", {}).get("pearson_r", 0),
        reverse=True,
    ):
        id_r = m.get("in_distribution", {}).get("pearson_r", 0)
        snv_r = m.get("snv_abs", {}).get("pearson_r", 0)
        delta_r = m.get("snv_delta", {}).get("pearson_r", 0)
        ood_r = m.get("ood", {}).get("pearson_r", 0)
        print(f"{name:<30} {id_r:>8.4f} {snv_r:>8.4f} {delta_r:>8.4f} {ood_r:>8.4f}")


if __name__ == "__main__":
    main()
