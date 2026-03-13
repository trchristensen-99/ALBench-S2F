#!/usr/bin/env python
"""Generate yeast AlphaGenome oracle pseudo-labels for Experiment 1 test sets.

Loads the yeast AG oracle ensemble (S1, cached head-only) and generates oracle
predictions for the yeast test subsets (random, genomic, SNV).

Output::

    outputs/oracle_pseudolabels_yeast_ag/
        test_oracle_labels.npz
        summary.json

    data/yeast/test_sets_ag/
        random_oracle.npz
        genomic_oracle.npz
        snv_oracle.npz
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(pearsonr(a, b)[0])


def main() -> None:
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import (
        build_encoder_fn,
        build_head_only_predict_fn,
        reinit_head_params,
    )

    oracle_dir = Path(
        os.environ.get(
            "YEAST_AG_ORACLE_DIR",
            str(REPO / "outputs" / "oracle_alphagenome_yeast_ensemble"),
        )
    )
    output_dir = Path(
        os.environ.get(
            "OUTPUT_DIR",
            str(REPO / "outputs" / "oracle_pseudolabels_yeast_ag"),
        )
    )
    data_dir = Path(os.environ.get("YEAST_DATA_PATH", str(REPO / "data" / "yeast")))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup model
    head_name = "yeast_oracle_plabel"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten",
        task_mode="yeast",
        num_tracks=18,
        dropout_rate=0.0,
    )
    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )
    reinit_head_params(model, head_name, num_tokens=3, dim=1536)
    encoder_fn = build_encoder_fn(model)
    head_predict_fn = build_head_only_predict_fn(model, head_name)

    # Yeast flanks (AlphaGenome context)
    yeast_f5 = "GCTAGCGCCGATATCCTAACGAAGTCACTACTACGTACTGCCCTGCACGATAGC"
    yeast_f3 = (
        "CCTGCAGCAGACGTCGACACGCGTCGTAAAGTGACGTTGTCCGAAACCCTT"
        "GCATTCGACACCAAACATTCTCTCAGTGCGTGCCCATGAAC"
    )
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def encode_yeast(seq: str) -> np.ndarray:
        seq = seq.upper()[:150] if len(seq) >= 150 else seq.upper()
        full = yeast_f5 + seq + yeast_f3
        out = np.zeros((384, 4), dtype=np.float32)
        start = max(0, (384 - len(full)) // 2)
        for i, c in enumerate(full):
            if i + start < 384 and c in mapping:
                out[i + start, mapping[c]] = 1.0
        return out

    def predict_ensemble(
        sequences: list[str], batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy.special import softmax

        all_preds = []
        for params in params_list:
            preds = []
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i : i + batch_size]
                x = np.stack([encode_yeast(s) for s in batch])
                emb = encoder_fn(jnp.array(x), jnp.zeros(len(batch), dtype=jnp.int32))
                emb_f32 = jnp.array(emb, dtype=jnp.float32)
                org = jnp.zeros(len(batch), dtype=jnp.int32)
                p = np.array(head_predict_fn(params, emb_f32, org))
                if p.ndim == 2 and p.shape[1] == 18:
                    probs = softmax(p, axis=1)
                    p = (probs * np.arange(18)).sum(axis=1)
                preds.append(p.reshape(-1))
            all_preds.append(np.concatenate(preds))
        arr = np.stack(all_preds, axis=0)
        return arr.mean(axis=0).astype(np.float32), arr.std(axis=0).astype(np.float32)

    # Load oracle checkpoints
    from collections.abc import Mapping as MappingABC

    import orbax.checkpoint as ocp

    def _merge(base, override):
        if not isinstance(override, MappingABC) or not isinstance(base, MappingABC):
            return override
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], MappingABC) and isinstance(v, MappingABC):
                merged[k] = _merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    checkpointer = ocp.StandardCheckpointer()
    params_list = []
    for ckpt_dir in sorted(oracle_dir.glob("oracle_*")):
        ckpt_path = ckpt_dir / "best_model" / "checkpoint"
        if not ckpt_path.exists():
            continue
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(_merge(model._params, loaded_params))
        params_list.append(jax.device_put(model._params))

    print(f"Loaded {len(params_list)} oracle folds from {oracle_dir}")
    if not params_list:
        raise FileNotFoundError(f"No oracle checkpoints found in {oracle_dir}")

    # Load yeast test data
    from data.yeast import YeastDataset
    from evaluation.yeast_testsets import load_yeast_test_subsets

    ds_test = YeastDataset(data_path=str(data_dir), split="test", context_mode="dream150")
    test_seqs = ds_test.sequences
    test_labels = ds_test.labels.astype(np.float32)
    subsets = load_yeast_test_subsets(data_dir / "test_subset_ids")

    # Full test set prediction
    print("Predicting on full test set...")
    t0 = time.time()
    oracle_mean, oracle_std = predict_ensemble([str(s) for s in test_seqs])
    elapsed = time.time() - t0
    print(f"  {len(test_seqs):,} sequences in {elapsed:.1f}s")

    r = _safe_corr(oracle_mean, test_labels)
    print(f"  Full test Pearson r vs true: {r:.4f}")

    np.savez_compressed(
        output_dir / "test_oracle_labels.npz",
        oracle_mean=oracle_mean,
        oracle_std=oracle_std,
        true_label=test_labels,
    )

    summary = {"n_folds": len(params_list), "full_test_pearson": r}

    # Create Exp 1 test set NPZs
    test_npz_dir = data_dir / "test_sets_ag"
    test_npz_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating test set NPZs in {test_npz_dir}")

    idx_random = subsets["random_idx"].astype(int)
    idx_genomic = subsets["genomic_idx"].astype(int)
    snv_pairs = subsets["snv_pairs"].astype(int)

    # Random (in-distribution for yeast)
    np.savez_compressed(
        test_npz_dir / "random_oracle.npz",
        sequences=test_seqs[idx_random],
        oracle_labels=oracle_mean[idx_random],
        oracle_std=oracle_std[idx_random],
        true_labels=test_labels[idx_random],
    )
    r_random = _safe_corr(oracle_mean[idx_random], test_labels[idx_random])
    summary["random"] = {"n": len(idx_random), "pearson_r": r_random}
    print(f"  random_oracle.npz: {len(idx_random):,} seqs, r={r_random:.4f}")

    # Genomic (OOD for yeast)
    np.savez_compressed(
        test_npz_dir / "genomic_oracle.npz",
        sequences=test_seqs[idx_genomic],
        oracle_labels=oracle_mean[idx_genomic],
        oracle_std=oracle_std[idx_genomic],
        true_labels=test_labels[idx_genomic],
    )
    r_genomic = _safe_corr(oracle_mean[idx_genomic], test_labels[idx_genomic])
    summary["genomic"] = {"n": len(idx_genomic), "pearson_r": r_genomic}
    print(f"  genomic_oracle.npz: {len(idx_genomic):,} seqs, r={r_genomic:.4f}")

    # SNV pairs
    if len(snv_pairs) > 0:
        ref_idx = snv_pairs[:, 0]
        alt_idx = snv_pairs[:, 1]
        np.savez_compressed(
            test_npz_dir / "snv_oracle.npz",
            ref_sequences=test_seqs[ref_idx],
            alt_sequences=test_seqs[alt_idx],
            ref_oracle_labels=oracle_mean[ref_idx],
            alt_oracle_labels=oracle_mean[alt_idx],
            delta_oracle_labels=oracle_mean[alt_idx] - oracle_mean[ref_idx],
            ref_oracle_std=oracle_std[ref_idx],
            alt_oracle_std=oracle_std[alt_idx],
            true_ref_labels=test_labels[ref_idx],
            true_alt_labels=test_labels[alt_idx],
            true_delta=test_labels[alt_idx] - test_labels[ref_idx],
        )
        summary["snv"] = {"n_pairs": len(snv_pairs)}
        print(f"  snv_oracle.npz: {len(snv_pairs):,} pairs")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll done. Summary in {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
