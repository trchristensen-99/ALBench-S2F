#!/usr/bin/env python
"""Prepare oracle-labeled test sets for Experiment 1.

Creates standardized NPZ test files that use oracle ensemble predictions as
ground-truth labels. All Experiment 1 student evaluations compare against these
oracle labels (not real experimental labels).

Output structure::

    data/k562/test_sets/
        genomic_oracle.npz     — in-distribution hashFrag test (40,718 seqs)
        snv_oracle.npz         — SNV ref + alt pairs (35,226 pairs)
        ood_oracle.npz         — designed high-activity sequences (22,862 seqs)
        random_10k_oracle.npz  — 10K random 200bp sequences (generated here)

    data/yeast/test_sets/
        random_oracle.npz      — random subset of MAUDE test (6,349 seqs)
        genomic_oracle.npz     — genomic/native yeast subset (964 seqs)
        snv_oracle.npz         — SNV ref + alt pairs

Each NPZ contains at minimum:
    sequences: array of DNA strings
    oracle_labels: oracle ensemble mean predictions
    oracle_std: oracle ensemble std

SNV NPZs additionally contain:
    ref_sequences, alt_sequences, ref_oracle_labels, alt_oracle_labels,
    delta_oracle_labels (alt - ref)

Usage (on HPC with GPU)::

    uv run --no-sync python scripts/prepare_exp1_test_sets.py

Set environment variables to override paths:
    K562_DATA_PATH, YEAST_DATA_PATH,
    K562_ORACLE_PSEUDOLABEL_DIR, YEAST_ORACLE_PSEUDOLABEL_DIR
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (defaults for CSHL HPC)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]

K562_DATA = Path(os.environ.get("K562_DATA_PATH", REPO / "data" / "k562"))
YEAST_DATA = Path(os.environ.get("YEAST_DATA_PATH", REPO / "data" / "yeast"))

K562_ORACLE_DIR = Path(
    os.environ.get(
        "K562_ORACLE_PSEUDOLABEL_DIR",
        REPO / "outputs" / "oracle_pseudolabels_stage2_k562_ag",
    )
)
YEAST_ORACLE_DIR = Path(
    os.environ.get(
        "YEAST_ORACLE_PSEUDOLABEL_DIR",
        REPO / "outputs" / "oracle_pseudolabels" / "yeast_dream_oracle_v2",
    )
)

RANDOM_SEED = 42
N_RANDOM_K562 = 10_000
NUCLEOTIDES = np.array(list("ACGT"))


# ---------------------------------------------------------------------------
# K562 test sets
# ---------------------------------------------------------------------------


def _prepare_k562_genomic(out_dir: Path) -> None:
    """Repackage in-distribution hashFrag test set with oracle labels."""
    tsv = K562_DATA / "test_sets" / "test_in_distribution_hashfrag.tsv"
    df = pd.read_csv(tsv, sep="\t")

    pl = dict(np.load(K562_ORACLE_DIR / "test_in_dist_oracle_labels.npz"))
    assert len(df) == len(pl["oracle_mean"]), "Length mismatch"

    np.savez_compressed(
        out_dir / "genomic_oracle.npz",
        sequences=df["sequence"].values,
        oracle_labels=pl["oracle_mean"],
        oracle_std=pl["oracle_std"],
        true_labels=pl["true_label"],
    )
    print(f"  genomic_oracle.npz: {len(df):,} sequences")


def _prepare_k562_snv(out_dir: Path) -> None:
    """Repackage SNV pairs with oracle labels for both ref and alt."""
    tsv = K562_DATA / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    df = pd.read_csv(tsv, sep="\t")

    pl = dict(np.load(K562_ORACLE_DIR / "test_snv_oracle_labels.npz"))

    np.savez_compressed(
        out_dir / "snv_oracle.npz",
        ref_sequences=df["sequence_ref"].values,
        alt_sequences=df["sequence_alt"].values,
        ref_oracle_labels=pl["ref_oracle_mean"],
        alt_oracle_labels=pl["alt_oracle_mean"],
        delta_oracle_labels=pl["delta_oracle_mean"],
        alt_oracle_std=pl["alt_oracle_std"],
        true_delta=pl["true_delta"],
    )
    print(f"  snv_oracle.npz: {len(df):,} pairs")


def _prepare_k562_ood(out_dir: Path) -> None:
    """Repackage OOD designed sequences with oracle labels."""
    tsv = K562_DATA / "test_sets" / "test_ood_designed_k562.tsv"
    df = pd.read_csv(tsv, sep="\t")

    pl = dict(np.load(K562_ORACLE_DIR / "test_ood_oracle_labels.npz"))
    assert len(df) == len(pl["oracle_mean"]), "Length mismatch"

    np.savez_compressed(
        out_dir / "ood_oracle.npz",
        sequences=df["sequence"].values,
        oracle_labels=pl["oracle_mean"],
        oracle_std=pl["oracle_std"],
        true_labels=pl["true_label"],
    )
    print(f"  ood_oracle.npz: {len(df):,} sequences")


def _prepare_k562_random(out_dir: Path) -> None:
    """Generate 10K random 200bp sequences and label with K562 oracle.

    This requires the oracle ensemble to be available for inference.
    If oracle pseudolabels don't cover random sequences (they won't since
    these are newly generated), we need to run inference. This function
    tries to use cached oracle models; if not available, it saves just the
    sequences and prints a warning.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(0, 4, size=(N_RANDOM_K562, 200))
    sequences = np.array(["".join(NUCLEOTIDES[row]) for row in indices])

    # Try to label with oracle
    oracle_labels = _label_k562_sequences_with_oracle(list(sequences))

    if oracle_labels is not None:
        np.savez_compressed(
            out_dir / "random_10k_oracle.npz",
            sequences=sequences,
            oracle_labels=oracle_labels[0],  # mean
            oracle_std=oracle_labels[1],  # std
        )
        print(f"  random_10k_oracle.npz: {N_RANDOM_K562:,} sequences (oracle-labeled)")
    else:
        np.savez_compressed(
            out_dir / "random_10k_oracle.npz",
            sequences=sequences,
            oracle_labels=np.full(N_RANDOM_K562, np.nan, dtype=np.float32),
            oracle_std=np.full(N_RANDOM_K562, np.nan, dtype=np.float32),
        )
        print(
            f"  random_10k_oracle.npz: {N_RANDOM_K562:,} sequences "
            "(NOT labeled — run oracle inference separately)"
        )


def _label_k562_sequences_with_oracle(
    sequences: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Label sequences using the K562 AG oracle ensemble.

    Returns (mean, std) arrays or None if oracle is not loadable.
    """
    try:
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        from alphagenome_ft import create_model_with_heads

        from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
        from models.alphagenome_heads import register_s2f_head
    except ImportError:
        print("  [WARN] AlphaGenome/JAX not available — skipping oracle inference")
        return None

    # Find oracle checkpoints
    oracle_dir = REPO / "outputs" / "ag_hashfrag_oracle_cached"
    if not oracle_dir.exists():
        # Try stage 2 oracle
        oracle_dir = REPO / "outputs" / "stage2_k562_full_train"
    ckpt_paths = sorted(
        [
            p / "best_model" / "checkpoint"
            for p in sorted(oracle_dir.glob("oracle_*"))
            if (p / "best_model" / "checkpoint").exists()
        ]
    )
    if not ckpt_paths:
        print(f"  [WARN] No oracle checkpoints found in {oracle_dir}")
        return None

    print(f"  Loading {len(ckpt_paths)} oracle models for random sequence labeling...")

    # Build MPRA flanks
    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def seq_to_600bp(seq_str: str) -> np.ndarray:
        seq_str = seq_str.upper()
        if len(seq_str) < 200:
            pad = 200 - len(seq_str)
            seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
        elif len(seq_str) > 200:
            start = (len(seq_str) - 200) // 2
            seq_str = seq_str[start : start + 200]
        core = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(seq_str):
            if c in mapping:
                core[i, mapping[c]] = 1.0
        f5 = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(flank_5):
            if c in mapping:
                f5[i, mapping[c]] = 1.0
        f3 = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(flank_3):
            if c in mapping:
                f3[i, mapping[c]] = 1.0
        return np.concatenate([f5, core, f3], axis=0)

    # Register head and build model
    head_name = "boda_flatten_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten",
        task_mode="human",
        num_tracks=1,
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

    @jax.jit
    def predict_step(params, state, seqs):
        return model._predict(
            params,
            state,
            seqs,
            jnp.zeros(len(seqs), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    # Load all oracle params
    from collections.abc import Mapping

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
    params_list = []
    for ckpt_path in ckpt_paths:
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(_merge(model._params, loaded_params))
        params_list.append(jax.device_put(model._params))
    model_state = model._state

    # Predict
    batch_size = 128
    x_fwd = np.stack([seq_to_600bp(s) for s in sequences])
    x_rev = x_fwd[:, ::-1, ::-1]
    n = len(sequences)

    all_preds = []
    for params in params_list:
        preds_fwd, preds_rev = [], []
        for i in range(0, n, batch_size):
            chunk_f = jnp.array(x_fwd[i : i + batch_size])
            chunk_r = jnp.array(x_rev[i : i + batch_size])
            preds_fwd.append(np.array(predict_step(params, model_state, chunk_f)).reshape(-1))
            preds_rev.append(np.array(predict_step(params, model_state, chunk_r)).reshape(-1))
        p = (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0
        all_preds.append(p)

    arr = np.stack(all_preds, axis=0)
    return arr.mean(axis=0).astype(np.float32), arr.std(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Yeast test sets
# ---------------------------------------------------------------------------


def _prepare_yeast_test_sets(out_dir: Path) -> None:
    """Prepare yeast oracle-labeled test sets from existing pseudolabels."""
    from data.yeast import YeastDataset
    from evaluation.yeast_testsets import load_yeast_test_subsets

    ds_test = YeastDataset(data_path=str(YEAST_DATA), split="test", context_mode="dream150")
    test_seqs = ds_test.sequences
    test_labels = ds_test.labels.astype(np.float32)

    pl = dict(np.load(YEAST_ORACLE_DIR / "test_oracle_labels.npz"))
    oracle_mean = pl["oracle_mean"]
    oracle_std = pl["oracle_std"]

    subsets = load_yeast_test_subsets(YEAST_DATA / "test_subset_ids")
    idx_random = subsets["random_idx"].astype(int)
    idx_genomic = subsets["genomic_idx"].astype(int)
    snv_pairs = subsets["snv_pairs"].astype(int)

    # Random (in-distribution)
    np.savez_compressed(
        out_dir / "random_oracle.npz",
        sequences=test_seqs[idx_random],
        oracle_labels=oracle_mean[idx_random],
        oracle_std=oracle_std[idx_random],
        true_labels=test_labels[idx_random],
    )
    print(f"  random_oracle.npz: {len(idx_random):,} sequences")

    # Genomic (OOD)
    np.savez_compressed(
        out_dir / "genomic_oracle.npz",
        sequences=test_seqs[idx_genomic],
        oracle_labels=oracle_mean[idx_genomic],
        oracle_std=oracle_std[idx_genomic],
        true_labels=test_labels[idx_genomic],
    )
    print(f"  genomic_oracle.npz: {len(idx_genomic):,} sequences")

    # SNV pairs
    if len(snv_pairs) > 0:
        ref_idx = snv_pairs[:, 0]
        alt_idx = snv_pairs[:, 1]
        np.savez_compressed(
            out_dir / "snv_oracle.npz",
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
        print(f"  snv_oracle.npz: {len(snv_pairs):,} pairs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Preparing Experiment 1 oracle-labeled test sets ===\n")

    # K562
    k562_out = K562_DATA / "test_sets"
    k562_out.mkdir(parents=True, exist_ok=True)
    print("K562 test sets:")

    if (K562_ORACLE_DIR / "test_in_dist_oracle_labels.npz").exists():
        _prepare_k562_genomic(k562_out)
        _prepare_k562_snv(k562_out)
        _prepare_k562_ood(k562_out)
    else:
        print("  [SKIP] K562 oracle pseudolabels not found at", K562_ORACLE_DIR)

    _prepare_k562_random(k562_out)

    # Yeast
    yeast_out = YEAST_DATA / "test_sets"
    yeast_out.mkdir(parents=True, exist_ok=True)
    print("\nYeast test sets:")

    if (YEAST_ORACLE_DIR / "test_oracle_labels.npz").exists():
        _prepare_yeast_test_sets(yeast_out)
    else:
        print("  [SKIP] Yeast oracle pseudolabels not found at", YEAST_ORACLE_DIR)

    # Summary
    print("\n=== Summary ===")
    for d, name in [(k562_out, "K562"), (yeast_out, "Yeast")]:
        npzs = sorted(d.glob("*_oracle.npz"))
        print(f"{name}: {len(npzs)} oracle test sets in {d}")
        for p in npzs:
            data = dict(np.load(p))
            n = len(next(iter(data.values())))
            print(f"  {p.name}: {n:,} entries, keys={list(data.keys())}")


if __name__ == "__main__":
    main()
