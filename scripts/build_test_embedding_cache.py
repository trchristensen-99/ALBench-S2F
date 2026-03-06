#!/usr/bin/env python
"""Build embedding cache for K562 hashFrag test sets.

Since the AlphaGenome encoder is frozen across all oracle models,
the test set embeddings are identical for every fold.  Pre-computing
them avoids the ~2h JIT compilation penalty and ~30 min/fold encoder
inference in the pseudolabel generation script.

Output (under cache_dir):
    test_in_dist_canonical.npy  (40718, T, D)
    test_in_dist_rc.npy
    test_snv_ref_canonical.npy  (35226, T, D)
    test_snv_ref_rc.npy
    test_snv_alt_canonical.npy  (35226, T, D)
    test_snv_alt_rc.npy
    test_ood_canonical.npy      (22862, T, D)
    test_ood_rc.npy

Usage::

    sbatch scripts/slurm/build_test_embedding_cache.sh
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from alphagenome_ft import create_model_with_heads
from tqdm import tqdm

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import build_encoder_fn

# ── MPRA flanks ──────────────────────────────────────────────────────────────
_FLANK_5_STR = MPRA_UPSTREAM[-200:]
_FLANK_3_STR = MPRA_DOWNSTREAM[:200]
_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def _one_hot(seq: str, length: int = 200) -> np.ndarray:
    seq = seq.upper()
    if len(seq) < length:
        pad = length - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > length:
        start = (len(seq) - length) // 2
        seq = seq[start : start + length]
    arr = np.zeros((length, 4), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in _MAPPING:
            arr[i, _MAPPING[c]] = 1.0
    return arr


_FLANK_5_ENC = _one_hot(_FLANK_5_STR, 200)
_FLANK_3_ENC = _one_hot(_FLANK_3_STR, 200)


def _seq_to_600bp(seq_str: str) -> np.ndarray:
    core = _one_hot(seq_str, 200)
    return np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)


def _encode_and_save(
    encoder_fn,
    seqs: list[str],
    cache_dir: Path,
    prefix: str,
    batch_size: int = 256,
    dtype: np.dtype = np.float16,
) -> None:
    """Encode sequences and save canonical + RC caches."""
    can_path = cache_dir / f"{prefix}_canonical.npy"
    rc_path = cache_dir / f"{prefix}_rc.npy"

    if can_path.exists() and rc_path.exists():
        print(f"  {prefix}: cache already exists — skipping.")
        return

    N = len(seqs)
    x_fwd = np.stack([_seq_to_600bp(s) for s in tqdm(seqs, desc=f"  {prefix} one-hot")])
    x_rev = x_fwd[:, ::-1, ::-1]

    # Determine T, D from a dummy forward pass
    dummy = jnp.zeros((1, 600, 4), dtype=jnp.float32)
    dummy_org = jnp.zeros((1,), dtype=jnp.int32)
    sample = np.array(encoder_fn(dummy, dummy_org))
    T, D = sample.shape[1], sample.shape[2]

    can_buf = np.lib.format.open_memmap(can_path, mode="w+", dtype=dtype, shape=(N, T, D))
    rc_buf = np.lib.format.open_memmap(rc_path, mode="w+", dtype=dtype, shape=(N, T, D))

    for i in tqdm(range(0, N, batch_size), desc=f"  {prefix} encode"):
        end = min(i + batch_size, N)
        org_idx = jnp.zeros((end - i,), dtype=jnp.int32)

        emb_can = np.array(encoder_fn(jnp.array(x_fwd[i:end]), org_idx))
        emb_rc = np.array(encoder_fn(jnp.array(x_rev[i:end]), org_idx))

        if dtype == np.float16:
            can_buf[i:end] = np.clip(emb_can, -65504, 65504).astype(dtype)
            rc_buf[i:end] = np.clip(emb_rc, -65504, 65504).astype(dtype)
        else:
            can_buf[i:end] = emb_can.astype(dtype)
            rc_buf[i:end] = emb_rc.astype(dtype)

    print(f"  {prefix}: saved {N} embeddings → {can_path.name}, {rc_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/ag_hashfrag/embedding_cache"),
    )
    parser.add_argument("--k562-data-path", type=Path, default=Path("data/k562"))
    parser.add_argument(
        "--weights-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype. float32 avoids precision loss from bfloat16→float16 truncation.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    _DTYPE_MAP = {"float16": np.float16, "float32": np.float32}
    storage_dtype = _DTYPE_MAP[args.dtype]

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    weights_path = str(Path(args.weights_path).expanduser().resolve())

    # Need a head registered to create model (head params are unused for encoder-only)
    register_s2f_head(
        head_name="alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4",
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.0,
    )
    model = create_model_with_heads(
        "all_folds",
        heads=["alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )

    encoder_fn = build_encoder_fn(model)
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    test_dir = args.k562_data_path / "test_sets"

    print(f"Building test set embedding caches in {cache_dir}")
    print(f"Test set directory: {test_dir}")

    # ── In-distribution test set ──────────────────────────────────────────────
    in_dist_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    _encode_and_save(
        encoder_fn,
        in_dist_df["sequence"].tolist(),
        cache_dir,
        "test_in_dist",
        args.batch_size,
        dtype=storage_dtype,
    )

    # ── SNV pairs (ref + alt separately) ──────────────────────────────────────
    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    _encode_and_save(
        encoder_fn,
        snv_df["sequence_ref"].tolist(),
        cache_dir,
        "test_snv_ref",
        args.batch_size,
        dtype=storage_dtype,
    )
    _encode_and_save(
        encoder_fn,
        snv_df["sequence_alt"].tolist(),
        cache_dir,
        "test_snv_alt",
        args.batch_size,
        dtype=storage_dtype,
    )

    # ── OOD designed CRE ──────────────────────────────────────────────────────
    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
    _encode_and_save(
        encoder_fn,
        ood_df["sequence"].tolist(),
        cache_dir,
        "test_ood",
        args.batch_size,
        dtype=storage_dtype,
    )

    print("\nDone! All test set embedding caches built.")


if __name__ == "__main__":
    main()
