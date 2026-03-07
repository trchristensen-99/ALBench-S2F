#!/usr/bin/env python
"""Build Nucleotide Transformer v3 650M embedding cache for K562 hashFrag data.

Extracts mean-pooled final-layer embeddings from the frozen NTv3 encoder.
Each sequence is tokenized at single-base resolution, passed through the
U-Net + transformer architecture, and the output is mean-pooled (excluding
padding) to produce a single embedding vector per sequence.

Sequences are padded to 640bp (next multiple of 128 above 600bp) for U-Net
compatibility (7 downsamples → divisor = 2^7 = 128).

Cache layout::

    outputs/ntv3_k562_cached/embedding_cache/
        train_canonical.npy   (N_train, 1536)  float16
        train_rc.npy
        val_canonical.npy     (N_val, 1536)    float16
        val_rc.npy
        test_in_dist_canonical.npy   (40718, 1536)
        test_in_dist_rc.npy
        test_snv_ref_canonical.npy   (35226, 1536)
        test_snv_ref_rc.npy
        test_snv_alt_canonical.npy   (35226, 1536)
        test_snv_alt_rc.npy
        test_ood_canonical.npy       (22862, 1536)
        test_ood_rc.npy

Usage::

    uv run --no-sync python scripts/build_ntv3_embedding_cache.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Reverse complement helper ────────────────────────────────────────────────
_RC_MAP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_MAP)[::-1]


def _standardize_to_200bp(sequence: str) -> str:
    target_len = 200
    curr_len = len(sequence)
    if curr_len == target_len:
        return sequence
    if curr_len < target_len:
        pad_needed = target_len - curr_len
        left_pad = pad_needed // 2
        right_pad = pad_needed - left_pad
        return "N" * left_pad + sequence + "N" * right_pad
    start = (curr_len - target_len) // 2
    return sequence[start : start + target_len]


# ── MPRA flanking sequences ──────────────────────────────────────────────────
_MPRA_UPSTREAM = (
    "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTAC"
    "GTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTAT"
    "TTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAA"
    "ACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCC"
    "TAACTGGCCGCTTGACG"
)
_MPRA_DOWNSTREAM = (
    "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCC"
    "TCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTG"
    "TTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCT"
    "GGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAG"
    "CTGACCCTGAAGTTCATCT"
)
_FLANK_5 = _MPRA_UPSTREAM[-200:]
_FLANK_3 = _MPRA_DOWNSTREAM[:200]


def _make_600bp(seq: str) -> str:
    """Pad a sequence to 200bp then add MPRA flanks for 600bp total."""
    core = _standardize_to_200bp(seq)
    return _FLANK_5 + core + _FLANK_3


def _encode_sequences(
    ntv3_wrapper,
    sequences: list[str],
    cache_dir: Path,
    prefix: str,
    batch_size: int = 32,
    dtype: np.dtype = np.float16,
    use_flanks: bool = True,
) -> None:
    """Encode sequences with NTv3 and save canonical + RC caches."""
    can_path = cache_dir / f"{prefix}_canonical.npy"
    rc_path = cache_dir / f"{prefix}_rc.npy"

    if can_path.exists() and rc_path.exists():
        print(f"  {prefix}: cache already exists — skipping.")
        return

    N = len(sequences)
    D = ntv3_wrapper.embed_dim
    cache_dir.mkdir(parents=True, exist_ok=True)

    can_buf = np.lib.format.open_memmap(can_path, mode="w+", dtype=dtype, shape=(N, D))
    rc_buf = np.lib.format.open_memmap(rc_path, mode="w+", dtype=dtype, shape=(N, D))

    for i in tqdm(range(0, N, batch_size), desc=f"  {prefix}"):
        batch_seqs = sequences[i : i + batch_size]

        if use_flanks:
            can_seqs = [_make_600bp(s) for s in batch_seqs]
            rc_seqs = [_reverse_complement(_make_600bp(s)) for s in batch_seqs]
        else:
            can_seqs = [_standardize_to_200bp(s) for s in batch_seqs]
            rc_seqs = [_reverse_complement(_standardize_to_200bp(s)) for s in batch_seqs]

        emb_can = ntv3_wrapper.extract_embeddings(can_seqs)  # (B, D) float32
        emb_rc = ntv3_wrapper.extract_embeddings(rc_seqs)

        end = min(i + batch_size, N)
        if dtype == np.float16:
            can_buf[i:end] = np.clip(emb_can, -65504, 65504).astype(dtype)
            rc_buf[i:end] = np.clip(emb_rc, -65504, 65504).astype(dtype)
        else:
            can_buf[i:end] = emb_can.astype(dtype)
            rc_buf[i:end] = emb_rc.astype(dtype)

    print(f"  {prefix}: saved {N} embeddings ({D}D) → {can_path.name}, {rc_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/k562")
    parser.add_argument("--cache-dir", default="outputs/ntv3_k562_cached/embedding_cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="K562Dataset splits to cache (train, val).",
    )
    parser.add_argument("--include-test", action="store_true", help="Also cache test sets.")
    parser.add_argument("--no-flanks", action="store_true", help="Use 200bp only, no MPRA flanks.")
    args = parser.parse_args()

    from data.k562 import K562Dataset
    from models.nt_v3_wrapper import NTv3Wrapper

    print("Loading Nucleotide Transformer v3 650M (pre-trained)...")
    ntv3 = NTv3Wrapper(model_name="NTv3_650M_pre", use_bfloat16=True)
    print(f"  Embed dim: {ntv3.embed_dim}")

    cache_dir = Path(args.cache_dir)
    data_path = Path(args.data_path)
    use_flanks = not args.no_flanks

    for split in args.splits:
        ds = K562Dataset(data_path=str(data_path), split=split)
        sequences = [ds.sequences[i] for i in range(len(ds))]
        print(f"\n{split}: {len(sequences):,} sequences")
        _encode_sequences(ntv3, sequences, cache_dir, split, args.batch_size, use_flanks=use_flanks)

    if args.include_test:
        test_dir = data_path / "test_sets"
        print("\nBuilding test set caches...")

        in_dist_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
        _encode_sequences(
            ntv3,
            in_dist_df["sequence"].tolist(),
            cache_dir,
            "test_in_dist",
            args.batch_size,
            use_flanks=use_flanks,
        )

        snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
        _encode_sequences(
            ntv3,
            snv_df["sequence_ref"].tolist(),
            cache_dir,
            "test_snv_ref",
            args.batch_size,
            use_flanks=use_flanks,
        )
        _encode_sequences(
            ntv3,
            snv_df["sequence_alt"].tolist(),
            cache_dir,
            "test_snv_alt",
            args.batch_size,
            use_flanks=use_flanks,
        )

        ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
        _encode_sequences(
            ntv3,
            ood_df["sequence"].tolist(),
            cache_dir,
            "test_ood",
            args.batch_size,
            use_flanks=use_flanks,
        )

    print(f"\nDone! Cache at {cache_dir}")


if __name__ == "__main__":
    main()
