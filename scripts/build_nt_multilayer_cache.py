#!/usr/bin/env python
"""Build Nucleotide Transformer v2 250M multi-layer embedding cache.

Concatenates mean-pooled embeddings from layers 6, 12, 18, 24 to produce
a richer representation (4 × 768 = 3072D) compared to the single-layer
cache (768D).

Cache layout::

    outputs/nt_k562_multilayer/embedding_cache/
        train_canonical.npy   (N_train, 3072)  float16
        train_rc.npy
        ...

Usage::

    uv run --no-sync python scripts/build_nt_multilayer_cache.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from nucleotide_transformer.pretrained import get_pretrained_model
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
    core = _standardize_to_200bp(seq)
    return _FLANK_5 + core + _FLANK_3


LAYERS = (6, 12, 18, 24)
EMBED_DIM_PER_LAYER = 768
TOTAL_DIM = len(LAYERS) * EMBED_DIM_PER_LAYER  # 3072


class NTMultiLayerExtractor:
    """Extract concatenated multi-layer embeddings from NT v2 250M."""

    def __init__(self, max_positions: int = 128):
        parameters, forward_fn, tokenizer, config = get_pretrained_model(
            model_name="250M_multi_species_v2",
            embeddings_layers_to_save=LAYERS,
            max_positions=max_positions,
        )
        self.parameters = parameters
        self.forward_fn = hk.transform(forward_fn)
        self.tokenizer = tokenizer
        self._rng = jax.random.PRNGKey(0)

    def extract(self, sequences: list[str]) -> np.ndarray:
        """Extract concatenated multi-layer embeddings. Returns (B, 3072)."""
        tokens_batch = self.tokenizer.batch_tokenize(sequences)
        token_ids = jnp.asarray([t[1] for t in tokens_batch], dtype=jnp.int32)

        outs = self.forward_fn.apply(self.parameters, self._rng, token_ids)

        # For each layer: remove CLS, masked mean-pool, then concatenate
        layer_embs = []
        for layer in LAYERS:
            emb = outs[f"embeddings_{layer}"]  # (B, T, 768)
            emb_no_cls = emb[:, 1:, :]
            tokens_no_cls = token_ids[:, 1:]
            pad_mask = jnp.expand_dims(tokens_no_cls != self.tokenizer.pad_token_id, axis=-1)
            masked = emb_no_cls * pad_mask
            seq_lens = jnp.sum(pad_mask, axis=1)
            pooled = jnp.sum(masked, axis=1) / jnp.maximum(seq_lens, 1)  # (B, 768)
            layer_embs.append(pooled)

        concatenated = jnp.concatenate(layer_embs, axis=-1)  # (B, 3072)
        return np.asarray(concatenated, dtype=np.float32)


def _encode_and_save(
    extractor: NTMultiLayerExtractor,
    sequences: list[str],
    cache_dir: Path,
    prefix: str,
    batch_size: int = 64,
    dtype: np.dtype = np.float16,
) -> None:
    can_path = cache_dir / f"{prefix}_canonical.npy"
    rc_path = cache_dir / f"{prefix}_rc.npy"

    if can_path.exists() and rc_path.exists():
        print(f"  {prefix}: cache already exists — skipping.")
        return

    N = len(sequences)
    cache_dir.mkdir(parents=True, exist_ok=True)

    can_buf = np.lib.format.open_memmap(can_path, mode="w+", dtype=dtype, shape=(N, TOTAL_DIM))
    rc_buf = np.lib.format.open_memmap(rc_path, mode="w+", dtype=dtype, shape=(N, TOTAL_DIM))

    for i in tqdm(range(0, N, batch_size), desc=f"  {prefix}"):
        batch_seqs = sequences[i : i + batch_size]
        can_seqs = [_make_600bp(s) for s in batch_seqs]
        rc_seqs = [_reverse_complement(s) for s in can_seqs]

        emb_can = extractor.extract(can_seqs)
        emb_rc = extractor.extract(rc_seqs)

        end = min(i + batch_size, N)
        if dtype == np.float16:
            can_buf[i:end] = np.clip(emb_can, -65504, 65504).astype(dtype)
            rc_buf[i:end] = np.clip(emb_rc, -65504, 65504).astype(dtype)
        else:
            can_buf[i:end] = emb_can.astype(dtype)
            rc_buf[i:end] = emb_rc.astype(dtype)

    print(f"  {prefix}: saved {N} embeddings ({TOTAL_DIM}D) → {can_path.name}, {rc_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/k562")
    parser.add_argument("--cache-dir", default="outputs/nt_k562_multilayer/embedding_cache")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()

    from data.k562 import K562Dataset

    print(f"Loading NT v2 250M (multi-layer: {LAYERS})...")
    extractor = NTMultiLayerExtractor(max_positions=128)
    print(f"  Total embed dim: {TOTAL_DIM}")

    cache_dir = Path(args.cache_dir)
    data_path = Path(args.data_path)

    for split in args.splits:
        ds = K562Dataset(data_path=str(data_path), split=split)
        sequences = [ds.sequences[i] for i in range(len(ds))]
        print(f"\n{split}: {len(sequences):,} sequences")
        _encode_and_save(extractor, sequences, cache_dir, split, args.batch_size)

    if args.include_test:
        test_dir = data_path / "test_sets"
        print("\nBuilding test set caches...")

        in_dist_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
        _encode_and_save(
            extractor, in_dist_df["sequence"].tolist(), cache_dir, "test_in_dist", args.batch_size
        )

        snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
        _encode_and_save(
            extractor, snv_df["sequence_ref"].tolist(), cache_dir, "test_snv_ref", args.batch_size
        )
        _encode_and_save(
            extractor, snv_df["sequence_alt"].tolist(), cache_dir, "test_snv_alt", args.batch_size
        )

        ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
        _encode_and_save(
            extractor, ood_df["sequence"].tolist(), cache_dir, "test_ood", args.batch_size
        )

    print(f"\nDone! Cache at {cache_dir}")


if __name__ == "__main__":
    main()
