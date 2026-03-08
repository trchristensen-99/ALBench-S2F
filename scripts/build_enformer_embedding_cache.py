#!/usr/bin/env python
"""Build Enformer embedding cache for K562 hashFrag data.

Extracts mean-pooled trunk embeddings from the frozen Enformer encoder.
Sequences are padded with MPRA flanks (200bp → 600bp), then zero-padded
to 196,608bp. Center bins from the trunk output are mean-pooled.

Cache layout::

    outputs/enformer_k562_cached/embedding_cache/
        train_canonical.npy   (N_train, 3072)  float16
        train_rc.npy
        val_canonical.npy     (N_val, 3072)    float16
        val_rc.npy
        test_*_canonical.npy / test_*_rc.npy

Usage::

    uv run --no-sync python scripts/build_enformer_embedding_cache.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from data.utils import one_hot_encode

# ── MPRA flanking sequences ──────────────────────────────────────────────────
_FLANK_5_STR = MPRA_UPSTREAM[-200:]
_FLANK_3_STR = MPRA_DOWNSTREAM[:200]

_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}

_FLANK_5_ENC = np.zeros((4, 200), dtype=np.float32)
for _i, _c in enumerate(_FLANK_5_STR):
    if _c in _MAPPING:
        _FLANK_5_ENC[_MAPPING[_c], _i] = 1.0

_FLANK_3_ENC = np.zeros((4, 200), dtype=np.float32)
for _i, _c in enumerate(_FLANK_3_STR):
    if _c in _MAPPING:
        _FLANK_3_ENC[_MAPPING[_c], _i] = 1.0


def _standardize_to_200bp(sequence: str) -> str:
    target_len = 200
    curr_len = len(sequence)
    if curr_len == target_len:
        return sequence
    if curr_len < target_len:
        pad = target_len - curr_len
        return "N" * (pad // 2) + sequence + "N" * (pad - pad // 2)
    start = (curr_len - target_len) // 2
    return sequence[start : start + target_len]


def _seq_to_one_hot_600(seq: str) -> np.ndarray:
    """Convert sequence to (4, 600) one-hot with MPRA flanks."""
    core = _standardize_to_200bp(seq)
    oh = one_hot_encode(core, add_singleton_channel=False)  # (4, 200)
    return np.concatenate([_FLANK_5_ENC, oh, _FLANK_3_ENC], axis=1)  # (4, 600)


def _encode_and_save(
    model,
    sequences: list[str],
    cache_dir: Path,
    prefix: str,
    device: torch.device,
    batch_size: int = 4,
    n_center_bins: int = 4,
    dtype: np.dtype = np.float16,
) -> None:
    """Encode sequences with Enformer and save canonical + RC caches."""
    from models.enformer_wrapper import ENFORMER_SEQ_LEN, ENFORMER_TARGET_LEN

    can_path = cache_dir / f"{prefix}_canonical.npy"
    rc_path = cache_dir / f"{prefix}_rc.npy"

    if can_path.exists() and rc_path.exists():
        print(f"  {prefix}: cache already exists — skipping.")
        return

    N = len(sequences)
    D = 3072
    cache_dir.mkdir(parents=True, exist_ok=True)

    can_buf = np.lib.format.open_memmap(can_path, mode="w+", dtype=dtype, shape=(N, D))
    rc_buf = np.lib.format.open_memmap(rc_path, mode="w+", dtype=dtype, shape=(N, D))

    center = ENFORMER_TARGET_LEN // 2  # 448
    half = n_center_bins // 2

    for i in tqdm(range(0, N, batch_size), desc=f"  {prefix}"):
        end = min(i + batch_size, N)
        batch_seqs = sequences[i:end]

        # One-hot encode (4, 600) then construct canonical and RC
        oh_batch = np.stack([_seq_to_one_hot_600(s) for s in batch_seqs])  # (B, 4, 600)
        oh_rc = oh_batch[:, ::-1, ::-1].copy()  # RC: flip channels and sequence

        # Convert to (B, 600, 4) channels-last for Enformer
        can_cl = torch.from_numpy(oh_batch.transpose(0, 2, 1)).float()  # (B, 600, 4)
        rc_cl = torch.from_numpy(oh_rc.transpose(0, 2, 1)).float()

        # Pad to 196,608
        pad_total = ENFORMER_SEQ_LEN - 600
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        can_padded = torch.nn.functional.pad(can_cl, (0, 0, pad_left, pad_right), value=0.0)
        rc_padded = torch.nn.functional.pad(rc_cl, (0, 0, pad_left, pad_right), value=0.0)

        can_padded = can_padded.to(device)
        rc_padded = rc_padded.to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            emb_can = model(can_padded, return_only_embeddings=True)  # (B, 896, 3072)
            emb_rc = model(rc_padded, return_only_embeddings=True)

        # Extract center bins and mean-pool
        emb_can = emb_can[:, center - half : center + half, :].mean(dim=1)  # (B, 3072)
        emb_rc = emb_rc[:, center - half : center + half, :].mean(dim=1)

        emb_can_np = emb_can.cpu().float().numpy()
        emb_rc_np = emb_rc.cpu().float().numpy()

        if dtype == np.float16:
            can_buf[i:end] = np.clip(emb_can_np, -65504, 65504).astype(dtype)
            rc_buf[i:end] = np.clip(emb_rc_np, -65504, 65504).astype(dtype)
        else:
            can_buf[i:end] = emb_can_np.astype(dtype)
            rc_buf[i:end] = emb_rc_np.astype(dtype)

    print(f"  {prefix}: saved {N} embeddings ({D}D) → {can_path.name}, {rc_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/k562")
    parser.add_argument("--cache-dir", default="outputs/enformer_k562_cached/embedding_cache")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], help="K562Dataset splits to cache."
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--n-center-bins", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    from enformer_pytorch import Enformer

    from data.k562 import K562Dataset

    # Fix for transformers>=5.3 which expects all_tied_weights_keys on PreTrainedModel
    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = getattr(Enformer, "_tied_weights_keys", set())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Enformer on {device}...")
    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    cache_dir = Path(args.cache_dir)
    data_path = Path(args.data_path)

    for split in args.splits:
        ds = K562Dataset(data_path=str(data_path), split=split)
        sequences = [ds.sequences[i] for i in range(len(ds))]
        print(f"\n{split}: {len(sequences):,} sequences")
        _encode_and_save(
            model,
            sequences,
            cache_dir,
            split,
            device,
            args.batch_size,
            args.n_center_bins,
        )

    if args.include_test:
        test_dir = data_path / "test_sets"
        print("\nBuilding test set caches...")

        in_dist_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
        _encode_and_save(
            model,
            in_dist_df["sequence"].tolist(),
            cache_dir,
            "test_in_dist",
            device,
            args.batch_size,
            args.n_center_bins,
        )

        snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
        _encode_and_save(
            model,
            snv_df["sequence_ref"].tolist(),
            cache_dir,
            "test_snv_ref",
            device,
            args.batch_size,
            args.n_center_bins,
        )
        _encode_and_save(
            model,
            snv_df["sequence_alt"].tolist(),
            cache_dir,
            "test_snv_alt",
            device,
            args.batch_size,
            args.n_center_bins,
        )

        ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
        _encode_and_save(
            model,
            ood_df["sequence"].tolist(),
            cache_dir,
            "test_ood",
            device,
            args.batch_size,
            args.n_center_bins,
        )

    print(f"\nDone! Cache at {cache_dir}")


if __name__ == "__main__":
    main()
