#!/usr/bin/env python
"""Build Borzoi embedding cache for K562 hashFrag data.

Extracts mean-pooled trunk embeddings from the frozen Borzoi encoder.
Sequences are padded with MPRA flanks (200bp → 600bp), then zero-padded
to 196,608bp. Embeddings from get_embs_after_crop() are mean-pooled.

Cache layout::

    outputs/borzoi_k562_cached/embedding_cache/
        train_canonical.npy   (N_train, 1536)  float16
        train_rc.npy
        val_canonical.npy     (N_val, 1536)    float16
        val_rc.npy
        test_*_canonical.npy / test_*_rc.npy

Usage::

    uv run --no-sync python scripts/build_borzoi_embedding_cache.py
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

BORZOI_SEQ_LEN = 196_608


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
    batch_size: int = 2,
    dtype: np.dtype = np.float16,
) -> None:
    """Encode sequences with Borzoi and save canonical + RC caches."""
    can_path = cache_dir / f"{prefix}_canonical.npy"
    rc_path = cache_dir / f"{prefix}_rc.npy"

    if can_path.exists() and rc_path.exists():
        print(f"  {prefix}: cache already exists — skipping.")
        return

    N = len(sequences)
    D = 1536
    cache_dir.mkdir(parents=True, exist_ok=True)

    can_buf = np.lib.format.open_memmap(can_path, mode="w+", dtype=dtype, shape=(N, D))
    rc_buf = np.lib.format.open_memmap(rc_path, mode="w+", dtype=dtype, shape=(N, D))

    for i in tqdm(range(0, N, batch_size), desc=f"  {prefix}"):
        end = min(i + batch_size, N)
        batch_seqs = sequences[i:end]

        # One-hot encode (4, 600) channels-first
        oh_batch = np.stack([_seq_to_one_hot_600(s) for s in batch_seqs])  # (B, 4, 600)
        oh_rc = oh_batch[:, ::-1, ::-1].copy()  # RC: flip channels and sequence

        # Pad to 196,608 (Borzoi uses channels-first: B, 4, L)
        can_t = torch.from_numpy(oh_batch).float()
        rc_t = torch.from_numpy(oh_rc).float()

        pad_total = BORZOI_SEQ_LEN - 600
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        can_padded = torch.nn.functional.pad(can_t, (pad_left, pad_right), value=0.0)
        rc_padded = torch.nn.functional.pad(rc_t, (pad_left, pad_right), value=0.0)

        can_padded = can_padded.to(device)
        rc_padded = rc_padded.to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            emb_can = model.get_embs_after_crop(can_padded)  # (B, 1536, 6144)
            emb_rc = model.get_embs_after_crop(rc_padded)

        # Mean-pool over sequence dimension
        emb_can = emb_can.mean(dim=2)  # (B, 1536)
        emb_rc = emb_rc.mean(dim=2)

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
    parser.add_argument("--cache-dir", default="outputs/borzoi_k562_cached/embedding_cache")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], help="K562Dataset splits to cache."
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    from borzoi_pytorch import Borzoi

    from data.k562 import K562Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Borzoi replicate 0 on {device}...")
    model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify positional embeddings are valid after loading
    from borzoi_pytorch.pytorch_borzoi_transformer import Attention

    for name, mod in model.named_modules():
        if isinstance(mod, Attention):
            if torch.isnan(mod.positions).any():
                print(f"  WARNING: NaN in {name}.positions — recomputing...")
                from borzoi_pytorch.pytorch_borzoi_transformer import get_positional_embed

                mod.positions = get_positional_embed(
                    4096, mod.num_rel_pos_features, mod.to_v.weight.device
                ).to(device)
            else:
                print(f"  {name}.positions OK")

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
        )

        snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
        _encode_and_save(
            model,
            snv_df["sequence_ref"].tolist(),
            cache_dir,
            "test_snv_ref",
            device,
            args.batch_size,
        )
        _encode_and_save(
            model,
            snv_df["sequence_alt"].tolist(),
            cache_dir,
            "test_snv_alt",
            device,
            args.batch_size,
        )

        ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
        _encode_and_save(
            model,
            ood_df["sequence"].tolist(),
            cache_dir,
            "test_ood",
            device,
            args.batch_size,
        )

    print(f"\nDone! Cache at {cache_dir}")


if __name__ == "__main__":
    main()
