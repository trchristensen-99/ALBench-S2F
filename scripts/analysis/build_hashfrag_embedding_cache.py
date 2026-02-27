#!/usr/bin/env python
"""Build AlphaGenome encoder embedding cache for K562 hashFrag splits.

Runs the AlphaGenome encoder once over the requested hashFrag splits
(train / pool / val) and writes canonical + RC embedding caches.
This eliminates the dominant encoder cost in subsequent training runs,
enabling ~20–50× faster per-epoch training when aug_mode=no_shift.

Cache layout::

    outputs/ag_hashfrag/embedding_cache/
        train_canonical.npy   shape (N_train, T=5, D=1536)  float16
        train_rc.npy
        pool_canonical.npy    shape (N_pool,  T=5, D=1536)  float16
        pool_rc.npy
        val_canonical.npy     shape (N_val,   T=5, D=1536)  float16
        val_rc.npy

Each split takes roughly 10–30 minutes on an H100 NVL (batch_size=128).
Total for train+pool+val (~330K seqs): ~45–75 minutes.

Usage::

    uv run --no-sync python scripts/analysis/build_hashfrag_embedding_cache.py

    # custom cache dir:
    uv run --no-sync python scripts/analysis/build_hashfrag_embedding_cache.py \\
        --cache_dir outputs/ag_hashfrag/embedding_cache_f32 --dtype float32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from alphagenome_ft import create_model_with_heads

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import build_embedding_cache

WEIGHTS_PATH = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
# Lightweight dummy head — only encoder output is needed; head params are irrelevant.
_DUMMY_HEAD = "alphagenome_k562_head_boda_sum_512_512_v4"
_DTYPE_MAP = {"float16": np.float16, "float32": np.float32}

# ── Pre-encode MPRA flanks (200 bp each) ────────────────────────────────────
_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}

_FLANK_5_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(MPRA_UPSTREAM[-200:]):
    if _c in _MAPPING:
        _FLANK_5_ENC[_i, _MAPPING[_c]] = 1.0

_FLANK_3_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(MPRA_DOWNSTREAM[:200]):
    if _c in _MAPPING:
        _FLANK_3_ENC[_i, _MAPPING[_c]] = 1.0


# ── Dataset wrapper ──────────────────────────────────────────────────────────


class HashFragCacheDataset(torch.utils.data.Dataset):
    """Wraps K562Dataset, returning (4, 600) MPRA-context tensors.

    K562Dataset returns (5, 200) tensors (200 bp one-hot + RC channel).
    This wrapper prepends/appends the 200 bp MPRA flanks to produce the
    full 600 bp context expected by the AlphaGenome encoder, in the
    (4, 600) layout that ``build_embedding_cache`` expects.
    """

    def __init__(self, k562_dataset: K562Dataset) -> None:
        self.ds = k562_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple:
        seq_tensor, label = self.ds[idx]
        core = seq_tensor.numpy()[:4, :].T  # (200, 4) — strip RC channel, transpose
        full_seq = np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)  # (600, 4)
        # Return (4, 600) to match the [:4, :].T access pattern in build_embedding_cache
        return torch.tensor(full_seq.T, dtype=torch.float32), label


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/k562", help="K562 hashFrag data directory.")
    p.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Output directory for .npy cache files. "
            "Defaults to outputs/ag_hashfrag/embedding_cache (float16) "
            "or outputs/ag_hashfrag/embedding_cache_f32 (float32)."
        ),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "pool", "val"],
        help="Which K562Dataset splits to cache. Default: train pool val.",
    )
    p.add_argument(
        "--dtype",
        default="float16",
        choices=list(_DTYPE_MAP),
        help=(
            "Storage dtype. float16 halves disk/RAM usage (default). "
            "float32 avoids clipping — needed if training with full_aug on cached embeddings."
        ),
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    if args.cache_dir is None:
        args.cache_dir = (
            "outputs/ag_hashfrag/embedding_cache_f32"
            if args.dtype == "float32"
            else "outputs/ag_hashfrag/embedding_cache"
        )
    cache_dir = Path(args.cache_dir)

    # Check if all requested splits are already cached
    all_cached = all(
        (cache_dir / f"{s}_canonical.npy").exists() and (cache_dir / f"{s}_rc.npy").exists()
        for s in args.splits
    )
    if all_cached:
        print(f"[build_hashfrag_cache] All splits already cached at {cache_dir}. Nothing to do.")
        return

    print(f"[build_hashfrag_cache] Loading AlphaGenome model …")
    register_s2f_head(
        head_name=_DUMMY_HEAD, arch="boda-sum-512-512", task_mode="human", num_tracks=1
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[_DUMMY_HEAD],
        checkpoint_path=WEIGHTS_PATH,
        use_encoder_output=True,
    )

    for split_name in args.splits:
        out_can = cache_dir / f"{split_name}_canonical.npy"
        out_rc = cache_dir / f"{split_name}_rc.npy"
        if out_can.exists() and out_rc.exists():
            print(f"[build_hashfrag_cache] {split_name}: already exists — skipping.")
            continue

        print(f"\n[build_hashfrag_cache] Loading K562Dataset split='{split_name}' …")
        ds_raw = K562Dataset(data_path=args.data_path, split=split_name)
        ds_wrapped = HashFragCacheDataset(ds_raw)
        print(f"[build_hashfrag_cache] {split_name}: {len(ds_wrapped):,} sequences")

        build_embedding_cache(
            model,
            ds_wrapped,
            cache_dir,
            split_name,
            max_seq_len=600,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dtype=dtype,
        )

    print(f"\n[build_hashfrag_cache] Done. Cache at {cache_dir} (dtype={args.dtype})")


if __name__ == "__main__":
    main()
