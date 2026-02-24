#!/usr/bin/env python
"""Build AlphaGenome encoder embedding cache for the K562 chromosome test set (chr 7, 13).

Run this once (on HPC, from repo root) before batch-evaluating multiple head checkpoints.
Subsequent calls to evaluate_chrom_test(..., cache_dir=...) skip the encoder entirely,
making per-head eval ~10× faster.

Usage:
  # 600 bp (standard, for evaluating existing no_shift / hybrid heads):
  uv run python scripts/analysis/build_test_embedding_cache.py

  # 384 bp compact window (for evaluating compact-window-trained heads):
  uv run python scripts/analysis/build_test_embedding_cache.py \\
      --cache_dir outputs/ag_compact/embedding_cache_compact --seq_len 384
"""

from __future__ import annotations

import argparse

import numpy as np
from alphagenome_ft import create_model_with_heads

from albench.data.k562_full import K562FullDataset
from albench.models.alphagenome_heads import register_s2f_head
from albench.models.embedding_cache import build_embedding_cache

WEIGHTS_PATH = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
# Lightweight dummy head — only the encoder output is saved; head params are irrelevant.
_DUMMY_HEAD = "alphagenome_k562_head_boda_sum_512_512_v4"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/k562", help="K562 data directory.")
    p.add_argument(
        "--cache_dir",
        default="outputs/ag_flatten/embedding_cache",
        help="Directory where test_canonical.npy and test_rc.npy will be written.",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=600,
        help="Input sequence length fed to the encoder (600 for standard, 384 for compact).",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    from pathlib import Path

    cache_dir = Path(args.cache_dir)
    if (cache_dir / "test_canonical.npy").exists() and (cache_dir / "test_rc.npy").exists():
        print(f"[build_test_cache] Cache already exists at {cache_dir}. Nothing to do.")
        return

    print(f"[build_test_cache] Loading AlphaGenome model …")
    register_s2f_head(
        head_name=_DUMMY_HEAD, arch="boda-sum-512-512", task_mode="human", num_tracks=1
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[_DUMMY_HEAD],
        checkpoint_path=WEIGHTS_PATH,
        use_encoder_output=True,
    )

    print(f"[build_test_cache] Loading K562 test split (chr 7, 13) from {args.data_path} …")
    if args.seq_len < 600:
        # Compact window: build fixed-W sequences from raw variable regions, no N-padding.
        ds_test = K562FullDataset(args.data_path, split="test", store_raw=True)
        min_var = int(np.min(ds_test.raw_lengths))
        ds_test.set_compact_window(min_var, window_bp=args.seq_len)
        print(
            f"[build_test_cache] Compact window: min_var={min_var}, W={args.seq_len} bp → T=3 tokens"
        )
    else:
        ds_test = K562FullDataset(args.data_path, split="test")

    build_embedding_cache(
        model,
        ds_test,
        cache_dir,
        "test",
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[build_test_cache] Done. Cache at {cache_dir}")


if __name__ == "__main__":
    main()
