#!/usr/bin/env python
"""Build AlphaGenome encoder embedding cache for yeast splits (train/val)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from alphagenome_ft import create_model_with_heads
from torch.utils.data import Subset

from data.yeast import YeastDataset
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import build_embedding_cache

WEIGHTS_PATH = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
_DUMMY_HEAD = "alphagenome_yeast_head_boda_flatten_512_512_v4"
_DTYPE_MAP = {"float16": np.float16, "float32": np.float32}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/yeast", help="Yeast data directory.")
    p.add_argument(
        "--cache_dir",
        default="outputs/ag_yeast/embedding_cache",
        help="Output directory for cache .npy files.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to cache. Default: train val.",
    )
    p.add_argument(
        "--dtype",
        default="float16",
        choices=list(_DTYPE_MAP),
        help="Storage dtype for cache arrays.",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Only cache the first N sequences (for limited cache when disk is tight).",
    )
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    dtype = _DTYPE_MAP[args.dtype]

    all_cached = all(
        (cache_dir / f"{s}_canonical.npy").exists() and (cache_dir / f"{s}_rc.npy").exists()
        for s in args.splits
    )
    if all_cached:
        print(f"[build_yeast_cache] All requested splits already cached at {cache_dir}.")
        return

    register_s2f_head(
        head_name=_DUMMY_HEAD,
        arch="boda-flatten-512-512",
        task_mode="yeast",
        num_tracks=18,
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[_DUMMY_HEAD],
        checkpoint_path=WEIGHTS_PATH,
        use_encoder_output=True,
        detach_backbone=True,
    )

    for split_name in args.splits:
        out_can = cache_dir / f"{split_name}_canonical.npy"
        out_rc = cache_dir / f"{split_name}_rc.npy"
        if out_can.exists() and out_rc.exists():
            print(f"[build_yeast_cache] {split_name}: exists, skipping.")
            continue

        ds_full = YeastDataset(
            data_path=args.data_path,
            split=split_name,
            context_mode="alphagenome384",
        )
        if args.max_sequences and split_name == "train" and args.max_sequences < len(ds_full):
            ds = Subset(ds_full, range(args.max_sequences))
            print(
                f"[build_yeast_cache] {split_name}: {len(ds):,} of {len(ds_full):,} sequences"
                f" (limited by --max_sequences)"
            )
        else:
            ds = ds_full
            print(f"[build_yeast_cache] {split_name}: {len(ds):,} sequences")
        build_embedding_cache(
            model,
            ds,
            cache_dir,
            split_name,
            max_seq_len=384,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dtype=dtype,
        )

    print(f"[build_yeast_cache] Done. Cache at {cache_dir} (dtype={args.dtype}).")


if __name__ == "__main__":
    main()
