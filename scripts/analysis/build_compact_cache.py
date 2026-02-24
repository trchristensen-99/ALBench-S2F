#!/usr/bin/env python
"""Build 384-bp compact-window encoder embedding cache for K562 train and val splits.

Run via scripts/slurm/build_compact_cache.sh BEFORE submitting compact training jobs.
The four compact head training jobs (sum/mean/max/center) all share this cache via
--dependency=afterok so they never race to build it simultaneously.

Output:
    outputs/ag_compact/embedding_cache_compact/train_canonical.npy  (N_train, 3, 1536)
    outputs/ag_compact/embedding_cache_compact/train_rc.npy
    outputs/ag_compact/embedding_cache_compact/val_canonical.npy    (N_val, 3, 1536)
    outputs/ag_compact/embedding_cache_compact/val_rc.npy

Storage: ~627k train × 3 × 1536 × 2 B × 2 (can+rc) ≈ 7.3 GB total (vs 21.5 GB for T=5).
"""

from __future__ import annotations

import argparse

import numpy as np
from alphagenome_ft import create_model_with_heads

from albench.data.k562_full import K562FullDataset
from albench.models.alphagenome_heads import register_s2f_head
from albench.models.embedding_cache import build_embedding_cache

WEIGHTS_PATH = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
_DUMMY_HEAD = "alphagenome_k562_head_boda_sum_512_512_v4"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/k562")
    p.add_argument("--cache_dir", default="outputs/ag_compact/embedding_cache_compact")
    p.add_argument("--seq_len", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    from pathlib import Path

    cache_dir = Path(args.cache_dir)

    print(f"[build_compact_cache] Loading AlphaGenome model …")
    register_s2f_head(
        head_name=_DUMMY_HEAD, arch="boda-sum-512-512", task_mode="human", num_tracks=1
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[_DUMMY_HEAD],
        checkpoint_path=WEIGHTS_PATH,
        use_encoder_output=True,
    )

    for split in ("train", "val"):
        can_path = cache_dir / f"{split}_canonical.npy"
        rc_path = cache_dir / f"{split}_rc.npy"
        if can_path.exists() and rc_path.exists():
            print(f"[build_compact_cache] {split} cache already exists – skipping.")
            continue

        print(f"[build_compact_cache] Loading K562 {split} split from {args.data_path} …")
        ds = K562FullDataset(args.data_path, split=split, store_raw=True)
        min_var = int(np.min(ds.raw_lengths))
        ds.set_compact_window(min_var, window_bp=args.seq_len)
        print(
            f"[build_compact_cache] Compact window: min_var={min_var}, "
            f"W={ds.sequence_length} bp → T≈3 tokens. N={len(ds)}"
        )

        build_embedding_cache(
            model,
            ds,
            cache_dir,
            split,
            max_seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print(f"[build_compact_cache] All done. Cache at {cache_dir}")


if __name__ == "__main__":
    main()
