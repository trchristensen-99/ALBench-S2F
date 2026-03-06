#!/usr/bin/env python
"""Concatenate chunked embedding cache files into a single cache.

After parallel cache building (--chunk_id/--num_chunks), run this to
merge chunk_0/, chunk_1/, ... into the final train_canonical.npy and
train_rc.npy files.

Usage:
    python scripts/analysis/concat_yeast_cache_chunks.py \
        --cache_dir outputs/ag_yeast/embedding_cache_full \
        --num_chunks 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", required=True, help="Cache directory containing chunk_*/ dirs.")
    p.add_argument("--num_chunks", type=int, required=True)
    p.add_argument("--split", default="train")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    split = args.split

    out_can = cache_dir / f"{split}_canonical.npy"
    out_rc = cache_dir / f"{split}_rc.npy"

    if out_can.exists() and out_rc.exists():
        print(f"Final cache already exists: {out_can}, {out_rc}")
        return

    # Verify all chunks exist
    chunk_files_can = []
    chunk_files_rc = []
    total_rows = 0
    for i in range(args.num_chunks):
        chunk_dir = cache_dir / f"chunk_{i}"
        f_can = chunk_dir / f"{split}_canonical.npy"
        f_rc = chunk_dir / f"{split}_rc.npy"
        if not f_can.exists() or not f_rc.exists():
            raise FileNotFoundError(f"Missing chunk {i}: {f_can} or {f_rc}")
        arr = np.load(f_can, mmap_mode="r")
        chunk_files_can.append(f_can)
        chunk_files_rc.append(f_rc)
        total_rows += arr.shape[0]
        print(f"  chunk_{i}: {arr.shape[0]:,} rows, shape={arr.shape}")

    # Get dtype and shape from first chunk
    sample = np.load(chunk_files_can[0], mmap_mode="r")
    T, D = sample.shape[1], sample.shape[2]
    dtype = sample.dtype
    print(
        f"\nConcatenating {args.num_chunks} chunks: {total_rows:,} total rows, T={T}, D={D}, dtype={dtype}"
    )

    # Write concatenated files using memmap (low RAM usage)
    for label, chunk_files, out_path in [
        ("canonical", chunk_files_can, out_can),
        ("rc", chunk_files_rc, out_rc),
    ]:
        print(f"  Writing {label} → {out_path}")
        buf = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(total_rows, T, D))
        ptr = 0
        for f in chunk_files:
            chunk = np.load(f, mmap_mode="r")
            n = chunk.shape[0]
            buf[ptr : ptr + n] = chunk
            ptr += n
        del buf
        print(f"    Done: {ptr:,} rows written")

    # Clean up chunk directories
    print("\nCleaning up chunk directories...")
    for i in range(args.num_chunks):
        chunk_dir = cache_dir / f"chunk_{i}"
        for f in chunk_dir.iterdir():
            f.unlink()
        chunk_dir.rmdir()
        print(f"  Removed {chunk_dir}")

    print(f"\nFinal cache at {cache_dir}: {total_rows:,} rows")


if __name__ == "__main__":
    main()
