#!/usr/bin/env python3
"""Check embedding cache alignment with test set files."""

from pathlib import Path

import numpy as np
import pandas as pd


def check_alignment():
    models = {
        "enformer": ("outputs/enformer_k562_cached/embedding_cache", 3072),
        "borzoi": ("outputs/borzoi_k562_cached/embedding_cache", 1536),
        "ntv3_post": ("outputs/ntv3_post_k562_cached/embedding_cache", 1536),
    }

    test_files = {
        "test_in_dist": "data/k562/test_sets/test_in_distribution_hashfrag.tsv",
        "test_ood": "data/k562/test_sets/test_ood_designed_k562.tsv",
        "test_snv_ref": "data/k562/test_sets/test_snv_pairs_hashfrag.tsv",
    }

    print("=== Cache vs File Alignment ===\n")

    for model_name, (cache_dir, embed_dim) in models.items():
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            print(f"{model_name}: cache dir not found")
            continue

        print(f"--- {model_name} ---")
        for prefix, tsv_path in test_files.items():
            npy_file = cache_path / f"{prefix}_canonical.npy"
            if not npy_file.exists():
                print(f"  {prefix}: NO CACHE FILE")
                continue

            arr = np.load(str(npy_file), mmap_mode="r")
            n_cache = arr.shape[0]

            tsv = Path(tsv_path)
            if tsv.exists():
                if prefix == "test_snv_ref":
                    n_file = len(pd.read_csv(tsv, sep="\t"))
                else:
                    n_file = len(pd.read_csv(tsv, sep="\t"))
                status = "OK" if n_cache == n_file else f"MISMATCH ({n_cache} vs {n_file})"
            else:
                n_file = "?"
                status = "TSV NOT FOUND"

            print(f"  {prefix}: cache={n_cache}, file={n_file} — {status}")

        # Also check HepG2/SKNSH OOD caches
        for cell in ["hepg2", "sknsh"]:
            cell_cache = Path(f"outputs/{model_name}_{cell}_cached/embedding_cache")
            ood_npy = cell_cache / "test_ood_canonical.npy"
            ood_tsv = Path(f"data/{cell}/test_sets/test_ood_designed_{cell}.tsv")
            if ood_npy.exists() and ood_tsv.exists():
                n_c = np.load(str(ood_npy), mmap_mode="r").shape[0]
                n_f = len(pd.read_csv(ood_tsv, sep="\t"))
                status = "OK" if n_c == n_f else f"MISMATCH ({n_c} vs {n_f})"
                print(f"  {cell}_ood: cache={n_c}, file={n_f} — {status}")

        print()


if __name__ == "__main__":
    check_alignment()
