#!/usr/bin/env python3
"""Save test predictions for foundation S1 models using cached embeddings.

Loads trained head checkpoints + cached embeddings to generate
test_predictions.npz files for scatter plots.

Usage:
    python scripts/save_foundation_predictions.py --model enformer --cell k562
    python scripts/save_foundation_predictions.py --model borzoi --cell k562
    python scripts/save_foundation_predictions.py --model ntv3_post --cell k562
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, ".")
from experiments.train_foundation_cached import MLPHead  # noqa: I001


EMBED_DIMS = {"enformer": 3072, "borzoi": 1536, "ntv3_post": 1536}

RESULT_DIRS = {
    ("enformer", "k562"): [
        "outputs/enformer_k562_3seeds_v2",
        "outputs/enformer_k562_3seeds",
    ],
    ("borzoi", "k562"): [
        "outputs/borzoi_k562_3seeds_v2",
        "outputs/borzoi_k562_3seeds",
    ],
    ("ntv3_post", "k562"): [
        "outputs/ntv3_post_k562_3seeds_v2",
        "outputs/ntv3_post_k562_3seeds",
    ],
    # HepG2
    ("enformer", "hepg2"): ["outputs/enformer_hepg2_cached"],
    ("borzoi", "hepg2"): ["outputs/borzoi_hepg2_cached"],
    ("ntv3_post", "hepg2"): ["outputs/ntv3_post_hepg2_cached"],
    # SKNSH
    ("enformer", "sknsh"): ["outputs/enformer_sknsh_cached"],
    ("borzoi", "sknsh"): ["outputs/borzoi_sknsh_cached"],
    ("ntv3_post", "sknsh"): ["outputs/ntv3_post_sknsh_cached"],
}

CACHE_DIRS = {
    ("enformer", "k562"): "outputs/enformer_k562_cached/embedding_cache",
    ("borzoi", "k562"): "outputs/borzoi_k562_cached/embedding_cache",
    ("ntv3_post", "k562"): "outputs/ntv3_post_k562_cached/embedding_cache",
    ("enformer", "hepg2"): "outputs/enformer_hepg2_cached/embedding_cache",
    ("borzoi", "hepg2"): "outputs/borzoi_hepg2_cached/embedding_cache",
    ("ntv3_post", "hepg2"): "outputs/ntv3_post_hepg2_cached/embedding_cache",
    ("enformer", "sknsh"): "outputs/enformer_sknsh_cached/embedding_cache",
    ("borzoi", "sknsh"): "outputs/borzoi_sknsh_cached/embedding_cache",
    ("ntv3_post", "sknsh"): "outputs/ntv3_post_sknsh_cached/embedding_cache",
}


def predict_from_cache(head, cache_dir, prefix):
    """Load cached embeddings and predict."""
    can_path = Path(cache_dir) / f"{prefix}_canonical.npy"
    rc_path = Path(cache_dir) / f"{prefix}_rc.npy"
    if not can_path.exists():
        return None

    emb_c = torch.tensor(np.load(str(can_path), mmap_mode="r"), dtype=torch.float32)
    if rc_path.exists():
        emb_r = torch.tensor(np.load(str(rc_path), mmap_mode="r"), dtype=torch.float32)
        emb = (emb_c + emb_r) / 2
    else:
        emb = emb_c

    with torch.no_grad():
        # Process in batches to avoid OOM
        preds = []
        for i in range(0, len(emb), 4096):
            preds.append(head(emb[i : i + 4096]).numpy().reshape(-1))
        return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["enformer", "borzoi", "ntv3_post"])
    parser.add_argument("--cell", default="k562")
    args = parser.parse_args()

    embed_dim = EMBED_DIMS[args.model]
    cache_dir = CACHE_DIRS.get((args.model, args.cell))
    result_dirs = RESULT_DIRS.get((args.model, args.cell), [])

    if not cache_dir or not Path(cache_dir).exists():
        # Fall back to K562 cache (same sequences, different labels)
        k562_cache = CACHE_DIRS.get((args.model, "k562"))
        if k562_cache and Path(k562_cache).exists():
            print(f"Cache dir not found: {cache_dir}")
            print(f"Falling back to K562 cache: {k562_cache}")
            cache_dir = k562_cache
        else:
            print(f"Cache dir not found: {cache_dir} (and no K562 fallback)")
            return

    print(f"Model: {args.model}, Cell: {args.cell}")
    print(f"Cache: {cache_dir}")
    print(f"Embed dim: {embed_dim}")

    for result_dir in result_dirs:
        rdir = Path(result_dir)
        if not rdir.exists():
            continue

        for seed_dir in sorted(rdir.iterdir()):
            if not seed_dir.is_dir():
                continue

            # Find best_model.pt
            ckpt_path = None
            for candidate in [
                seed_dir / "best_model.pt",
                *seed_dir.glob("*/best_model.pt"),
            ]:
                if candidate.exists():
                    ckpt_path = candidate
                    break

            if ckpt_path is None:
                continue

            pred_path = ckpt_path.parent / "test_predictions.npz"
            if pred_path.exists():
                print(f"  {seed_dir.name}: predictions already exist, skipping")
                continue

            print(f"  {seed_dir.name}: loading {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            head = MLPHead(embed_dim, 512, 0.1)
            head.load_state_dict(ckpt["model_state_dict"])
            head.eval()

            arrays = {}
            for prefix in [
                "test_in_dist",
                "test_ood",
                "test_snv_ref",
                "test_snv_alt",
            ]:
                preds = predict_from_cache(head, cache_dir, prefix)
                if preds is not None:
                    arrays[f"{prefix}_pred"] = preds
                    print(f"    {prefix}: {len(preds)} predictions")

            if arrays:
                np.savez_compressed(pred_path, **arrays)
                print(f"    Saved to {pred_path}")


if __name__ == "__main__":
    main()
