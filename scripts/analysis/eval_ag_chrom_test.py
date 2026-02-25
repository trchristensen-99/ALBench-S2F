#!/usr/bin/env python
"""Evaluate AlphaGenome Boda heads (sum, mean, max, center) on chr 7, 13 test set.

Writes outputs/ag_chrom_test_results.json for direct comparison with Malinois.
Run from repo root (on HPC or locally with checkpoints and data).

  uv run python scripts/analysis/eval_ag_chrom_test.py
  uv run python scripts/analysis/eval_ag_chrom_test.py --data_path data/k562 --output outputs/ag_chrom_test_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval_ag import evaluate_chrom_test

# 4-tuple: (label, ckpt_dir, head_name, cache_dir_override)
# cache_dir_override=None  → fall back to --cache_dir CLI arg (standard 600bp cache).
# cache_dir_override=False → explicitly skip cache (full_aug models trained via live encoder).
# cache_dir_override=<str> → use that specific cache dir (e.g. compact heads).
_COMPACT_CACHE = "outputs/ag_compact/embedding_cache_compact"
_NO_CACHE = False  # sentinel: bypass test-embedding cache for full_aug models

CONFIGS = [
    # no_shift heads (precomputed 600bp embedding cache, canonical + RC)
    ("boda_sum", "outputs/ag_sum/best_model", "alphagenome_k562_head_boda_sum_512_512_v4", None),
    ("boda_mean", "outputs/ag_mean/best_model", "alphagenome_k562_head_boda_mean_512_512_v4", None),
    ("boda_max", "outputs/ag_max/best_model", "alphagenome_k562_head_boda_max_512_512_v4", None),
    (
        "boda_center",
        "outputs/ag_center/best_model",
        "alphagenome_k562_head_boda_center_512_512_v4",
        None,
    ),
    (
        "boda_flatten",
        "outputs/ag_flatten/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
        None,
    ),
    # hybrid heads (50% cache + 50% live encoder with ±15 bp shift augmentation)
    (
        "boda_sum_hybrid",
        "outputs/ag_sum_hybrid/best_model",
        "alphagenome_k562_head_boda_sum_512_512_v4",
        None,
    ),
    (
        "boda_mean_hybrid",
        "outputs/ag_mean_hybrid/best_model",
        "alphagenome_k562_head_boda_mean_512_512_v4",
        None,
    ),
    (
        "boda_max_hybrid",
        "outputs/ag_max_hybrid/best_model",
        "alphagenome_k562_head_boda_max_512_512_v4",
        None,
    ),
    (
        "boda_center_hybrid",
        "outputs/ag_center_hybrid/best_model",
        "alphagenome_k562_head_boda_center_512_512_v4",
        None,
    ),
    (
        "boda_flatten_hybrid",
        "outputs/ag_flatten_hybrid/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
        None,
    ),
    # v2 heads: dropout=0.1 + plateau LR (512-512) or constant LR (1024)
    (
        "boda_sum_v2",
        "outputs/ag_sum_v2/best_model",
        "alphagenome_k562_head_boda_sum_512_512_v4",
        None,
    ),
    (
        "boda_mean_v2",
        "outputs/ag_mean_v2/best_model",
        "alphagenome_k562_head_boda_mean_512_512_v4",
        None,
    ),
    (
        "boda_flatten_v2",
        "outputs/ag_flatten_v2/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
        None,
    ),
    (
        "boda_flatten_v2_do02",
        "outputs/ag_flatten_v2_do02/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
        None,
    ),
    (
        "boda_flatten_ref",
        "outputs/ag_flatten_ref/best_model",
        "alphagenome_k562_head_boda_flatten_1024_dropout_v4",
        None,
    ),
    (
        "boda_flatten_ref_do02",
        "outputs/ag_flatten_ref_do02/best_model",
        "alphagenome_k562_head_boda_flatten_1024_dropout_v4",
        None,
    ),
    (
        "boda_flatten_ref_do05",
        "outputs/ag_flatten_ref_do05/best_model",
        "alphagenome_k562_head_boda_flatten_1024_dropout_v4",
        None,
    ),
    (
        "boda_flatten_ref_lr01",
        "outputs/ag_flatten_ref_lr01/best_model",
        "alphagenome_k562_head_boda_flatten_1024_dropout_v4",
        None,
    ),
    # Novel architecture heads (no_shift, 600bp cache)
    (
        "boda_flatten_512_256",
        "outputs/ag_flatten_512_256/best_model",
        "alphagenome_k562_head_boda_flatten_512_256_v4",
        None,
    ),
    (
        "boda_flatten_1024_512",
        "outputs/ag_flatten_1024_512/best_model",
        "alphagenome_k562_head_boda_flatten_1024_512_v4",
        None,
    ),
    (
        "boda_sum_1024_ref",
        "outputs/ag_sum_1024_ref/best_model",
        "alphagenome_k562_head_boda_sum_1024_dropout_v4",
        None,
    ),
    # mean-1024 no_shift
    (
        "boda_mean_1024_ref",
        "outputs/ag_mean_1024_ref/best_model",
        "alphagenome_k562_head_boda_mean_1024_dropout_v4",
        None,
    ),
    # Full-aug variants: novel arch + augmentation combos
    # Use _NO_CACHE: trained via live encoder (model._predict()), incompatible with head-only cache path.
    (
        "boda_flatten_ref_full_aug",
        "outputs/ag_flatten_ref_full_aug/best_model",
        "alphagenome_k562_head_boda_flatten_1024_dropout_v4",
        _NO_CACHE,
    ),
    (
        "boda_pool_flatten_full_aug",
        "outputs/ag_pool_flatten_full_aug/best_model",
        "alphagenome_k562_head_pool_flatten_v4",
        _NO_CACHE,
    ),
    (
        "boda_mean_full_aug",
        "outputs/ag_mean_full_aug/best_model",
        "alphagenome_k562_head_boda_mean_512_512_v4",
        _NO_CACHE,
    ),
    (
        "boda_flatten_full_aug_shift25",
        "outputs/ag_flatten_full_aug_shift25/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
        _NO_CACHE,
    ),
    # MLP-512-512 and pool-flatten heads with dropout=0.1
    (
        "mlp_ref",
        "outputs/ag_mlp_ref/best_model",
        "alphagenome_k562_head_mlp_512_512_v4",
        None,
    ),
    (
        "mlp_plateau",
        "outputs/ag_mlp_plateau/best_model",
        "alphagenome_k562_head_mlp_512_512_v4",
        None,
    ),
    (
        "pool_flatten_ref",
        "outputs/ag_pool_flatten_ref/best_model",
        "alphagenome_k562_head_pool_flatten_v4",
        None,
    ),
    (
        "pool_flatten_plateau",
        "outputs/ag_pool_flatten_plateau/best_model",
        "alphagenome_k562_head_pool_flatten_v4",
        None,
    ),
    # Full shift augmentation heads (aug_mode=full, encoder on every batch)
    # Use _NO_CACHE: incompatible with head-only cache path.
    (
        "boda_flatten_full_aug",
        "outputs/ag_flatten_full_aug/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
        _NO_CACHE,
    ),
    (
        "boda_flatten_full_aug_plateau",
        "outputs/ag_flatten_full_aug_plateau/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
        _NO_CACHE,
    ),
    # Cosine LR variants: same full-aug training but with cosine annealing schedule
    (
        "boda_flatten_full_aug_cosine",
        "outputs/ag_flatten_full_aug_cosine/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
        _NO_CACHE,
    ),
    (
        "boda_pool_flatten_full_aug_cosine",
        "outputs/ag_pool_flatten_full_aug_cosine/best_model",
        "alphagenome_k562_head_pool_flatten_v4",
        _NO_CACHE,
    ),
    # Sum head variants with full augmentation
    (
        "boda_sum_full_aug",
        "outputs/ag_sum_full_aug/best_model",
        "alphagenome_k562_head_boda_sum_512_512_v4",
        _NO_CACHE,
    ),
    (
        "boda_sum_1024_full_aug",
        "outputs/ag_sum_1024_full_aug/best_model",
        "alphagenome_k562_head_boda_sum_1024_dropout_v4",
        _NO_CACHE,
    ),
    # 384bp compact-window heads (T=3 tokens; separate embedding cache)
    (
        "boda_sum_compact",
        "outputs/ag_sum_compact/best_model",
        "alphagenome_k562_head_boda_sum_512_512_v4",
        _COMPACT_CACHE,
    ),
    (
        "boda_mean_compact",
        "outputs/ag_mean_compact/best_model",
        "alphagenome_k562_head_boda_mean_512_512_v4",
        _COMPACT_CACHE,
    ),
    (
        "boda_max_compact",
        "outputs/ag_max_compact/best_model",
        "alphagenome_k562_head_boda_max_512_512_v4",
        _COMPACT_CACHE,
    ),
    (
        "boda_center_compact",
        "outputs/ag_center_compact/best_model",
        "alphagenome_k562_head_boda_center_512_512_v4",
        _COMPACT_CACHE,
    ),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/k562", help="K562 data dir (for K562FullDataset).")
    p.add_argument(
        "--output", default="outputs/ag_chrom_test_results.json", help="Output JSON path."
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Pre-built test embedding cache dir for 600bp heads (contains test_canonical.npy / "
            "test_rc.npy). If provided and files exist, encoder is skipped for ~10× faster "
            "per-head eval. Build with: uv run python scripts/analysis/build_test_embedding_cache.py"
        ),
    )
    args = p.parse_args()

    # Load existing results so we can skip already-evaluated models.
    out_path = Path(args.output)
    out: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            out = json.load(f)
        print(
            f"[eval_ag_chrom_test] Loaded {len(out)} existing results from {out_path}",
            file=sys.stderr,
        )

    for label, ckpt_dir, head_name, per_cfg_cache in CONFIGS:
        if label in out:
            print(f"[eval_ag_chrom_test] Skip {label}: already in results", file=sys.stderr)
            continue
        ckpt_path = Path(ckpt_dir).resolve()
        if not (ckpt_path / "checkpoint").exists():
            print(
                f"[eval_ag_chrom_test] Skip {label}: no checkpoint at {ckpt_path}", file=sys.stderr
            )
            continue
        # Per-config cache_dir takes priority; fall back to --cache_dir CLI arg.
        # per_cfg_cache=False (_NO_CACHE) means skip cache even if --cache_dir is set.
        if per_cfg_cache is False:
            cache = None
        elif per_cfg_cache is not None:
            cache = per_cfg_cache
        else:
            cache = args.cache_dir
        print(f"[eval_ag_chrom_test] Evaluating {label} on chr 7, 13 ...", file=sys.stderr)
        out[label] = evaluate_chrom_test(
            str(ckpt_path), head_name, data_path=args.data_path, cache_dir=cache
        )

    if not out:
        print("[eval_ag_chrom_test] No checkpoints found.", file=sys.stderr)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval_ag_chrom_test] Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
