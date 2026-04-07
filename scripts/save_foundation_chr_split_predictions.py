#!/usr/bin/env python
"""Save predictions.npz for foundation model S1 heads in chr_split.

Searches for best_model.pt checkpoints and embedding caches across multiple
possible directory layouts:

  S1 checkpoints (searched in order, first match wins):
    outputs/chr_split/{cell}/{model}_s1_v2/seed_*/seed_*/best_model.pt
    outputs/chr_split/{cell}/{model}_s1/seed_*/seed_*/best_model.pt
    outputs/{model}_{cell}_cached/seed_*/seed_*/best_model.pt

  Embedding caches (searched in order, first with test_in_dist_canonical.npy wins):
    outputs/chr_split/{cell}/{cache_name}_cached_v3/embedding_cache/  (preferred, correct alignment)
    outputs/chr_split/{cell}/{cache_name}_cached_v2/embedding_cache/
    outputs/chr_split/{cell}/{cache_name}_cached/embedding_cache/
    outputs/{model}_{cell}_cached/embedding_cache/

Produces predictions.npz with: in_dist_pred, in_dist_true, snv_ref_pred,
snv_alt_pred, snv_alt_true, snv_delta_pred, snv_delta_true, ood_pred, ood_true.

Usage::

    python scripts/save_foundation_chr_split_predictions.py --cell k562
    python scripts/save_foundation_chr_split_predictions.py --cell hepg2
    python scripts/save_foundation_chr_split_predictions.py --cell sknsh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.train_foundation_cached import (  # noqa: E402
    MLPHead,
    evaluate_test_sets_cached,
)

# Foundation model configurations
# cache_names: alternative directory name prefixes to search for embedding caches
# (e.g. ntv3_post models may have caches under "ntv3" instead of "ntv3_post")
FOUNDATION_MODELS = {
    "enformer": {"embed_dim": 3072, "hidden_dim": 512, "cache_names": ["enformer"]},
    "borzoi": {"embed_dim": 1536, "hidden_dim": 512, "cache_names": ["borzoi"]},
    "ntv3_post": {
        "embed_dim": 1536,
        "hidden_dim": 512,
        "cache_names": ["ntv3_post", "ntv3"],
    },
}


def _find_cache_dir(cell: str, model_name: str, cache_names: list[str]) -> Path | None:
    """Search candidate cache directories, return first with test embeddings."""
    candidates = []
    chr_split_base = REPO / "outputs" / "chr_split" / cell
    for name in cache_names:
        # Versioned dirs under chr_split (higher version first)
        candidates.append(chr_split_base / f"{name}_cached_v3" / "embedding_cache")
        candidates.append(chr_split_base / f"{name}_cached_v2" / "embedding_cache")
        candidates.append(chr_split_base / f"{name}_cached" / "embedding_cache")
    # Flat layout: outputs/{model}_{cell}_cached/embedding_cache/
    candidates.append(REPO / "outputs" / f"{model_name}_{cell}_cached" / "embedding_cache")

    for d in candidates:
        if d.is_dir() and (d / "test_in_dist_canonical.npy").exists():
            return d

    # If none have test embeddings, return first that exists at all (caller can
    # decide whether to skip based on missing test files)
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _find_s1_dirs(cell: str, model_name: str) -> list[Path]:
    """Return all seed directories containing best_model.pt across candidate S1 dirs."""
    candidates = [
        REPO / "outputs" / "chr_split" / cell / f"{model_name}_s1_v2",
        REPO / "outputs" / "chr_split" / cell / f"{model_name}_s1",
        REPO / "outputs" / f"{model_name}_{cell}_cached",
    ]
    seed_dirs: list[Path] = []
    seen: set[str] = set()  # deduplicate by seed name
    for base in candidates:
        if not base.is_dir():
            continue
        for sd in sorted(base.glob("seed_*/seed_*")):
            seed_name = sd.name  # e.g. "seed_42"
            if seed_name not in seen and (sd / "best_model.pt").exists():
                seed_dirs.append(sd)
                seen.add(seed_name)
    return seed_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Save predictions for foundation S1 heads in chr_split"
    )
    parser.add_argument(
        "--cell",
        required=True,
        choices=["k562", "hepg2", "sknsh"],
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing predictions.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_path = REPO / "data" / args.cell

    for model_name, cfg in FOUNDATION_MODELS.items():
        cache_dir = _find_cache_dir(args.cell, model_name, cfg["cache_names"])
        seed_dirs = _find_s1_dirs(args.cell, model_name)

        if not seed_dirs:
            print(f"\nSKIP {model_name}: no S1 seed dirs with best_model.pt found")
            continue
        if cache_dir is None:
            print(f"\nSKIP {model_name}: no embedding cache directory found")
            continue
        if not (cache_dir / "test_in_dist_canonical.npy").exists():
            print(
                f"\nSKIP {model_name}: cache at {cache_dir} has no test embeddings "
                f"(missing test_in_dist_canonical.npy)"
            )
            continue

        print(f"\n  Found cache: {cache_dir}")
        print(
            f"  Found {len(seed_dirs)} seed dir(s): {[str(s.relative_to(REPO)) for s in seed_dirs]}"
        )

        for seed_dir in seed_dirs:
            ckpt_path = seed_dir / "best_model.pt"
            pred_path = seed_dir / "predictions.npz"

            if pred_path.exists() and not args.force:
                print(f"\n  SKIP (exists): {seed_dir.relative_to(REPO)}")
                continue
            if not ckpt_path.exists():
                print(f"\n  SKIP (no checkpoint): {seed_dir.relative_to(REPO)}")
                continue

            print(f"\n{'=' * 60}")
            print(f"  Model: {model_name} | Cell: {args.cell}")
            print(f"  Checkpoint: {ckpt_path}")
            print(f"  Cache: {cache_dir}")
            print(f"{'=' * 60}")

            try:
                # Load model
                model = MLPHead(
                    embed_dim=cfg["embed_dim"],
                    hidden_dim=cfg["hidden_dim"],
                ).to(device)
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                # Evaluate on all test sets
                metrics, preds = evaluate_test_sets_cached(
                    model=model,
                    cache_dir=cache_dir,
                    data_path=data_path,
                    device=device,
                    cell_line=args.cell,
                    chr_split=True,
                )

                if preds:
                    np.savez_compressed(str(pred_path), **preds)
                    print(f"  Saved: {pred_path} ({pred_path.stat().st_size / 1024:.0f} KB)")
                    print(f"  Keys: {sorted(preds.keys())}")
                    for k, v in metrics.items():
                        pr = v.get("pearson_r", 0)
                        print(f"  {k}: pearson={pr:.4f}")

                    # Update result.json with metrics
                    result_path = seed_dir / "result.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        result["test_metrics"] = metrics
                    else:
                        result = {
                            "model": model_name,
                            "cell": args.cell,
                            "test_metrics": metrics,
                        }
                    result_path.write_text(json.dumps(result, indent=2, default=str))

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
