#!/usr/bin/env python
"""Save predictions.npz for foundation model S1 heads in chr_split.

Loads best_model.pt checkpoints from outputs/chr_split/{cell}/{model}_s1/
and the corresponding embedding caches from outputs/chr_split/{cell}/{model}_cached/embedding_cache/.

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
FOUNDATION_MODELS = {
    "enformer": {"embed_dim": 3072, "hidden_dim": 512},
    "borzoi": {"embed_dim": 1536, "hidden_dim": 512},
    "ntv3_post": {"embed_dim": 1536, "hidden_dim": 512},
}


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
    data_path = REPO / "data" / "k562"

    for model_name, cfg in FOUNDATION_MODELS.items():
        s1_dir = REPO / "outputs" / "chr_split" / args.cell / f"{model_name}_s1"
        cache_dir = (
            REPO / "outputs" / "chr_split" / args.cell / f"{model_name}_cached" / "embedding_cache"
        )

        if not s1_dir.exists():
            print(f"\nSKIP {model_name}: {s1_dir} does not exist")
            continue
        if not cache_dir.exists():
            print(f"\nSKIP {model_name}: cache dir {cache_dir} does not exist")
            continue
        # Verify cache has test embeddings
        if not (cache_dir / "test_in_dist_canonical.npy").exists():
            print(f"\nSKIP {model_name}: test embeddings not in cache")
            continue

        # Find all seed directories
        for seed_dir in sorted(s1_dir.glob("seed_*/seed_*")):
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
