#!/usr/bin/env python
"""Save test predictions for all S1 models that have checkpoints.

Loads saved best_model.pt checkpoints and runs inference on test sets,
saving pred vs true arrays for scatter plots. Also recomputes MSE.

Models with cached embeddings (Enformer, Borzoi, NTv3): ~30s each.
Models requiring full inference (DREAM-RNN, Malinois): ~5-15 min each.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
DATA_PATH = REPO / "data" / "k562"
TEST_DIR = DATA_PATH / "test_sets"


def _corr(pred, true, fn):
    mask = np.isfinite(pred) & np.isfinite(true)
    return float(fn(pred[mask], true[mask])[0]) if mask.sum() >= 3 else 0.0


def save_predictions_cached(
    name: str,
    cache_dir: Path,
    ckpt_dir: Path,
    embed_dim: int,
    device: torch.device,
) -> None:
    """Save predictions for a model with cached embeddings."""
    from experiments.train_foundation_cached import MLPHead

    pred_path = ckpt_dir / "test_predictions.npz"
    if pred_path.exists():
        print(f"{name}: predictions already saved at {pred_path}")
        return

    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"{name}: no checkpoint at {ckpt_path}")
        return

    print(f"\n=== {name} (cached embeddings) ===")
    model = MLPHead(embed_dim, hidden_dim=512, dropout=0.1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    def predict_cached(prefix: str) -> np.ndarray:
        can = np.load(cache_dir / f"{prefix}_canonical.npy", mmap_mode="r")
        rc = np.load(cache_dir / f"{prefix}_rc.npy", mmap_mode="r")
        preds = []
        with torch.no_grad():
            for i in range(0, len(can), 512):
                end = min(i + 512, len(can))
                c = torch.from_numpy(can[i:end].astype(np.float32)).to(device)
                r = torch.from_numpy(rc[i:end].astype(np.float32)).to(device)
                preds.append(((model(c) + model(r)) / 2).cpu().numpy())
        return np.concatenate(preds).squeeze()

    _save_test_predictions(name, predict_cached, ckpt_dir)


def _save_test_predictions(
    name: str,
    predict_fn,
    out_dir: Path,
) -> None:
    """Run predictions on all test sets and save."""
    # In-dist
    in_df = pd.read_csv(TEST_DIR / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_pred = predict_fn("test_in_dist")
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)

    # SNV
    snv_df = pd.read_csv(TEST_DIR / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ref_pred = predict_fn("test_snv_ref")
    alt_pred = predict_fn("test_snv_alt")
    alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    delta_pred = alt_pred - ref_pred
    delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)

    # OOD
    ood_df = pd.read_csv(TEST_DIR / "test_ood_designed_k562.tsv", sep="\t")
    ood_pred = predict_fn("test_ood")
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    # Save predictions
    pred_path = out_dir / "test_predictions.npz"
    np.savez_compressed(
        pred_path,
        in_dist_pred=in_pred,
        in_dist_true=in_true,
        snv_alt_pred=alt_pred,
        snv_alt_true=alt_true,
        snv_ref_pred=ref_pred,
        snv_delta_pred=delta_pred,
        snv_delta_true=delta_true,
        ood_pred=ood_pred,
        ood_true=ood_true,
    )

    # Also save/update result.json with complete metrics including MSE
    metrics = {
        "in_distribution": {
            "pearson_r": _corr(in_pred, in_true, pearsonr),
            "spearman_r": _corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
            "n": len(in_true),
        },
        "snv_abs": {
            "pearson_r": _corr(alt_pred, alt_true, pearsonr),
            "spearman_r": _corr(alt_pred, alt_true, spearmanr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            "n": len(alt_true),
        },
        "snv_delta": {
            "pearson_r": _corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            "n": len(delta_true),
        },
        "ood": {
            "pearson_r": _corr(ood_pred, ood_true, pearsonr),
            "spearman_r": _corr(ood_pred, ood_true, spearmanr),
            "mse": float(np.mean((ood_pred - ood_true) ** 2)),
            "n": len(ood_true),
        },
    }

    # Update existing result.json if it exists, otherwise create new
    result_path = out_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        result["test_metrics"] = metrics
    else:
        result = {"model": name, "test_metrics": metrics}
    result_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"  Predictions: {pred_path} ({pred_path.stat().st_size / 1024:.0f} KB)")
    for k, v in metrics.items():
        print(f"  {k}: pearson={v['pearson_r']:.4f}, mse={v['mse']:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Foundation models with cached embeddings
    cached_models = [
        (
            "enformer",
            REPO / "outputs" / "enformer_k562_cached" / "embedding_cache",
            REPO / "outputs" / "enformer_k562_3seeds" / "seed_598125057",
            3072,
        ),
        (
            "ntv3_post",
            REPO / "outputs" / "ntv3_post_k562_cached" / "embedding_cache",
            REPO
            / "outputs"
            / "foundation_grid_search"
            / "ntv3_post"
            / "lr0.0005_wd1e-6_do0.1"
            / "seed_42"
            / "seed_42",
            1536,
        ),
    ]

    # Only add Borzoi if cache exists
    borzoi_cache = (
        REPO / "outputs" / "borzoi_k562_cached" / "embedding_cache" / "train_canonical.npy"
    )
    if borzoi_cache.exists():
        cached_models.append(
            (
                "borzoi",
                REPO / "outputs" / "borzoi_k562_cached" / "embedding_cache",
                REPO / "outputs" / "borzoi_k562_3seeds" / "seed_824292012",
                1536,
            )
        )
    else:
        print("Borzoi: cache not ready, skipping")

    for name, cache_dir, ckpt_dir, embed_dim in cached_models:
        if not (cache_dir / "test_in_dist_canonical.npy").exists():
            print(f"{name}: test cache missing, skipping")
            continue
        save_predictions_cached(name, cache_dir, ckpt_dir, embed_dim, device)

    print("\n=== Done ===")
    # List all saved predictions
    for p in sorted(REPO.rglob("test_predictions.npz")):
        print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
