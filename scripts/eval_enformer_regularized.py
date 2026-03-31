#!/usr/bin/env python3
"""Evaluate Enformer S1 regularized heads on K562 test sets.

Loads trained head checkpoints from outputs/enformer_k562_regularized/
and computes test metrics + saves predictions using cached embeddings.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, ".")
from experiments.train_foundation_cached import MLPHead


def _safe_corr(a, b, fn):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    return float(fn(a[mask], b[mask])[0])


def evaluate_head(head, cache_dir, test_set_dir, cell_line="k562"):
    """Evaluate a trained head on all test sets."""
    cache = Path(cache_dir)
    metrics = {}
    arrays = {}

    CELL_COLS = {"k562": "K562_log2FC", "hepg2": "HepG2_log2FC", "sknsh": "SKNSH_log2FC"}
    fc_col = CELL_COLS.get(cell_line, "K562_log2FC")

    def predict(prefix):
        can = cache / f"{prefix}_canonical.npy"
        rc = cache / f"{prefix}_rc.npy"
        if not can.exists():
            return None
        emb_c = torch.tensor(np.load(str(can), mmap_mode="r"), dtype=torch.float32)
        emb_r = (
            torch.tensor(np.load(str(rc), mmap_mode="r"), dtype=torch.float32)
            if rc.exists()
            else emb_c
        )
        emb = (emb_c + emb_r) / 2
        with torch.no_grad():
            return head(emb).numpy().reshape(-1)

    # In-dist
    import pandas as pd

    test_dir = Path(test_set_dir)
    in_path = test_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        df = pd.read_csv(in_path, sep="\t")
        preds = predict("test_in_dist")
        if preds is not None and fc_col in df.columns:
            true = df[fc_col].to_numpy(dtype=np.float32)
            metrics["in_distribution"] = {
                "pearson_r": _safe_corr(preds, true, pearsonr),
                "spearman_r": _safe_corr(preds, true, spearmanr),
                "mse": float(np.mean((preds - true) ** 2)),
                "n": len(true),
            }
            arrays["in_dist_pred"] = preds

    # OOD
    ood_path = test_dir / f"test_ood_designed_{cell_line}.tsv"
    if ood_path.exists():
        df = pd.read_csv(ood_path, sep="\t")
        preds = predict("test_ood")
        if preds is not None:
            label_col = fc_col if fc_col in df.columns else "K562_log2FC"
            true = df[label_col].to_numpy(dtype=np.float32)
            # Handle cache/file size mismatch (OOD file may have been regenerated)
            n_min = min(len(preds), len(true))
            if len(preds) != len(true):
                print(
                    f"  [WARN] OOD size mismatch: cache={len(preds)} vs file={len(true)}, using first {n_min}"
                )
                preds = preds[:n_min]
                true = true[:n_min]
            metrics["ood"] = {
                "pearson_r": _safe_corr(preds, true, pearsonr),
                "spearman_r": _safe_corr(preds, true, spearmanr),
                "mse": float(np.mean((preds - true) ** 2)),
                "n": len(true),
            }
            arrays["ood_pred"] = preds

    # SNV
    snv_path = test_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        df = pd.read_csv(snv_path, sep="\t")
        ref_pred = predict("test_snv_ref")
        alt_pred = predict("test_snv_alt")
        if ref_pred is not None and alt_pred is not None:
            alt_col = f"{fc_col}_alt" if f"{fc_col}_alt" in df.columns else "K562_log2FC_alt"
            alt_true = df[alt_col].to_numpy(dtype=np.float32)
            metrics["snv_abs"] = {
                "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
                "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
                "n": len(alt_true),
            }
            delta_pred = alt_pred - ref_pred
            delta_col = f"delta_{fc_col}" if f"delta_{fc_col}" in df.columns else "delta_log2FC"
            delta_true = df[delta_col].to_numpy(dtype=np.float32)
            metrics["snv_delta"] = {
                "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
                "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
                "n": len(delta_true),
            }

    return metrics, arrays


def main():
    cache_dir = "outputs/enformer_k562_cached/embedding_cache"
    test_set_dir = "data/k562/test_sets"
    base = Path("outputs/enformer_k562_regularized")

    for config_dir in sorted(base.iterdir()):
        if not config_dir.is_dir():
            continue
        print(f"\n=== {config_dir.name} ===")
        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            # Find best_model.pt (may be in seed_dir or seed_dir/seed_*)
            ckpt_path = None
            for candidate in [seed_dir / "best_model.pt"] + list(seed_dir.glob("*/best_model.pt")):
                if candidate.exists():
                    ckpt_path = candidate
                    break
            if ckpt_path is None:
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            head = MLPHead(3072, 512, 0.1)
            head.load_state_dict(ckpt["model_state_dict"])
            head.eval()

            metrics, arrays = evaluate_head(head, cache_dir, test_set_dir)

            # Save
            result_path = ckpt_path.parent / "result.json"
            result_path.write_text(json.dumps({"test_metrics": metrics}, indent=2))

            if arrays:
                np.savez_compressed(ckpt_path.parent / "test_predictions.npz", **arrays)

            id_r = metrics.get("in_distribution", {}).get("pearson_r", "?")
            ood_r = metrics.get("ood", {}).get("pearson_r", "?")

            def fmt(x):
                return f"{x:.4f}" if isinstance(x, float) else str(x)

            print(f"  {seed_dir.name}: id={fmt(id_r)} ood={fmt(ood_r)} -> {result_path}")


if __name__ == "__main__":
    main()
