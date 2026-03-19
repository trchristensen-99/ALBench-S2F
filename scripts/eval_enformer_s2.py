#!/usr/bin/env python
"""Eval-only: load Enformer S2 checkpoint and save predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_foundation_stage2 import (
    _forward_enformer,
    _predict_test_sequences,
    _safe_corr,
)

REPO = Path(__file__).resolve().parents[1]


def main():
    device = torch.device("cuda")
    out_dir = REPO / "outputs" / "enformer_k562_stage2_v2" / "run_0"
    test_dir = REPO / "data" / "k562" / "test_sets"

    # Load Enformer encoder
    from enformer_pytorch import Enformer

    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = {}
    encoder = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    print("Enformer encoder loaded")

    # Load S2 head
    from experiments.train_foundation_cached import MLPHead

    head = MLPHead(3072, hidden_dim=512, dropout=0.1)
    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["model_state_dict"])
    head.to(device).eval()
    print(f"Head loaded (epoch {ckpt.get('epoch', '?')})")

    def pred(sequences):
        return _predict_test_sequences(
            encoder, head, _forward_enformer, sequences, device, batch_size=4
        )

    # In-dist
    print("Predicting in_dist...")
    in_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_pred = pred(in_df["sequence"].tolist())
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
    print(f"  in_dist: R={pearsonr(in_pred, in_true)[0]:.4f}")

    # SNV
    print("Predicting SNV...")
    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ref_pred = pred(snv_df["sequence_ref"].tolist())
    alt_pred = pred(snv_df["sequence_alt"].tolist())
    alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    delta_pred = alt_pred - ref_pred
    delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)

    # OOD
    print("Predicting OOD...")
    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
    ood_pred = pred(ood_df["sequence"].tolist())
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    # Save predictions
    np.savez_compressed(
        out_dir / "test_predictions.npz",
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

    # Save result
    metrics = {
        "in_distribution": {
            "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
            "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
        },
        "snv_abs": {
            "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
        },
        "snv_delta": {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
        },
        "ood": {
            "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
            "mse": float(np.mean((ood_pred - ood_true) ** 2)),
        },
    }
    (out_dir / "result.json").write_text(
        json.dumps({"test_metrics": metrics, "epoch": ckpt.get("epoch", 0)}, indent=2)
    )

    print("Saved predictions + result.json")
    for k, v in metrics.items():
        print(f"  {k}: R={v['pearson_r']:.4f}, MSE={v['mse']:.4f}")


if __name__ == "__main__":
    main()
