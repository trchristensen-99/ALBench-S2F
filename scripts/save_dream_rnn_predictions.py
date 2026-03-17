#!/usr/bin/env python
"""Save DREAM-RNN test predictions from a trained checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.utils import one_hot_encode
from models.dream_rnn import create_dream_rnn

REPO = Path(__file__).resolve().parents[1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = (
        REPO / "outputs" / "dream_rnn_k562_with_preds" / "seed_42" / "seed_42" / "fraction_1.0000"
    )
    test_dir = REPO / "data" / "k562" / "test_sets"

    model = create_dream_rnn(input_channels=5, sequence_length=200, task_mode="k562")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"DREAM-RNN loaded from {ckpt_dir}")

    def predict(sequences: list[str]) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(sequences), 256):
                batch = sequences[i : i + 256]
                oh = np.stack([one_hot_encode(s, add_singleton_channel=False) for s in batch])
                x = torch.from_numpy(oh).float().to(device)
                p = model.predict(x, use_reverse_complement=True)
                preds.append(p.cpu().numpy().squeeze())
        return np.concatenate(preds)

    in_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_pred = predict(in_df["sequence"].tolist())
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
    print(f"in_dist: R={pearsonr(in_pred, in_true)[0]:.4f}")

    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ref_pred = predict(snv_df["sequence_ref"].tolist())
    alt_pred = predict(snv_df["sequence_alt"].tolist())
    alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    delta_pred = alt_pred - ref_pred
    delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)

    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
    ood_pred = predict(ood_df["sequence"].tolist())
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    pred_path = ckpt_dir / "test_predictions.npz"
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
    print(f"Saved: {pred_path}")


if __name__ == "__main__":
    main()
