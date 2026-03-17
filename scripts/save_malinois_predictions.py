#!/usr/bin/env python
"""Save Malinois test predictions from a trained checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]

# MPRA flanking sequences
_MPRA_UPSTREAM = (
    "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTAC"
    "GTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTAT"
    "TTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAA"
    "ACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCC"
    "TAACTGGCCGCTTGACG"
)
_MPRA_DOWNSTREAM = (
    "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCC"
    "TCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTG"
    "TTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCT"
    "GGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAG"
    "CTGACCCTGAAGTTCATCT"
)
_FLANK_5 = _MPRA_UPSTREAM[-200:]
_FLANK_3 = _MPRA_DOWNSTREAM[:200]
_NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_600bp(seq: str) -> np.ndarray:
    """Encode a 200bp sequence with flanks to (4, 600) one-hot."""
    # Standardize to 200bp
    if len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > 200:
        start = (len(seq) - 200) // 2
        seq = seq[start : start + 200]
    full = _FLANK_5 + seq + _FLANK_3
    oh = np.zeros((4, 600), dtype=np.float32)
    for i, c in enumerate(full):
        if c in _NUC_MAP:
            oh[_NUC_MAP[c], i] = 1.0
    return oh


class _SeqDataset(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(_encode_600bp(self.sequences[idx]))


def predict(model, sequences: list[str], device: torch.device) -> np.ndarray:
    ds = _SeqDataset(sequences)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            out = model(xb).squeeze(-1)
            # RC average
            xb_rc = xb.flip(1).flip(2)
            out_rc = model(xb_rc).squeeze(-1)
            preds.append(((out + out_rc) / 2).cpu().numpy())
    return np.concatenate(preds)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = REPO / "outputs" / "malinois_k562_with_preds" / "seed_42" / "seed_42"
    test_dir = REPO / "data" / "k562" / "test_sets"

    from models.basset_branched import BassetBranched

    model = BassetBranched(n_outputs=1)
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Malinois loaded from {ckpt_dir}")

    in_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_pred = predict(model, in_df["sequence"].tolist(), device)
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
    print(f"in_dist: R={pearsonr(in_pred, in_true)[0]:.4f}")

    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ref_pred = predict(model, snv_df["sequence_ref"].tolist(), device)
    alt_pred = predict(model, snv_df["sequence_alt"].tolist(), device)
    alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    delta_pred = alt_pred - ref_pred
    delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)

    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
    ood_pred = predict(model, ood_df["sequence"].tolist(), device)
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
