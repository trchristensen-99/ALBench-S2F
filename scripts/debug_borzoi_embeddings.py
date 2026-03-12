#!/usr/bin/env python
"""Quick diagnostic: check if Borzoi produces distinct embeddings for different
sequences with center-bin pooling under bfloat16 vs fp32.

Run on GPU node:
    uv run --no-sync python scripts/debug_borzoi_embeddings.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from data.utils import one_hot_encode

# MPRA flanks
_FLANK_5 = MPRA_UPSTREAM[-200:]
_FLANK_3 = MPRA_DOWNSTREAM[:200]
_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

_F5 = np.zeros((4, 200), dtype=np.float32)
for i, c in enumerate(_FLANK_5):
    if c in _MAP:
        _F5[_MAP[c], i] = 1.0
_F3 = np.zeros((4, 200), dtype=np.float32)
for i, c in enumerate(_FLANK_3):
    if c in _MAP:
        _F3[_MAP[c], i] = 1.0

SEQ_LEN = 196_608


def load_borzoi(device):
    from experiments.train_foundation_stage2 import _fix_borzoi_attention, _load_borzoi

    model, embed_dim = _load_borzoi()
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def encode_sequences(model, sequences, device, center_bins=0, use_amp=True):
    """Encode sequences, return (N, 1536) embeddings."""
    ohs = []
    for seq in sequences:
        seq = seq.upper()
        if len(seq) < 200:
            pad = 200 - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        oh = one_hot_encode(seq, add_singleton_channel=False)
        oh_600 = np.concatenate([_F5, oh, _F3], axis=1)
        ohs.append(oh_600)

    batch = torch.from_numpy(np.stack(ohs)).float()
    pad_total = SEQ_LEN - 600
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = F.pad(batch, (pad_left, pad_right), value=0.0).to(device)

    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = model.get_embs_after_crop(padded)
        else:
            emb = model.get_embs_after_crop(padded)

    # Center-bin crop
    if center_bins > 0:
        total = emb.shape[2]
        c = total // 2
        h = center_bins // 2
        emb = emb[:, :, c - h : c + h]

    # Mean pool
    return emb.mean(dim=2).cpu().float().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_borzoi(device)

    ds = K562Dataset(data_path="data/k562", split="val")
    rng = np.random.RandomState(42)
    indices = rng.choice(len(ds), 20, replace=False)
    sequences = [ds.sequences[i] for i in indices]
    labels = [float(ds.labels[i]) for i in indices]

    print(f"\n{'=' * 60}")
    print(f"Testing {len(sequences)} val sequences")
    print(f"Labels: min={min(labels):.2f} max={max(labels):.2f} std={np.std(labels):.3f}")
    print(f"{'=' * 60}")

    for center_bins in [0, 32]:
        for use_amp in [True, False]:
            amp_label = "bfloat16" if use_amp else "fp32"
            pool_label = f"center_{center_bins}" if center_bins > 0 else "all_bins"
            print(f"\n--- {pool_label} / {amp_label} ---")

            embs = encode_sequences(model, sequences, device, center_bins, use_amp)

            # Per-sequence stats
            emb_norms = np.linalg.norm(embs, axis=1)
            print(f"  Embedding shape: {embs.shape}")
            print(f"  Per-seq L2 norm: mean={emb_norms.mean():.4f} std={emb_norms.std():.6f}")
            print(f"  Embedding values: mean={embs.mean():.6f} std={embs.std():.6f}")
            print(f"  Embedding range: [{embs.min():.6f}, {embs.max():.6f}]")

            # Inter-sequence variation
            pairwise_dists = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    pairwise_dists.append(np.linalg.norm(embs[i] - embs[j]))
            pairwise_dists = np.array(pairwise_dists)
            print(
                f"  Pairwise L2 dists: mean={pairwise_dists.mean():.6f} "
                f"std={pairwise_dists.std():.6f} "
                f"min={pairwise_dists.min():.6f} max={pairwise_dists.max():.6f}"
            )

            # Cosine similarity
            norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
            cos_sims = norms @ norms.T
            # Extract upper triangle
            upper = cos_sims[np.triu_indices(len(embs), k=1)]
            print(
                f"  Pairwise cosine sim: mean={upper.mean():.6f} "
                f"std={upper.std():.6f} min={upper.min():.6f}"
            )

            # Simple head prediction test (random head)
            head_w = np.random.randn(1, 1536).astype(np.float32) * 0.01
            preds = (embs @ head_w.T).flatten()
            pearson = np.corrcoef(preds, labels)[0, 1]
            print(f"  Random head Pearson r: {pearson:.4f}")
            print(f"  Predictions: mean={preds.mean():.6f} std={preds.std():.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
