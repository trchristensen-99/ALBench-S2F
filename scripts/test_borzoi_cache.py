#!/usr/bin/env python
"""Test Borzoi embedding extraction with different precision modes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

FLANK_5 = MPRA_UPSTREAM[-200:]
FLANK_3 = MPRA_DOWNSTREAM[:200]
MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def to_600bp_oh(seq: str) -> np.ndarray:
    if len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > 200:
        start = (len(seq) - 200) // 2
        seq = seq[start : start + 200]
    full = FLANK_5 + seq + FLANK_3
    oh = np.zeros((4, 600), dtype=np.float32)
    for i, c in enumerate(full):
        if c in MAPPING:
            oh[MAPPING[c], i] = 1.0
    return oh


def encode_batch(model, seqs, device, use_autocast=False, autocast_dtype=torch.float16):
    oh = np.stack([to_600bp_oh(s) for s in seqs])
    oh_rc = oh[:, ::-1, ::-1].copy()
    can = torch.from_numpy(oh).float().to(device)
    rc = torch.from_numpy(oh_rc).float().to(device)
    pad_total = 196608 - 600
    pad_l = pad_total // 2
    can_p = torch.nn.functional.pad(can, (pad_l, pad_total - pad_l))
    rc_p = torch.nn.functional.pad(rc, (pad_l, pad_total - pad_l))

    if use_autocast:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            emb_c = model.get_embs_after_crop(can_p).mean(dim=2)
            emb_r = model.get_embs_after_crop(rc_p).mean(dim=2)
    else:
        with torch.no_grad():
            emb_c = model.get_embs_after_crop(can_p).mean(dim=2)
            emb_r = model.get_embs_after_crop(rc_p).mean(dim=2)

    emb_c = emb_c.cpu().float().numpy()
    emb_r = emb_r.cpu().float().numpy()
    return (emb_c + emb_r) / 2.0


def main():
    import transformers

    print(f"transformers: {transformers.__version__}")
    print(f"torch: {torch.__version__}")

    from borzoi_pytorch import Borzoi

    # Fix for transformers>=5.3
    if not hasattr(Borzoi, "all_tied_weights_keys"):
        Borzoi.all_tied_weights_keys = {}

    device = torch.device("cuda")
    model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    model.eval().to(device)
    print("Borzoi loaded")

    ds = K562Dataset(data_path="data/k562", split="val")
    seqs = [ds.sequences[i] for i in range(200)]
    labels = ds.labels[:200].astype(np.float32)

    for mode_name, use_ac, ac_dtype in [
        ("float32 (no autocast)", False, None),
        ("float16 autocast", True, torch.float16),
        ("bfloat16 autocast", True, torch.bfloat16),
    ]:
        print(f"\n=== {mode_name} ===")
        embs = []
        nan_count = 0
        for i in range(0, len(seqs), 2):
            batch = seqs[i : i + 2]
            emb = encode_batch(model, batch, device, use_ac, ac_dtype)
            if np.isnan(emb).any():
                nan_count += 1
            embs.append(emb)
        embs = np.concatenate(embs)
        nan_total = np.isnan(embs).any(axis=1).sum()
        print(f"  NaN batches: {nan_count}/{len(seqs) // 2}, NaN rows: {nan_total}/{len(embs)}")

        if nan_total == 0:
            sims = [1 - cosine(embs[i], embs[j]) for i, j in [(0, 1), (0, 50), (0, 100)]]
            print(f"  Cosine sims: {[f'{s:.6f}' for s in sims]}")

            # Quick head training
            from experiments.train_foundation_cached import MLPHead

            head = MLPHead(1536, hidden_dim=512, dropout=0.1).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=0.001)
            et = torch.from_numpy(embs[:160].astype(np.float32)).to(device)
            lt = torch.from_numpy(labels[:160]).to(device)
            ev = torch.from_numpy(embs[160:].astype(np.float32)).to(device)
            lv = labels[160:]
            for ep in range(30):
                head.train()
                p = head(et).squeeze()
                loss = torch.nn.functional.mse_loss(p, lt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if ep % 10 == 0:
                    head.eval()
                    with torch.no_grad():
                        vp = head(ev).squeeze().cpu().numpy()
                    r = pearsonr(vp, lv)[0]
                    print(f"  Epoch {ep}: val_r={r:.4f}")
            head.eval()
            with torch.no_grad():
                vp = head(ev).squeeze().cpu().numpy()
            print(f"  Final: val_r={pearsonr(vp, lv)[0]:.4f}")
        else:
            print("  Skipping head training (NaN embeddings)")


if __name__ == "__main__":
    main()
