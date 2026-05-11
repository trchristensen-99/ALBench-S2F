"""Generate the Colab notebook for Peter (D=20k LegNet exploration).

Outputs:
    scripts/colab/k562_d20k_hpsearch.ipynb
    scripts/colab/bundle_d20k.tar.gz   (after make_data_bundle.py + train job)

Cells are written as a list of dicts and converted to .ipynb JSON.

Notebook structure:
  1. Header: what this is + dataset + best model perf
  2. Setup: pip installs + wget bundle
  3. Data loading: parquet → torch DataLoaders
  4. Self-contained LegNet model (inline, no project deps)
  5. Training loop (AdamW + OneCycle + early stop)
  6. Eval on val + test (Pearson R, Spearman R, MSE)
  7. "Things to try" — ideas for Peter
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts/colab/k562_d20k_hpsearch.ipynb"

# URL where the bundle will live (will be a GitHub release asset).
# Replace placeholder when the release is uploaded.
BUNDLE_URL = "https://github.com/trchristensen-99/ALBench-S2F/releases/download/colab-d20k/bundle_d20k.tar.gz"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md(
        f"""# K562 MPRA HP Search Sandbox — D=20,000 subset

**Sent to Peter Koo, May 11 2026, for architecture exploration.**

This notebook contains:
- A **20,000-sequence training subset** of the Tewhey K562 MPRA dataset
  (sampled deterministically from the chromosome-split train pool, AG-oracle
  pseudo-labels — same labels used in the broader HP search)
- The held-out **validation set** (chromosomes 19/21/X)
- The held-out **test set** (chromosomes 7/13)
- The **best LegNet model** trained on this subset (see `model.pt`)
- Self-contained code: just run the cells

The training pool spans D ∈ [600, 600,000] sequences in the larger study.
20k is a "middle" point — small enough that you can iterate quickly, large
enough that overfitting is meaningful.

**Goal:** swap in different architectures (AG modules, conv-only, different
dropout placement, etc.) and see what beats the current best.

Data URL: `{BUNDLE_URL}`
"""
    ),
    md(
        """## 1. Setup

Run these once at the top. The bundle is ~50 MB.
"""
    ),
    code(
        """!pip install -q torch numpy pandas pyarrow scipy tqdm
"""
    ),
    code(
        f"""# Download + extract the data + model bundle
!wget -q {BUNDLE_URL} -O bundle_d20k.tar.gz
!tar -xzf bundle_d20k.tar.gz
!ls -la k562_d20k/
"""
    ),
    md(
        """## 2. Data loading

Sequences are 200 bp (the MPRA insert payload, no flanking).
Labels are AG-oracle pseudolabels on the train split, real K562_log2FC on
val/test (so eval numbers are directly comparable to "real-data" training).

The one-hot encoder uses 4 channels: A, C, G, T. Anything else (N) maps to all-zeros.
"""
    ),
    code(
        """import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "k562_d20k"
PAYLOAD_LEN = 200

train_df = pd.read_parquet(f"{DATA_DIR}/train_d20k.parquet")
val_df   = pd.read_parquet(f"{DATA_DIR}/val.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")

print(f"train: {len(train_df):,} sequences")
print(f"val:   {len(val_df):,} sequences")
print(f"test:  {len(test_df):,} sequences")
print(f"\\nTrain label stats: mean={train_df['label'].mean():.3f}  std={train_df['label'].std():.3f}")
train_df.head()
"""
    ),
    code(
        """_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot(seqs, L=PAYLOAD_LEN):
    out = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        if len(s) < L:
            pad = L - len(s)
            s = "N" * (pad // 2) + s + "N" * (pad - pad // 2)
        elif len(s) > L:
            start = (len(s) - L) // 2
            s = s[start : start + L]
        for j, nuc in enumerate(s[:L]):
            idx = _NUC_TO_IDX.get(nuc)
            if idx is not None:
                out[i, idx, j] = 1.0
    return out


class SeqDataset(Dataset):
    def __init__(self, df):
        self.x = torch.from_numpy(one_hot(df["sequence"].tolist()))
        self.y = torch.from_numpy(df["label"].values.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_ds = SeqDataset(train_df)
val_ds   = SeqDataset(val_df)
test_ds  = SeqDataset(test_df)
print(f"Train tensor: {train_ds.x.shape}")
"""
    ),
    md(
        """## 3. Model — LegNet (self-contained)

This is the architecture currently leading our HP search. It's an inverted-residual
1-D CNN with squeeze-and-excitation, descended from the DREAM Challenge 2022 winner.

Things you might want to change:
- `block_sizes` — channel counts per stage (length = depth)
- `ks` — kernel size for the depthwise conv inside each block
- `dropout` — applied to the spatial output of each stage (we currently use one dropout knob; consider conv vs dense split)
- The block class itself — swap in AG modules / pure conv / etc.
"""
    ),
    code(
        '''from __future__ import annotations
import math
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """Squeeze-and-excitation channel attention."""
    def __init__(self, inp: int, oup: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), oup),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ResidualConcat(nn.Module):
    """Channel-concatenating residual."""
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x): return torch.cat([self.fn(x), x], dim=1)


class LocalBlock(nn.Module):
    """Conv → BN → SiLU."""
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, padding="same", bias=False),
            nn.BatchNorm1d(out_ch),
            activation(),
        )
    def forward(self, x): return self.block(x)


class EffBlock(nn.Module):
    """Inverted-residual block (1x1 expand → DW conv → SE → 1x1 project)."""
    def __init__(self, in_ch, ks, resize_factor, filter_per_group, activation,
                 out_ch=None, se_reduction=None, inner_dim_calculation="out"):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        se_reduction = resize_factor if se_reduction is None else se_reduction
        inner_dim = (out_ch if inner_dim_calculation == "out" else in_ch) * resize_factor
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner_dim, 1, padding="same", bias=False),
            nn.BatchNorm1d(inner_dim),
            activation(),
            nn.Conv1d(inner_dim, inner_dim, ks, groups=inner_dim // filter_per_group,
                      padding="same", bias=False),
            nn.BatchNorm1d(inner_dim),
            activation(),
            SELayer(in_ch, inner_dim, reduction=se_reduction),
            nn.Conv1d(inner_dim, in_ch, 1, padding="same", bias=False),
            nn.BatchNorm1d(in_ch),
            activation(),
        )
    def forward(self, x): return self.block(x)


class MappingBlock(nn.Module):
    """1x1 conv → activation, for channel rescale at the head."""
    def __init__(self, in_ch, out_ch, activation):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, padding="same"), activation())
    def forward(self, x): return self.block(x)


class LegNet(nn.Module):
    """LegNet (NoGINet)."""
    def __init__(
        self,
        in_channels: int = 4,
        block_sizes=None,
        ks: int = 5,
        resize_factor: int = 4,
        activation: Type[nn.Module] = nn.SiLU,
        final_activation: Type[nn.Module] = nn.SiLU,
        filter_per_group: int = 2,
        se_reduction: int = 4,
        inner_dim_calculation: str = "out",
        dropout: float = 0.0,
    ):
        super().__init__()
        if block_sizes is None:
            block_sizes = [256, 256, 128, 128, 64, 64, 32, 32]
        self.block_sizes = block_sizes
        self.stem_block = LocalBlock(in_channels, ks, activation, out_ch=block_sizes[0])

        blocks = []
        for prev_sz, sz in zip(block_sizes[:-1], block_sizes[1:]):
            layers = [
                ResidualConcat(EffBlock(prev_sz, ks, resize_factor, filter_per_group,
                                         activation, out_ch=sz,
                                         inner_dim_calculation=inner_dim_calculation)),
                LocalBlock(2 * prev_sz, ks, activation, out_ch=sz),
            ]
            if dropout > 0:
                layers.append(nn.Dropout1d(dropout))
            blocks.append(nn.Sequential(*layers))
        self.main = nn.Sequential(*blocks)
        self.mapper = MappingBlock(block_sizes[-1], 1, activation=final_activation)

    def forward(self, x):
        x = self.stem_block(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        return x.squeeze(-1)


# Sanity check
model = LegNet()
n_params = sum(p.numel() for p in model.parameters())
print(f"LegNet default: {n_params:,} params")
print(f"Block sizes: {model.block_sizes}")
x = torch.zeros(2, 4, PAYLOAD_LEN)
print(f"Forward pass output shape: {model(x).shape}")
'''
    ),
    md(
        """## 4. Training loop — AdamW + OneCycleLR

Drop-in trainer. You can swap the optimizer (Adam, AdamW, Muon) here; change the
LR schedule; add per-layer dropout; etc.
"""
    ),
    code(
        '''import math
import time
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr


def train_model(model, train_loader, val_loader, *,
                lr=3e-3, weight_decay=0.05,
                epochs=80, patience=15, use_rc_aug=True):
    """Train with AdamW + OneCycleLR + RC augmentation + early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps, pct_start=0.1)
    crit = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    def rc_flip(x):
        out = x.flip(dims=[2]).clone()
        out[:, [0, 1, 2, 3]] = out[:, [3, 2, 1, 0]]
        return out

    def eval_pearson(loader):
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                yhat = model(xb)
                ps.append(yhat.cpu().numpy()); ts.append(yb.cpu().numpy())
        p = np.concatenate(ps); t = np.concatenate(ts)
        return float(pearsonr(p, t)[0]), float(np.mean((p - t) ** 2))

    best_val_mse = math.inf
    best_state = None
    epochs_since_best = 0
    for epoch in range(epochs):
        model.train()
        ep_loss = 0; n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            if use_rc_aug and torch.rand(1).item() < 0.5:
                xb = rc_flip(xb)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                yhat = model(xb)
                loss = crit(yhat, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item(); n += 1
        train_loss = ep_loss / max(1, n)

        val_pearson, val_mse = eval_pearson(val_loader)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        print(f"  ep {epoch+1:3d}  train_mse={train_loss:.4f}  val_mse={val_mse:.4f}  "
              f"val_pearson={val_pearson:.4f}  lr={sched.get_last_lr()[0]:.5f}")

        if epochs_since_best >= patience:
            print(f"  early stop @ ep {epoch+1} (patience {patience})")
            break

    model.load_state_dict(best_state)
    return model, best_val_mse
'''
    ),
    md(
        """## 5. Train a fresh model

Default takes ~5–15 min on a Colab T4.
Iterate quickly: try `block_sizes=[128, 128, 64, 64]` for a smaller model, or change the optimizer / dropout / etc.
"""
    ),
    code(
        """train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

model = LegNet(in_channels=4, block_sizes=[256, 256, 128, 128, 64, 64, 32, 32], ks=5, dropout=0.1)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

model, best_val_mse = train_model(
    model, train_loader, val_loader,
    lr=3e-3, weight_decay=0.05, epochs=80, patience=15, use_rc_aug=True,
)
print(f"\\nBest val_mse: {best_val_mse:.4f}")
"""
    ),
    md(
        """## 6. Evaluate on test set
"""
    ),
    code(
        """import os

def evaluate(model, loader, name="set"):
    model.eval()
    device = next(model.parameters()).device
    ps, ts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            yhat = model(xb)
            ps.append(yhat.cpu().numpy()); ts.append(yb.cpu().numpy())
    p = np.concatenate(ps); t = np.concatenate(ts)
    pearson = pearsonr(p, t)[0]
    spearman = spearmanr(p, t).correlation
    mse = np.mean((p - t) ** 2)
    print(f"{name:8s}: Pearson R = {pearson:.4f}  Spearman R = {spearman:.4f}  MSE = {mse:.4f}")
    return pearson, spearman, mse

if os.path.exists("k562_d20k/best_model.pt"):
    print("=== Our pre-trained best model ===")
    ckpt = torch.load("k562_d20k/best_model.pt", map_location="cpu", weights_only=False)
    pretrained = LegNet(in_channels=4, block_sizes=ckpt["block_sizes"], ks=5,
                         dropout=ckpt.get("dropout", 0.1))
    pretrained.load_state_dict(ckpt["model_state_dict"])
    pretrained = pretrained.to("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(pretrained, val_loader,  "val ")
    evaluate(pretrained, test_loader, "test")
else:
    print("(best_model.pt not yet in bundle — re-pull the bundle once Trevor has uploaded it.)")

print("\\n=== Your fresh training ===")
evaluate(model, val_loader,  "val ")
evaluate(model, test_loader, "test")
"""
    ),
    md(
        """## 7. Things to try

Things Peter suggested over Slack:

**Optimizer:** try Adam, AdamW, [Muon](https://github.com/KellerJordan/Muon) — the last one
is sometimes surprisingly competitive for these small-data convolutional setups.

**Dropout placement:** the current LegNet `dropout` is applied to the conv output of
each stage. Conv layers typically need *less* dropout than dense layers. Try splitting:
`conv_dropout=0.05–0.15` for the EffBlock stages, `dense_dropout=0.3–0.5` if you add
an MLP head.

**Per-layer widths:** `block_sizes` is currently a small ladder. Try independent per-layer
widths (so each layer is its own HP) — e.g. `[256, 512, 256, 128, 64]`.

**AG modules:** Yash Moon (Koo lab) has code for AG-style inductive-bias blocks that
have been beating vanilla conv in our regulatory-genomics benchmarks. Worth swapping
into the LegNet body and seeing how it compares.

**Conv-only:** for DREAM-ATTN, try removing the attention blocks entirely. Sometimes
the conv stem + SE is enough.

**Data augmentations** (currently we only do RC at 50%):
- Shift augmentation (slide the payload window over flanking adapters) — adapters are
  `LEFT="AGGACCGGATCAACT"`, `RIGHT="CATTGCGTGAACCGA"` (15 bp each, MPRA constants)
- EvoAug structural mutations (deletion / insertion / inversion / translocation /
  tandem duplication / point mutation) — from [Lee & Koo 2023](https://www.biorxiv.org/content/10.1101/2023.06.16.545475v1).
  Implemented in `models/evoaug_transform.py` upstream.

**Bigger Ds:** if 20k is too small for the architecture you want to test, just resample
from the parquet file — the full train pool is ~600k sequences on the chromosome split.
"""
    ),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
