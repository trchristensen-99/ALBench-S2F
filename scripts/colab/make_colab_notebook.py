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
        f"""# K562 MPRA HP Search Sandbox

Self-contained Colab for exploring sequence-to-function architectures on the
Tewhey K562 MPRA dataset.

**Contents (after running the wget cell):**
- `train_d20k.parquet` — 20,000-sequence train subset (AG-oracle OOF labels).
  This is what the notebook loads by default.
- `train_full.parquet` — full chromosome-split train pool (~617k sequences).
  Optional cell at the bottom subsamples this to any D you want.
- `val.parquet` — chr 19/21/X held-out (real K562_log2FC labels)
- `test.parquet` — chr 7/13 held-out (real K562_log2FC labels)
- `best_model.pt` — reference LegNet checkpoint, evaluated near the bottom of the notebook.

D=20k is a useful default — fast to iterate, large enough that architecture
choices matter. The full pool spans D ∈ [600, 617,000].

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
        f"""# Download + extract the data + model bundle (~70 MB)
!wget -q {BUNDLE_URL} -O bundle.tar.gz
!tar -xzf bundle.tar.gz
!ls -la k562_data/
"""
    ),
    md(
        """## 2. Data loading

Sequences are 200 bp (the MPRA insert payload). The MPRA reporter construct
flanks each insert with fixed 15-bp adapters:
```
LEFT_ADAPTER = "AGGACCGGATCAACT"   RIGHT_ADAPTER = "CATTGCGTGAACCGA"
```
These are exposed as constants so shift-augmentation (Section 5) has real
flanking context to slide into.

Labels: train uses AG-oracle pseudolabels (denoised, less noisy at small D);
val/test use real K562_log2FC. The one-hot encoder uses 4 channels (ACGT);
unknown bases map to all-zeros.

This cell loads the **default D=20k subset** (`train_d20k.parquet`). Skip to
Section 11 at the bottom if you want a different D from the full pool.
"""
    ),
    code(
        """import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "k562_data"
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
        """## 3. Building blocks

Three drop-in conv block classes, all with the same signature
`__init__(in_ch, ks)` and `forward: (B, in_ch, L) → (B, in_ch, L)`.
That makes them interchangeable — just change `BLOCK_CLASS` in the config.

| Class | What it is |
|---|---|
| `EffBlock` | Inverted-residual (1x1 expand → DW conv → SE → 1x1 project). Current LegNet default. |
| `PlainConvBlock` | Vanilla 2-layer conv → BN → SiLU. Simplest baseline. |
| `AGStyleBlock` | GroupNorm + depthwise-separable conv + GELU. Closer to AlphaGenome / Borzoi style — stronger inductive bias for regulatory genomics. |

To add a new block: write a class with the same signature and add it to `BLOCK_CLASSES`.
"""
    ),
    code(
        '''import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """Squeeze-and-excitation channel attention."""
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp // reduction)), nn.SiLU(),
            nn.Linear(int(inp // reduction), oup), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        return x * self.fc(y).view(b, c, 1)


class EffBlock(nn.Module):
    """Inverted-residual block (LegNet default): 1x1 expand → DW conv → SE → 1x1 project."""
    def __init__(self, in_ch, ks=5, resize_factor=4, filter_per_group=2, se_reduction=4):
        super().__init__()
        inner = in_ch * resize_factor
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner, 1, padding="same", bias=False),
            nn.BatchNorm1d(inner), nn.SiLU(),
            nn.Conv1d(inner, inner, ks, groups=inner // filter_per_group,
                      padding="same", bias=False),
            nn.BatchNorm1d(inner), nn.SiLU(),
            SELayer(in_ch, inner, reduction=se_reduction),
            nn.Conv1d(inner, in_ch, 1, padding="same", bias=False),
            nn.BatchNorm1d(in_ch), nn.SiLU(),
        )

    def forward(self, x): return self.block(x)


class PlainConvBlock(nn.Module):
    """Vanilla 2-layer conv block — no inverted residual, no SE."""
    def __init__(self, in_ch, ks=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, ks, padding="same", bias=False),
            nn.BatchNorm1d(in_ch), nn.SiLU(),
            nn.Conv1d(in_ch, in_ch, ks, padding="same", bias=False),
            nn.BatchNorm1d(in_ch), nn.SiLU(),
        )

    def forward(self, x): return self.block(x)


class AGStyleBlock(nn.Module):
    """AlphaGenome-style block: GroupNorm + depthwise-separable conv with GELU.

    Stronger genomic inductive bias than plain conv. Pre-norm transformer-flavored
    layout. This is a community approximation of the AG-style blocks used in
    AlphaGenome / Borzoi — drop in Yash Moon's reference implementation if you
    want the exact one.
    """
    def __init__(self, in_ch, ks=5, expansion=2):
        super().__init__()
        inner = in_ch * expansion
        self.block = nn.Sequential(
            nn.GroupNorm(1, in_ch),  # 1D layer-norm equivalent
            nn.Conv1d(in_ch, inner, 1, bias=False), nn.GELU(),
            nn.Conv1d(inner, inner, ks, groups=inner, padding="same", bias=False),
            nn.GELU(),
            nn.Conv1d(inner, in_ch, 1, bias=False),
        )

    def forward(self, x): return self.block(x)


BLOCK_CLASSES = {
    "eff":   EffBlock,
    "plain": PlainConvBlock,
    "ag":    AGStyleBlock,
}
'''
    ),
    md(
        """## 4. Model — modular LegNet

The model wires up the chosen block class into a multi-stage CNN. Each stage:
1. `ResidualConcat(block(in_ch))` — concat-style residual, doubles channel count
2. `LocalBlock(2*in_ch → out_ch, ks)` — 1-stage conv that picks the next channel size
3. Optional `Dropout1d(conv_dropout)` — conv-level dropout (low values, ~0.05–0.15)

After the conv stack, you can either pool-and-map (current default) or pool-and-MLP.
The MLP head uses `dense_dropout` separately — dense layers tolerate higher dropout
(~0.3–0.5) than conv layers.
"""
    ),
    code(
        '''class ResidualConcat(nn.Module):
    """fn(x) and x are channel-concatenated (channels double)."""
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x): return torch.cat([self.fn(x), x], dim=1)


class LocalBlock(nn.Module):
    """Conv → BN → SiLU. Used for stem + channel-resize steps."""
    def __init__(self, in_ch, out_ch, ks):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, padding="same", bias=False),
            nn.BatchNorm1d(out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.block(x)


class DenseHead(nn.Module):
    """Optional MLP head: pooled features → Linear stack → scalar.

    Dense layers tolerate higher dropout than conv. Set dense_dims=[] to skip and
    use the pooled-conv mapper instead (the LegNet default).
    """
    def __init__(self, in_dim, dense_dims, dense_dropout, out_dim=1):
        super().__init__()
        layers, prev = [], in_dim
        for d in dense_dims:
            layers += [nn.Linear(prev, d), nn.SiLU(), nn.Dropout(dense_dropout)]
            prev = d
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(-1)


class LegNet(nn.Module):
    """Modular LegNet — swap the conv block, change widths, toggle dense head."""

    def __init__(
        self,
        in_channels=4,
        block_sizes=(256, 256, 128, 128, 64, 64, 32, 32),
        ks=5,
        block_class=EffBlock,    # swap with PlainConvBlock / AGStyleBlock / your own
        conv_dropout=0.1,        # applied after each conv stage
        dense_dims=(),           # () = no MLP head (use pooled mapper); e.g. (256, 64) = MLP
        dense_dropout=0.3,       # dropout inside the optional MLP head
    ):
        super().__init__()
        self.block_sizes = list(block_sizes)

        self.stem = LocalBlock(in_channels, self.block_sizes[0], ks)

        stages = []
        for prev_sz, sz in zip(self.block_sizes[:-1], self.block_sizes[1:]):
            layers = [
                ResidualConcat(block_class(prev_sz, ks=ks)),
                LocalBlock(2 * prev_sz, sz, ks),
            ]
            if conv_dropout > 0:
                layers.append(nn.Dropout1d(conv_dropout))
            stages.append(nn.Sequential(*layers))
        self.body = nn.Sequential(*stages)

        if dense_dims:
            # Pool conv features → MLP head with dense_dropout
            self.head = DenseHead(self.block_sizes[-1], list(dense_dims), dense_dropout)
            self.mapper = None
        else:
            # Default: 1x1 conv mapper after pooling
            self.mapper = nn.Sequential(
                nn.Conv1d(self.block_sizes[-1], 1, 1, padding="same"), nn.SiLU(),
            )
            self.head = None

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        if self.head is not None:
            x = F.adaptive_avg_pool1d(x, 1).squeeze(2)   # (B, last_block)
            return self.head(x)
        x = self.mapper(x)
        return F.adaptive_avg_pool1d(x, 1).squeeze(2).squeeze(-1)


# Quick sanity check on all three block options
for name, cls in BLOCK_CLASSES.items():
    m = LegNet(block_class=cls, conv_dropout=0.1)
    n_params = sum(p.numel() for p in m.parameters())
    y = m(torch.zeros(2, 4, PAYLOAD_LEN))
    print(f"  block={name:6s}  params={n_params:>8,}  output={tuple(y.shape)}")
'''
    ),
    md(
        """## 5. Augmentations

Three sequence-domain augmentations. RC is on by default; shift requires
adapter-padded inputs (we expand the payload at batch time so the dataset
stays compact); EvoAug is provided as a pointer (separate library).
"""
    ),
    code(
        '''LEFT_ADAPTER  = "AGGACCGGATCAACT"   # 15 bp, 5' MPRA flank
RIGHT_ADAPTER = "CATTGCGTGAACCGA"   # 15 bp, 3' MPRA flank
MAX_SHIFT = len(LEFT_ADAPTER)       # = 15; each side gets up to 15 bp of context


def rc_flip(x):
    """Reverse-complement a (B, 4, L) one-hot batch."""
    out = x.flip(dims=[2]).clone()
    out[:, [0, 1, 2, 3]] = out[:, [3, 2, 1, 0]]
    return out


# Precompute adapter one-hots once (4 x 15) for fast batch-time concatenation
def _adapter_oh(s):
    a = torch.zeros(4, len(s))
    for j, c in enumerate(s.upper()):
        if c in _NUC_TO_IDX:
            a[_NUC_TO_IDX[c], j] = 1.0
    return a

_LEFT_OH  = _adapter_oh(LEFT_ADAPTER)
_RIGHT_OH = _adapter_oh(RIGHT_ADAPTER)


def pad_with_adapters(x):
    """(B, 4, 200) -> (B, 4, 200 + 2*MAX_SHIFT) by prepending/appending adapter one-hots."""
    B = x.shape[0]
    L = _LEFT_OH.unsqueeze(0).expand(B, -1, -1).to(x.device)
    R = _RIGHT_OH.unsqueeze(0).expand(B, -1, -1).to(x.device)
    return torch.cat([L, x, R], dim=2)


def shift_crop(x_padded, training=True, payload_len=PAYLOAD_LEN, max_shift=MAX_SHIFT):
    """Random offset sliding-window crop. At eval, returns the centered payload."""
    if not training:
        return x_padded[:, :, max_shift : max_shift + payload_len]
    B = x_padded.shape[0]
    offsets = torch.randint(0, 2 * max_shift + 1, (B,), device=x_padded.device)
    idx = offsets[:, None] + torch.arange(payload_len, device=x_padded.device)[None, :]
    idx = idx[:, None, :].expand(B, 4, payload_len)
    return x_padded.gather(2, idx)


# EvoAug (structural mutations: deletion, insertion, inversion, translocation, tandem dup,
# point mutation) — Lee & Koo 2023, https://www.biorxiv.org/content/10.1101/2023.06.16.545475v1
# Pip-installable as `evoaug-pytorch`. To use, transform `xb` per batch before model forward.
'''
    ),
    md(
        """## 6. Training loop — pluggable optimizer + AdamW default

`make_optimizer` supports `adam` / `adamw` / `muon` (last requires `pip install muon`).
`train_model` takes a model, loaders, and a config dict and does the rest.
"""
    ),
    code(
        '''import math
from scipy.stats import pearsonr, spearmanr


def make_optimizer(model, name, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "muon":
        try:
            from muon import Muon
        except ImportError:
            raise ImportError("pip install muon  (https://github.com/KellerJordan/Muon)")
        return Muon(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}  (use adam | adamw | muon)")


def train_model(model, train_loader, val_loader, cfg):
    """Train with optimizer + OneCycleLR + optional augmentations + early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = make_optimizer(model, cfg["optimizer"], cfg["lr"], cfg["weight_decay"])
    total_steps = cfg["epochs"] * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=total_steps, pct_start=0.1
    )
    crit = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

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
    for epoch in range(cfg["epochs"]):
        model.train()
        ep_loss = 0; n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            if cfg["use_shift_aug"]:
                xb = shift_crop(pad_with_adapters(xb), training=True)
            if cfg["use_rc_aug"] and torch.rand(1).item() < 0.5:
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

        if epochs_since_best >= cfg["patience"]:
            print(f"  early stop @ ep {epoch+1} (patience {cfg['patience']})")
            break

    model.load_state_dict(best_state)
    return model, best_val_mse
'''
    ),
    md(
        """## 7. Configure + train

**Everything you can change lives here.** Edit `CONFIG`, re-run the cell.
Default config trains in ~5–15 min on a Colab T4.
"""
    ),
    code(
        """CONFIG = {
    # Architecture
    "block_class":   "eff",                      # eff | plain | ag
    "block_sizes":   [256, 256, 128, 128, 64, 64, 32, 32],  # per-stage channels (depth = len-1 stages)
    "ks":            5,                          # conv kernel size

    # Dropout (conv gets less than dense)
    "conv_dropout":  0.1,                        # ~0.05-0.15 typical
    "dense_dims":    [],                         # e.g. [256, 64] = add MLP head; [] = pooled-conv mapper
    "dense_dropout": 0.3,                        # only used if dense_dims is non-empty

    # Optimizer + schedule
    "optimizer":     "adamw",                    # adam | adamw | muon
    "lr":            3e-3,
    "weight_decay":  0.05,

    # Training loop
    "batch_size":    512,
    "epochs":        80,
    "patience":      15,

    # Augmentations
    "use_rc_aug":    True,
    "use_shift_aug": False,
}

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False,
                          num_workers=2, pin_memory=True)

model = LegNet(
    in_channels=4,
    block_sizes=CONFIG["block_sizes"],
    ks=CONFIG["ks"],
    block_class=BLOCK_CLASSES[CONFIG["block_class"]],
    conv_dropout=CONFIG["conv_dropout"],
    dense_dims=CONFIG["dense_dims"],
    dense_dropout=CONFIG["dense_dropout"],
)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}  "
      f"(block={CONFIG['block_class']}, depth={len(CONFIG['block_sizes'])-1} stages)")

model, best_val_mse = train_model(model, train_loader, val_loader, CONFIG)
print(f"\\nBest val_mse: {best_val_mse:.4f}")
"""
    ),
    md(
        """## 8. Evaluate on test set
"""
    ),
    code(
        """def evaluate(model, loader, name="set"):
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


print("=== Your fresh training ===")
evaluate(model, val_loader,  "val ")
evaluate(model, test_loader, "test")
"""
    ),
    md(
        """## 9. Compare against the bundled reference model

The bundle ships one LegNet checkpoint as `best_model.pt` — a reasonable
reference to compare your fresh training against. Its training details
(D, label source, HPs) are saved inside the checkpoint dict.
"""
    ),
    code(
        """import os

ref_path = f"{DATA_DIR}/best_model.pt"
if not os.path.exists(ref_path):
    print("(no best_model.pt in bundle yet — skip this cell, or re-run wget when it lands)")
else:
    ckpt = torch.load(ref_path, map_location="cpu", weights_only=False)
    print(f"Reference model: D={ckpt.get('training_d', '?'):,}  "
          f"labels={ckpt.get('training_label_source', '?')!r}")
    print(f"  block_sizes={ckpt['block_sizes']}  ks={ckpt.get('ks', 5)}  "
          f"block_class={ckpt.get('block_class', 'eff')}")
    print(f"  meta: {ckpt.get('meta', {})}")

    ref = LegNet(
        in_channels=4,
        block_sizes=ckpt["block_sizes"],
        ks=ckpt.get("ks", 5),
        block_class=BLOCK_CLASSES[ckpt.get("block_class", "eff")],
        conv_dropout=ckpt.get("conv_dropout", 0.0),
        dense_dims=ckpt.get("dense_dims", []),
        dense_dropout=ckpt.get("dense_dropout", 0.0),
    )
    ref.load_state_dict(ckpt["model_state_dict"])
    ref = ref.to("cuda" if torch.cuda.is_available() else "cpu")
    print("\\n=== Reference model on held-out splits ===")
    evaluate(ref, val_loader,  "val ")
    evaluate(ref, test_loader, "test")
"""
    ),
    md(
        """## 10. Ideas to try

All of these are one-line `CONFIG` tweaks unless otherwise noted.

**Block class** — `CONFIG["block_class"] = "plain"` (vanilla conv) or `"ag"`
(AlphaGenome-style depthwise-separable). Easy to add a 4th: write a class with
`__init__(in_ch, ks)` and `forward(x: (B, in_ch, L)) -> (B, in_ch, L)`, then
`BLOCK_CLASSES["mine"] = MyBlock`.

**Optimizer** — `CONFIG["optimizer"] = "adam"` / `"adamw"` / `"muon"`.
[Muon](https://github.com/KellerJordan/Muon) requires `pip install muon`; it's
sometimes surprisingly competitive on small-data convolutional setups.

**Conv vs dense dropout** — `conv_dropout=0.1` is applied after each conv
stage; `dense_dropout=0.3` is applied inside the optional MLP head. Conv layers
generally tolerate *less* dropout than dense.

**Per-layer widths** — `block_sizes` is a list, no constraint that it has to
be monotone or use a constant ratio. Try `[256, 512, 256, 128, 64]`, etc.

**MLP head** — set `dense_dims=[256, 64]` to add an MLP after the conv stack
(replaces the pooled mapper). `dense_dropout` controls the dropout inside it.

**Augmentations** — `CONFIG["use_shift_aug"] = True` turns on shift via the
adapter-padded sliding-window crop (max ±15 bp). For EvoAug structural mutations
(deletion / insertion / inversion / translocation / tandem dup / point mutation —
[Lee & Koo 2023](https://www.biorxiv.org/content/10.1101/2023.06.16.545475v1)):
`pip install evoaug-pytorch` and apply per-batch before the model forward pass.

**Bigger / smaller Ds** — see Section 11 below for the one-cell recipe.
"""
    ),
    md(
        """## 11. (Optional) Use a custom training size

Skip this if D=20k is fine for what you want to try.

`train_full.parquet` is the full chromosome-split train pool (~617k sequences).
This cell subsamples it to whatever `D_TRAIN` you pick (deterministic via
`SUBSAMPLE_SEED`, so the same D always selects the same sequences). Then
re-run model build + `train_model(...)` to fit on the new size.
"""
    ),
    code(
        """# Pick any D in [600, 617_217]. Common picks: 500, 5_000, 100_000, 600_000.
D_TRAIN = 100_000
SUBSAMPLE_SEED = 42

train_full = pd.read_parquet(f"{DATA_DIR}/train_full.parquet")
D = min(D_TRAIN, len(train_full))
rng = np.random.default_rng(SUBSAMPLE_SEED)
idx = rng.choice(len(train_full), size=D, replace=False)
train_df_custom = train_full.iloc[idx].reset_index(drop=True)

train_ds_custom    = SeqDataset(train_df_custom)
train_loader_custom = DataLoader(train_ds_custom, batch_size=CONFIG["batch_size"],
                                  shuffle=True, num_workers=2, pin_memory=True)
print(f"Custom D={D:,} loader ready (subsampled from {len(train_full):,}, seed={SUBSAMPLE_SEED})")

# Then fit on the new loader:
# model = LegNet(in_channels=4, block_sizes=CONFIG["block_sizes"], ks=CONFIG["ks"],
#                block_class=BLOCK_CLASSES[CONFIG["block_class"]],
#                conv_dropout=CONFIG["conv_dropout"],
#                dense_dims=CONFIG["dense_dims"], dense_dropout=CONFIG["dense_dropout"])
# model, best = train_model(model, train_loader_custom, val_loader, CONFIG)
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
