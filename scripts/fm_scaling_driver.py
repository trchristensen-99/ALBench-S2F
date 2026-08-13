"""Foundation-model scaling driver (the FM analog of the from-scratch CNN scaling laws).

For a (model x reservoir dataset x D x seed) cell: load a pretrained FM (Flashzoi/Borzoi/...),
attach a regression head predicting the oracle MPRA activity, fine-tune on D oracle-labeled
reservoir sequences, evaluate on the held-out battery (+ eQTL). Records the scaling point (perf vs D).

Short-sequence (200 bp) handling: use the FM's CONV TRUNK embedding (Borzoi: conv_dna -> res_tower),
which is length-agnostic, then global-pool + Linear head — the same trick the Enformer MPRA head uses
(it drops the long-range transformer/crop that require ~500 kb inputs). The encoder stays intact so a
continual-learning wrapper (clgenomics) can attach extra heads later.

Two-stage fine-tune: (1) frozen trunk + head warmup, (2) full fine-tune. CL variant (--cl replay) via
clgenomics.ReplayContinualLearner is added AFTER the non-CL curves (per PI sequencing).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

_BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}
FLASHZOI_REPO = os.environ.get("FLASHZOI_REPO", "johahi/flashzoi-replicate-0")
BORZOI_REPO = os.environ.get("BORZOI_REPO", "johahi/borzoi-replicate-0")


def one_hot(seqs, length=None):
    """list[str] -> (N, 4, L) float tensor (channels-first for conv stems)."""
    L = length or max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), 4, L)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:L]):
            k = _BASE2IDX.get(ch)
            if k is not None:
                x[i, k, j] = 1.0
    return x


class BorzoiTrunkRegressor(nn.Module):
    """Borzoi conv trunk (conv_dna -> res_tower) -> global-avg-pool -> Linear head."""

    def __init__(self, borzoi, head_dim=1):
        super().__init__()
        self.borzoi = borzoi
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 512)
            emb = borzoi.res_tower(borzoi.conv_dna(dummy))  # (1, C, L')
        self.embed_dim = emb.shape[1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(self.embed_dim, head_dim)
        )

    def trunk(self, x):
        return self.borzoi.res_tower(self.borzoi.conv_dna(x))

    def forward(self, x):
        return self.head(self.trunk(x)).squeeze(-1)

    def set_encoder_trainable(self, flag):
        for p in self.borzoi.parameters():
            p.requires_grad = flag


def load_fm(model_name, head_dim=1):
    if model_name in ("flashzoi", "borzoi"):
        from borzoi_pytorch import Borzoi

        # We use only the CONV TRUNK (conv_dna + res_tower) — identical weights in borzoi/flashzoi;
        # FlashAttention (flashzoi) only speeds up the transformer, which we don't use. Load the
        # cached borzoi repo with flashed=False so no flash_attn dependency is needed.
        try:
            borzoi = Borzoi.from_pretrained(BORZOI_REPO, flashed=False)
        except TypeError:
            borzoi = Borzoi.from_pretrained(BORZOI_REPO)
            borzoi.flashed = False
        return BorzoiTrunkRegressor(borzoi, head_dim)
    raise NotImplementedError(f"wire {model_name} (NTv3 / AG-fold0) next")


def load_reservoir(cache_path, D, seed):
    z = np.load(cache_path, allow_pickle=True)
    key = "sequences"
    seqs = z[key]
    labels = z["oracle_labels"].astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=min(D, len(seqs)), replace=False)
    return [str(seqs[i]) for i in idx], labels[idx]


def load_battery(battery_dir, limit=None):
    out = {}
    for f in sorted(os.listdir(battery_dir)):
        if not f.endswith("_oracle.npz"):
            continue
        z = np.load(os.path.join(battery_dir, f), allow_pickle=True)
        lab = next((k for k in ("oracle_mean", "oracle_labels") if k in z.files), None)
        if "sequences" in z.files and lab:
            s = [str(x) for x in z["sequences"]]
            y = z[lab].astype(np.float32)
            if limit:
                s, y = s[:limit], y[:limit]
            out[f.replace("_oracle.npz", "")] = (s, y)
    return out


def _batches(seqs, labels, bs, L, device, shuffle=False):
    idx = np.arange(len(seqs))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        xb = one_hot([seqs[j] for j in b], L).to(device)
        yb = torch.tensor(labels[b], device=device)
        yield xb, yb


def train_fm(model, seqs, labels, args, device):
    L = max(len(s) for s in seqs)
    labels = np.asarray(labels)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4
    )
    lossf = nn.MSELoss()
    stage1 = max(1, args.epochs // 4)
    model.set_encoder_trainable(False)  # stage 1: head only
    for ep in range(args.epochs):
        if ep == stage1:  # stage 2: full fine-tune
            model.set_encoder_trainable(True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        model.train()
        tot = 0.0
        for xb, yb in _batches(seqs, labels, args.batch_size, L, device, shuffle=True):
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep} stage{1 if ep < stage1 else 2} mse={tot:.3f}", flush=True)


@torch.no_grad()
def evaluate(model, battery, device, L):
    model.eval()
    metrics = {}
    for name, (seqs, y) in battery.items():
        preds = []
        for xb, _ in _batches(seqs, np.zeros(len(seqs), np.float32), 64, L, device):
            preds.append(model(xb).cpu().numpy())
        p = np.concatenate(preds)
        m = np.isfinite(p) & np.isfinite(y)
        if m.sum() > 3:
            metrics[name] = float(pearsonr(p[m], y[m])[0])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="flashzoi", choices=["flashzoi", "borzoi", "ntv3", "alphagenome_fold0"]
    )
    ap.add_argument("--reservoir_cache")
    ap.add_argument(
        "--genomic_train",
        help="use chr_train_ref_only.npz-style npz (sequences+oracle_labels) instead of a reservoir cache",
    )
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--battery_dir", default="data/k562/test_sets_ag_s2_chrsplit")
    ap.add_argument(
        "--battery_limit", type=int, default=None, help="cap test-set size (smoke tests)"
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cl", choices=["none", "replay"], default="none")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    src = args.reservoir_cache or args.genomic_train
    seqs, labels = load_reservoir(src, args.D, args.seed)
    L = max(len(s) for s in seqs)
    battery = load_battery(args.battery_dir, args.battery_limit)
    print(
        f"[fm] model={args.model} D={len(seqs)} L={L} sets={list(battery)} device={device}",
        flush=True,
    )

    model = load_fm(args.model).to(device)
    print(f"[fm] {args.model} embed_dim={model.embed_dim}", flush=True)
    t0 = time.time()
    train_fm(model, seqs, labels, args, device)
    metrics = evaluate(model, battery, device, L)
    out = {
        "model": args.model,
        "D": len(seqs),
        "cl": args.cl,
        "train_sec": round(time.time() - t0, 1),
        "metrics": metrics,
    }
    json.dump(out, open(os.path.join(args.out_dir, "fm_scaling_point.json"), "w"), indent=2)
    print(
        f"[fm] DONE {args.model} D={len(seqs)} genomic={metrics.get('genomic')} -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
