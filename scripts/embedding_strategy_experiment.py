#!/usr/bin/env python3
"""Test different embedding extraction strategies for foundation models.

For each strategy: encode sequences → extract embeddings → train MLP head → evaluate.
Supports different flank sizes, center-bin counts, and pooling modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.k562 import K562Dataset  # noqa: E402
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM  # noqa: E402


def one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode DNA sequence to (4, L) array."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in mapping:
            arr[mapping[c], i] = 1.0
    return arr


def prepare_sequence(insert: str, flank_size: int) -> str:
    """Prepare full sequence with MPRA vector flanks."""
    if len(insert) < 200:
        pad = 200 - len(insert)
        insert = "N" * (pad // 2) + insert + "N" * (pad - pad // 2)
    elif len(insert) > 200:
        start = (len(insert) - 200) // 2
        insert = insert[start : start + 200]

    if flank_size == 0:
        return insert

    upstream = MPRA_UPSTREAM[-flank_size:] if flank_size <= len(MPRA_UPSTREAM) else MPRA_UPSTREAM
    downstream = (
        MPRA_DOWNSTREAM[:flank_size] if flank_size <= len(MPRA_DOWNSTREAM) else MPRA_DOWNSTREAM
    )
    return upstream + insert + downstream


def extract_embeddings_borzoi(
    sequences: list[str],
    flank_size: int,
    center_bins: int,
    pool_mode: str,
    device: torch.device,
    batch_size: int = 2,
) -> np.ndarray:
    """Extract Borzoi embeddings with configurable strategy."""
    import copy

    from borzoi_pytorch.modeling_borzoi import Borzoi

    if not hasattr(Borzoi, "all_tied_weights_keys"):
        Borzoi.all_tied_weights_keys = {}

    model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()

    SEQ_LEN = 196608
    all_embs = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_oh = []
        for seq in batch_seqs:
            full = prepare_sequence(seq, flank_size)
            oh = one_hot_encode(full)  # (4, L)
            # Pad to SEQ_LEN
            pad_total = SEQ_LEN - oh.shape[1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            oh = np.pad(oh, ((0, 0), (pad_left, pad_right)), mode="constant")
            batch_oh.append(oh)

        x = torch.tensor(np.stack(batch_oh), device=device)

        with torch.no_grad():
            out = model.get_embs_after_crop(x)  # (B, 1536, 6144)

        # Apply center-bin extraction
        if center_bins > 0 and center_bins < out.shape[2]:
            center = out.shape[2] // 2
            half = center_bins // 2
            out = out[:, :, center - half : center + half]  # (B, 1536, center_bins)

        # Apply pooling
        if pool_mode == "max":
            emb = out.max(dim=2).values  # (B, 1536)
        else:
            emb = out.mean(dim=2)  # (B, 1536)

        all_embs.append(emb.cpu().numpy())

        if (i // batch_size) % 100 == 0 and i > 0:
            print(f"  Borzoi: {i}/{len(sequences)}", flush=True)

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


def extract_embeddings_enformer(
    sequences: list[str],
    flank_size: int,
    center_bins: int,
    pool_mode: str,
    device: torch.device,
    batch_size: int = 2,
) -> np.ndarray:
    """Extract Enformer embeddings with configurable strategy."""
    import copy

    from enformer_pytorch import Enformer

    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = {}

    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()

    SEQ_LEN = 196608
    TARGET_LEN = 896
    all_embs = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_oh = []
        for seq in batch_seqs:
            full = prepare_sequence(seq, flank_size)
            oh = one_hot_encode(full)  # (4, L)
            # Enformer wants channels-last: (L, 4)
            oh_cl = oh.T  # (L, 4)
            pad_total = SEQ_LEN - oh_cl.shape[0]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            oh_cl = np.pad(oh_cl, ((pad_left, pad_right), (0, 0)), mode="constant")
            batch_oh.append(oh_cl)

        x = torch.tensor(np.stack(batch_oh), device=device)

        with torch.no_grad():
            emb = model(x, return_only_embeddings=True)  # (B, 896, 3072)

        # Apply center-bin extraction
        if center_bins > 0 and center_bins < emb.shape[1]:
            center = TARGET_LEN // 2
            half = center_bins // 2
            emb = emb[:, center - half : center + half, :]  # (B, center_bins, 3072)

        # Apply pooling
        if pool_mode == "max":
            emb = emb.max(dim=1).values  # (B, 3072)
        else:
            emb = emb.mean(dim=1)  # (B, 3072)

        all_embs.append(emb.cpu().numpy())

        if (i // batch_size) % 100 == 0 and i > 0:
            print(f"  Enformer: {i}/{len(sequences)}", flush=True)

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


def extract_embeddings_ntv3(
    sequences: list[str],
    flank_size: int,
    center_bins: int,
    pool_mode: str,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract NTv3 embeddings with configurable flank strategy."""
    from models.nt_v3_wrapper import NTv3Wrapper

    wrapper = NTv3Wrapper()
    all_embs = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        prepped = [prepare_sequence(s, flank_size) for s in batch_seqs]
        emb = wrapper.extract_embeddings(prepped)  # (B, 1536)
        all_embs.append(emb)

        if (i // batch_size) % 100 == 0 and i > 0:
            print(f"  NTv3: {i}/{len(sequences)}", flush=True)

    return np.concatenate(all_embs, axis=0)


class MLPHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_evaluate(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    embed_dim: int,
    lr: float = 3e-4,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 7,
) -> dict:
    """Train MLP head on embeddings and evaluate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPHead(embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    train_ds = TensorDataset(
        torch.tensor(train_emb, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val_r = -1
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = (
                model(torch.tensor(val_emb, dtype=torch.float32, device=device)).cpu().numpy()
            )
        val_r = float(pearsonr(val_pred, val_labels)[0]) if len(val_pred) > 2 else 0.0

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: val_r={val_r:.4f} (best={best_val_r:.4f})", flush=True)

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(test_emb, dtype=torch.float32, device=device)).cpu().numpy()

    mask = np.isfinite(test_labels)
    test_r = float(pearsonr(test_pred[mask], test_labels[mask])[0])
    test_sp = float(spearmanr(test_pred[mask], test_labels[mask])[0])
    test_mse = float(np.mean((test_pred[mask] - test_labels[mask]) ** 2))

    return {
        "best_val_pearson_r": best_val_r,
        "test_metrics": {
            "in_dist": {
                "pearson_r": test_r,
                "spearman_r": test_sp,
                "mse": test_mse,
                "n": int(mask.sum()),
            }
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["borzoi", "enformer", "ntv3_post"])
    parser.add_argument("--flank-size", type=int, default=200)
    parser.add_argument("--center-bins", type=int, default=0, help="0 = all bins")
    parser.add_argument("--pool-mode", default="mean", choices=["mean", "max"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default="data/k562")
    parser.add_argument("--cell-line", default="k562")
    parser.add_argument("--max-train", type=int, default=20000, help="Max training sequences")
    parser.add_argument("--max-val", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"Result exists: {result_path}, skipping.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_col = {"k562": "K562_log2FC", "hepg2": "HepG2_log2FC", "sknsh": "SKNSH_log2FC"}.get(
        args.cell_line, "K562_log2FC"
    )

    # Load data
    print("Loading data...", flush=True)
    train_ds = K562Dataset(args.data_path, split="train", label_column=label_col)
    val_ds = K562Dataset(args.data_path, split="val", label_column=label_col)
    test_ds = K562Dataset(args.data_path, split="test", label_column=label_col)

    # Subsample for speed
    rng = np.random.default_rng(42)
    if len(train_ds) > args.max_train:
        idx = rng.choice(len(train_ds), args.max_train, replace=False)
        train_seqs = [train_ds.sequences[i] for i in idx]
        train_labels = train_ds.labels[idx].astype(np.float32)
    else:
        train_seqs = list(train_ds.sequences)
        train_labels = train_ds.labels.astype(np.float32)

    if len(val_ds) > args.max_val:
        idx = rng.choice(len(val_ds), args.max_val, replace=False)
        val_seqs = [val_ds.sequences[i] for i in idx]
        val_labels = val_ds.labels[idx].astype(np.float32)
    else:
        val_seqs = list(val_ds.sequences)
        val_labels = val_ds.labels.astype(np.float32)

    test_seqs = list(test_ds.sequences)
    test_labels = test_ds.labels.astype(np.float32)

    print(
        f"Data: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}",
        flush=True,
    )

    # Extract embeddings
    print(
        f"Extracting {args.model} embeddings (flank={args.flank_size}, center={args.center_bins}, pool={args.pool_mode})...",
        flush=True,
    )
    t0 = time.time()

    extract_fn = {
        "borzoi": extract_embeddings_borzoi,
        "enformer": extract_embeddings_enformer,
        "ntv3_post": extract_embeddings_ntv3,
    }[args.model]

    all_seqs = train_seqs + val_seqs + test_seqs
    all_emb = extract_fn(all_seqs, args.flank_size, args.center_bins, args.pool_mode, device)

    train_emb = all_emb[: len(train_seqs)]
    val_emb = all_emb[len(train_seqs) : len(train_seqs) + len(val_seqs)]
    test_emb = all_emb[len(train_seqs) + len(val_seqs) :]

    embed_dim = train_emb.shape[1]
    print(f"Embeddings: dim={embed_dim}, took {time.time() - t0:.1f}s", flush=True)

    # Train and evaluate
    print("Training head...", flush=True)
    result = train_and_evaluate(
        train_emb, train_labels, val_emb, val_labels, test_emb, test_labels, embed_dim
    )

    result["config"] = {
        "model": args.model,
        "flank_size": args.flank_size,
        "center_bins": args.center_bins,
        "pool_mode": args.pool_mode,
        "embed_dim": embed_dim,
        "n_train": len(train_seqs),
    }

    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"\nResult: val_r={result['best_val_pearson_r']:.4f}, test_r={result['test_metrics']['in_dist']['pearson_r']:.4f}"
    )
    print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
