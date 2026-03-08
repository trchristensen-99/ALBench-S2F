#!/usr/bin/env python
"""Stage 2: Fine-tune Enformer or Borzoi encoder + MLP head on K562 MPRA data.

Loads the best Stage 1 (head-only) checkpoint and selectively unfreezes
encoder layers for joint fine-tuning with per-group learning rates.

Supported models:
  - Enformer (lucidrains enformer-pytorch, 251M params, 3072D embeddings)
  - Borzoi (borzoi-pytorch, 186M params, 1536D embeddings)

Both models use 196,608bp input (600bp MPRA sequences zero-padded).
Training uses gradient accumulation and mixed precision to fit on a single GPU.

Usage::

    python experiments/train_foundation_stage2.py \
        ++model_name=enformer \
        ++stage1_result_dir=outputs/foundation_grid_search/enformer/.../seed_42 \
        ++output_dir=outputs/enformer_k562_stage2/sweep_elr1e-4_transformer

Run via SLURM::

    sbatch scripts/slurm/enformer_stage2_sweep.sh
    sbatch scripts/slurm/borzoi_stage2_sweep.sh
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model_name": "enformer",  # "enformer" or "borzoi"
    "stage1_result_dir": None,  # path to best S1 seed dir (has best_model.pt)
    "output_dir": "outputs/enformer_k562_stage2",
    "data_path": "data/k562",
    "seed": None,
    "epochs": 15,
    "batch_size": 4,  # micro-batch (limited by GPU memory)
    "grad_accum_steps": 2,  # effective BS = batch_size * grad_accum_steps = 8
    "head_lr": 1e-3,
    "encoder_lr": 1e-4,
    "weight_decay": 1e-6,
    "hidden_dim": 512,
    "dropout": 0.1,
    "early_stop_patience": 5,
    "max_train_sequences": 20000,  # subsample train set for speed
    "max_val_sequences": 2000,  # subsample val set for speed
    "rc_aug": True,
    "unfreeze_mode": "transformer",  # "transformer" or "all"
    "grad_clip": 1.0,
    "num_workers": 4,
    "use_amp": True,  # bfloat16 autocast; set False for float32 (more stable)
}

# ── MPRA flanks as one-hot arrays ────────────────────────────────────────────
_FLANK_5_STR = MPRA_UPSTREAM[-200:]
_FLANK_3_STR = MPRA_DOWNSTREAM[:200]

_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}

_FLANK_5_ENC = np.zeros((4, 200), dtype=np.float32)
for _i, _c in enumerate(_FLANK_5_STR):
    if _c in _MAPPING:
        _FLANK_5_ENC[_MAPPING[_c], _i] = 1.0

_FLANK_3_ENC = np.zeros((4, 200), dtype=np.float32)
for _i, _c in enumerate(_FLANK_3_STR):
    if _c in _MAPPING:
        _FLANK_3_ENC[_MAPPING[_c], _i] = 1.0

# ── Constants ────────────────────────────────────────────────────────────────
ENFORMER_SEQ_LEN = 196_608
ENFORMER_TARGET_LEN = 896
ENFORMER_EMBED_DIM = 3072

BORZOI_SEQ_LEN = 196_608
BORZOI_EMBED_DIM = 1536


# ── MLP Head (matches train_foundation_cached.py) ───────────────────────────
class MLPHead(nn.Module):
    """LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(1)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_corr(pred: np.ndarray, target: np.ndarray, fn) -> float:
    if pred.size < 2 or np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def _add_flanks(oh_4ch: np.ndarray) -> np.ndarray:
    """Add MPRA flanks to (4, 200) one-hot -> (4, 600)."""
    return np.concatenate([_FLANK_5_ENC, oh_4ch, _FLANK_3_ENC], axis=1)


def _rc_onehot(oh: np.ndarray) -> np.ndarray:
    """Reverse complement (4, L) one-hot: flip channels and sequence."""
    return oh[::-1, ::-1].copy()


# ── Collate functions ────────────────────────────────────────────────────────
def _collate_train(batch, rc_aug: bool = True):
    """Collate K562Dataset items -> (B, 4, 600) one-hot with optional RC aug."""
    ohs = []
    labels = []
    for seq_5ch, label in batch:
        oh_4ch = np.asarray(seq_5ch)[:4]  # (4, 200)
        oh_600 = _add_flanks(oh_4ch)  # (4, 600)
        if rc_aug and np.random.rand() > 0.5:
            oh_600 = _rc_onehot(oh_600)
        ohs.append(oh_600)
        labels.append(float(label.numpy()) if hasattr(label, "numpy") else float(label))
    return (
        torch.from_numpy(np.stack(ohs)).float(),
        torch.tensor(labels, dtype=torch.float32),
    )


def _collate_eval(batch):
    """Collate for eval: returns canonical + RC one-hots for RC-averaged prediction."""
    can_ohs = []
    rc_ohs = []
    labels = []
    for seq_5ch, label in batch:
        oh_4ch = np.asarray(seq_5ch)[:4]
        oh_600 = _add_flanks(oh_4ch)
        can_ohs.append(oh_600)
        rc_ohs.append(_rc_onehot(oh_600))
        labels.append(float(label.numpy()) if hasattr(label, "numpy") else float(label))
    return (
        torch.from_numpy(np.stack(can_ohs)).float(),
        torch.from_numpy(np.stack(rc_ohs)).float(),
        torch.tensor(labels, dtype=torch.float32),
    )


# ── Model loading ───────────────────────────────────────────────────────────
def _load_enformer():
    from enformer_pytorch import Enformer

    # Fix for transformers>=5.3 which expects all_tied_weights_keys
    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = {}

    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    return model, ENFORMER_EMBED_DIM


def _load_borzoi():
    from borzoi_pytorch import Borzoi

    # Fix for transformers>=5.3 which expects all_tied_weights_keys
    if not hasattr(Borzoi, "all_tied_weights_keys"):
        Borzoi.all_tied_weights_keys = {}

    model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    return model, BORZOI_EMBED_DIM


# ── Forward pass functions ───────────────────────────────────────────────────
def _forward_enformer(model, one_hot_batch):
    """Forward pass through Enformer: (B, 4, 600) -> (B, 3072)."""
    # Enformer expects channels-last: (B, L, 4)
    seqs_cl = one_hot_batch.permute(0, 2, 1)  # (B, 600, 4)
    pad_total = ENFORMER_SEQ_LEN - seqs_cl.shape[1]
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = F.pad(seqs_cl, (0, 0, pad_left, pad_right), value=0.0)  # (B, 196608, 4)
    emb = model(padded, return_only_embeddings=True)  # (B, 896, 3072)
    center = ENFORMER_TARGET_LEN // 2  # 448
    center_emb = emb[:, center - 2 : center + 2, :]  # center 4 bins
    return center_emb.mean(dim=1)  # (B, 3072)


BORZOI_NUM_BINS = 6144
BORZOI_BIN_SIZE = 32  # 196608 / 6144 = 32bp per bin
# Center bins covering the actual 600bp sequence region (~19 bins at 32bp)
BORZOI_N_CENTER_BINS = 20


def _forward_borzoi(model, one_hot_batch):
    """Forward pass through Borzoi: (B, 4, 600) -> (B, 1536).

    Uses center bins (covering the actual sequence region) instead of
    mean-pooling all 6144 bins, which are 99.7% zero-padded and cause
    gradient instability during S2 fine-tuning.
    """
    pad_total = BORZOI_SEQ_LEN - one_hot_batch.shape[2]
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = F.pad(one_hot_batch, (pad_left, pad_right), value=0.0)  # (B, 4, 196608)
    emb = model.get_embs_after_crop(padded)  # (B, 1536, 6144)
    # Extract center bins corresponding to the actual sequence region
    center = BORZOI_NUM_BINS // 2  # 3072
    half = BORZOI_N_CENTER_BINS // 2
    center_emb = emb[:, :, center - half : center + half]  # (B, 1536, 20)
    return center_emb.mean(dim=2)  # (B, 1536)


# ── Selective unfreezing ─────────────────────────────────────────────────────
def _should_unfreeze_enformer(name: str, mode: str) -> bool:
    """Determine if an Enformer parameter should be unfrozen.

    Enformer registers modules as direct attributes (not via _trunk Sequential),
    so parameter names are: transformer.{block}.*, stem.*, conv_tower.*, etc.
    """
    if mode == "all":
        return True
    if mode == "transformer":
        return name.startswith("transformer.")
    return False


def _should_unfreeze_borzoi(name: str, mode: str) -> bool:
    """Determine if a Borzoi parameter should be unfrozen.

    Borzoi stores the transformer as self.transformer (8 blocks: 0-7).
    Modes:
      - "transformer": all 8 transformer blocks (126M params — often unstable)
      - "transformer_last2": only blocks 6-7 (~31M params — much more stable)
      - "all": entire encoder
    """
    if mode == "all":
        return True
    if mode == "transformer":
        return name.startswith("transformer.")
    if mode == "transformer_last2":
        for blk in (6, 7):
            if name.startswith(f"transformer.{blk}."):
                return True
        return False
    return False


# ── Test evaluation ──────────────────────────────────────────────────────────
def _predict_test_sequences(
    encoder_model,
    head,
    forward_fn,
    sequences_200bp: list[str],
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """RC-averaged predictions on raw 200bp test strings."""
    from data.utils import one_hot_encode

    if not sequences_200bp:
        return np.array([], dtype=np.float32)

    preds_fwd = []
    preds_rev = []

    encoder_model.eval()
    head.eval()

    for i in range(0, len(sequences_200bp), batch_size):
        batch_seqs = sequences_200bp[i : i + batch_size]

        can_batch = []
        rc_batch = []
        for seq in batch_seqs:
            seq = seq.upper()
            if len(seq) < 200:
                pad = 200 - len(seq)
                seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
            elif len(seq) > 200:
                start = (len(seq) - 200) // 2
                seq = seq[start : start + 200]
            oh = one_hot_encode(seq, add_singleton_channel=False)  # (4, 200)
            oh_600 = _add_flanks(oh)  # (4, 600)
            can_batch.append(oh_600)
            rc_batch.append(_rc_onehot(oh_600))

        can_t = torch.from_numpy(np.stack(can_batch)).float().to(device)
        rc_t = torch.from_numpy(np.stack(rc_batch)).float().to(device)

        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"),
        ):
            emb_can = forward_fn(encoder_model, can_t)
            emb_rc = forward_fn(encoder_model, rc_t)
            p_can = head(emb_can)
            p_rc = head(emb_rc)

        preds_fwd.append(p_can.cpu().float().numpy())
        preds_rev.append(p_rc.cpu().float().numpy())

    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def evaluate_all_test_sets(
    encoder_model,
    head,
    forward_fn,
    test_set_dir: Path,
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, dict[str, float]]:
    """Evaluate on hashFrag in-dist / SNV / OOD test sets."""
    import pandas as pd

    metrics: dict[str, dict[str, float]] = {}

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        df = pd.read_csv(in_path, sep="\t")
        pred = _predict_test_sequences(
            encoder_model, head, forward_fn, df["sequence"].tolist(), device, batch_size
        )
        true = df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["in_distribution"] = {
            "pearson_r": _safe_corr(pred, true, pearsonr),
            "spearman_r": _safe_corr(pred, true, spearmanr),
            "mse": float(np.mean((pred - true) ** 2)),
            "n": int(len(true)),
        }

    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        df = pd.read_csv(snv_path, sep="\t")
        ref_pred = _predict_test_sequences(
            encoder_model, head, forward_fn, df["sequence_ref"].tolist(), device, batch_size
        )
        alt_pred = _predict_test_sequences(
            encoder_model, head, forward_fn, df["sequence_alt"].tolist(), device, batch_size
        )
        alt_true = df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
        metrics["snv_abs"] = {
            "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
            "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            "n": int(len(alt_true)),
        }
        delta_pred = alt_pred - ref_pred
        delta_true = df["delta_log2FC"].to_numpy(dtype=np.float32)
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            "n": int(len(delta_true)),
        }

    ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        df = pd.read_csv(ood_path, sep="\t")
        pred = _predict_test_sequences(
            encoder_model, head, forward_fn, df["sequence"].tolist(), device, batch_size
        )
        true = df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["ood"] = {
            "pearson_r": _safe_corr(pred, true, pearsonr),
            "spearman_r": _safe_corr(pred, true, spearmanr),
            "mse": float(np.mean((pred - true) ** 2)),
            "n": int(len(true)),
        }

    return metrics


# ── Training loop ────────────────────────────────────────────────────────────
def train(cfg: dict):
    seed = cfg["seed"]
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    else:
        seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model_name"]

    # ── Load encoder ─────────────────────────────────────────────────────────
    print(f"Loading {model_name} encoder ...", flush=True)
    t0 = time.time()
    if model_name == "enformer":
        encoder_model, embed_dim = _load_enformer()
        forward_fn = _forward_enformer
        should_unfreeze = _should_unfreeze_enformer
    elif model_name == "borzoi":
        encoder_model, embed_dim = _load_borzoi()
        forward_fn = _forward_borzoi
        should_unfreeze = _should_unfreeze_borzoi
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    encoder_model.to(device)
    encoder_model.eval()
    print(f"Encoder loaded in {time.time() - t0:.1f}s", flush=True)

    # ── Borzoi diagnostic: test full pipeline with real data ──────────────
    if model_name == "borzoi":
        _tmphead = MLPHead(embed_dim, int(cfg["hidden_dim"]), float(cfg["dropout"]))
        _s1tmp = cfg["stage1_result_dir"]
        if _s1tmp:
            _ck = torch.load(
                Path(_s1tmp).expanduser().resolve() / "best_model.pt",
                map_location="cpu",
                weights_only=True,
            )
            _tmphead.load_state_dict(_ck["model_state_dict"])
        _tmphead.to(device)
        _tmphead.eval()

        # Get first real batch
        _tmpds = K562Dataset(data_path=str(Path(cfg["data_path"])), split="train")
        _tmploader = DataLoader(
            _tmpds, batch_size=4, collate_fn=lambda b: _collate_train(b, rc_aug=False)
        )
        _oh, _lbl = next(iter(_tmploader))
        _oh = _oh.to(device)
        _lbl = _lbl.to(device)

        # Test 1: no_grad forward
        with torch.no_grad():
            _emb = forward_fn(encoder_model, _oh)
            _pred = _tmphead(_emb)
            _loss = F.mse_loss(_pred, _lbl)
            print(
                f"[DIAG] no_grad: emb_nan={torch.isnan(_emb).any().item()} "
                f"emb_range=[{_emb.min().item():.2f}, {_emb.max().item():.2f}] "
                f"pred_nan={torch.isnan(_pred).any().item()} "
                f"pred_range=[{_pred.min().item():.4f}, {_pred.max().item():.4f}] "
                f"loss={_loss.item():.4f}",
                flush=True,
            )

        # Test 2: with grad on last 2 blocks
        for _n, _p in encoder_model.named_parameters():
            if _n.startswith("transformer.6.") or _n.startswith("transformer.7."):
                _p.requires_grad = True
        _emb2 = forward_fn(encoder_model, _oh)
        _pred2 = _tmphead(_emb2)
        _loss2 = F.mse_loss(_pred2, _lbl)
        print(
            f"[DIAG] with_grad: emb_nan={torch.isnan(_emb2).any().item()} "
            f"emb_range=[{_emb2.min().item():.2f}, {_emb2.max().item():.2f}] "
            f"pred_nan={torch.isnan(_pred2).any().item()} "
            f"pred_range=[{_pred2.min().item():.4f}, {_pred2.max().item():.4f}] "
            f"loss={_loss2.item():.4f}",
            flush=True,
        )

        # Reset
        for _p in encoder_model.parameters():
            _p.requires_grad = False
        del _tmphead, _tmpds, _tmploader, _oh, _lbl, _emb, _pred, _emb2, _pred2
        torch.cuda.empty_cache()

    # ── Load S1 head ─────────────────────────────────────────────────────────
    hidden_dim = int(cfg["hidden_dim"])
    dropout = float(cfg["dropout"])
    head = MLPHead(embed_dim, hidden_dim, dropout)

    s1_dir = cfg["stage1_result_dir"]
    if s1_dir is not None:
        s1_dir = Path(s1_dir).expanduser().resolve()
        ckpt = torch.load(s1_dir / "best_model.pt", map_location="cpu", weights_only=True)
        head.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Loaded S1 head from {s1_dir / 'best_model.pt'} (epoch {ckpt.get('epoch', '?')})",
            flush=True,
        )
    else:
        print("[WARN] No stage1_result_dir — starting head from random init", flush=True)

    head.to(device)

    # ── Selective unfreezing ─────────────────────────────────────────────────
    unfreeze_mode = cfg["unfreeze_mode"]
    encoder_params = []
    frozen_count = 0

    for name, param in encoder_model.named_parameters():
        if should_unfreeze(name, unfreeze_mode):
            param.requires_grad = True
            encoder_params.append(param)
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    head_params = list(head.parameters())
    n_head = sum(p.numel() for p in head_params)
    n_encoder = sum(p.numel() for p in encoder_params)

    print(
        f"Param groups — head: {n_head:,}  "
        f"encoder (trainable): {n_encoder:,}  "
        f"frozen: {frozen_count:,}  "
        f"total: {n_head + n_encoder + frozen_count:,}",
        flush=True,
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    head_lr = float(cfg["head_lr"])
    encoder_lr = float(cfg["encoder_lr"])
    wd = float(cfg["weight_decay"])
    grad_clip = float(cfg["grad_clip"])

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr, "weight_decay": wd},
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": wd},
        ]
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_path = Path(cfg["data_path"])
    train_ds = K562Dataset(data_path=str(data_path), split="train")
    val_ds = K562Dataset(data_path=str(data_path), split="val")

    max_train = int(cfg["max_train_sequences"])
    max_val = int(cfg["max_val_sequences"])

    if 0 < max_train < len(train_ds):
        rng = np.random.RandomState(seed)
        train_indices = rng.choice(len(train_ds), max_train, replace=False)
        train_ds = Subset(train_ds, train_indices)
        print(f"Subsampled train: {len(train_ds):,} sequences", flush=True)

    if 0 < max_val < len(val_ds):
        rng = np.random.RandomState(seed + 1)
        val_indices = rng.choice(len(val_ds), max_val, replace=False)
        val_ds = Subset(val_ds, val_indices)
        print(f"Subsampled val: {len(val_ds):,} sequences", flush=True)

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)

    rc_aug = bool(cfg["rc_aug"])
    batch_size = int(cfg["batch_size"])
    grad_accum_steps = int(cfg["grad_accum_steps"])
    n_workers = int(cfg["num_workers"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=lambda b: _collate_train(b, rc_aug=rc_aug),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=_collate_eval,
        pin_memory=True,
    )

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    # bfloat16 autocast (better dynamic range than float16, no GradScaler needed)
    use_amp = device.type == "cuda" and bool(cfg["use_amp"])
    amp_dtype = torch.bfloat16
    print(f"Mixed precision (bfloat16): {'enabled' if use_amp else 'disabled'}", flush=True)
    best_val_pearson = -1.0
    epochs_no_improve = 0
    early_stop_patience = int(cfg["early_stop_patience"])
    n_train_batches = len(train_loader)
    all_trainable = list(head.parameters()) + encoder_params

    for epoch in range(int(cfg["epochs"])):
        t_epoch = time.time()
        # Keep encoder in eval mode: disables dropout/BN updates for numerical
        # stability with 196K zero-padded input. Gradients still flow through
        # unfrozen params (requires_grad=True is independent of train/eval mode).
        encoder_model.eval()
        head.train()
        train_losses: list[float] = []

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (oh_batch, labels) in enumerate(train_loader):
            oh_batch = oh_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                emb = forward_fn(encoder_model, oh_batch)
                pred = head(emb)
                loss = F.mse_loss(pred, labels) / grad_accum_steps

            if torch.isnan(loss):
                print(
                    f"  [WARN] NaN loss at batch {batch_idx + 1}, skipping",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            train_losses.append(loss.item() * grad_accum_steps)

            is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == n_train_batches
            if is_accum_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(all_trainable, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch + 1} batch {batch_idx + 1}/{n_train_batches} "
                    f"loss={train_losses[-1]:.4f}",
                    flush=True,
                )

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ── Validation (RC-averaged) ─────────────────────────────────────────
        encoder_model.eval()
        head.eval()
        val_preds: list[np.ndarray] = []
        val_trues: list[np.ndarray] = []

        with torch.no_grad():
            for can_batch, rc_batch, labels in val_loader:
                can_batch = can_batch.to(device, non_blocking=True)
                rc_batch = rc_batch.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    emb_can = forward_fn(encoder_model, can_batch)
                    emb_rc = forward_fn(encoder_model, rc_batch)
                    p_can = head(emb_can)
                    p_rc = head(emb_rc)

                avg_pred = ((p_can + p_rc) / 2.0).cpu().float().numpy()
                val_preds.append(avg_pred)
                val_trues.append(labels.numpy())

        val_preds_arr = np.concatenate(val_preds)
        val_trues_arr = np.concatenate(val_trues)
        val_pearson = _safe_corr(val_preds_arr, val_trues_arr, pearsonr)
        val_spearman = _safe_corr(val_preds_arr, val_trues_arr, spearmanr)
        epoch_time = time.time() - t_epoch

        print(
            f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}  "
            f"val_pearson={val_pearson:.4f}  val_spearman={val_spearman:.4f}  "
            f"time={epoch_time:.1f}s",
            flush=True,
        )

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            epochs_no_improve = 0
            torch.save(
                {"model_state_dict": head.state_dict(), "epoch": epoch, "seed": seed},
                output_dir / "best_model.pt",
            )
            torch.save(
                {
                    "model_state_dict": encoder_model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                },
                output_dir / "best_encoder.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val Pearson={best_val_pearson:.4f})",
                    flush=True,
                )
                break

    # ── Restore best and evaluate on test sets ───────────────────────────────
    ckpt = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(ckpt["model_state_dict"])
    head.to(device)

    enc_ckpt = torch.load(output_dir / "best_encoder.pt", map_location="cpu", weights_only=False)
    encoder_model.load_state_dict(enc_ckpt["model_state_dict"])
    encoder_model.to(device)

    print("\n[eval] Evaluating on test sets ...", flush=True)
    test_set_dir = data_path / "test_sets"
    test_metrics = evaluate_all_test_sets(
        encoder_model, head, forward_fn, test_set_dir, device, batch_size=batch_size
    )

    results = {
        "model": model_name,
        "seed": seed,
        "encoder_lr": float(cfg["encoder_lr"]),
        "head_lr": float(cfg["head_lr"]),
        "unfreeze_mode": unfreeze_mode,
        "best_val_pearson": best_val_pearson,
        "best_epoch": int(ckpt["epoch"]) + 1,
        "n_train": len(train_ds),
        "param_counts": {
            "head": n_head,
            "encoder_trainable": n_encoder,
            "frozen": frozen_count,
        },
        "test_metrics": test_metrics,
        "config": {k: str(v) for k, v in cfg.items()},
    }
    out_json = output_dir / "result.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}", flush=True)

    for test_set, m in test_metrics.items():
        print(
            f"  {test_set}: pearson_r={m.get('pearson_r', 0.0):.4f}  "
            f"spearman_r={m.get('spearman_r', 0.0):.4f}  mse={m.get('mse', 0.0):.4f}",
            flush=True,
        )

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    cfg = dict(DEFAULT_CONFIG)
    for arg in sys.argv[1:]:
        if arg.startswith("++"):
            key, _, val = arg[2:].partition("=")
            if key in cfg:
                default_val = cfg[key]
                if default_val is None:
                    cfg[key] = None if val.lower() == "none" else val
                elif isinstance(default_val, bool):
                    cfg[key] = val.lower() in ("true", "1", "yes")
                elif isinstance(default_val, int):
                    cfg[key] = int(val)
                elif isinstance(default_val, float):
                    cfg[key] = float(val)
                else:
                    cfg[key] = val

    print("Config:", json.dumps({k: str(v) for k, v in cfg.items()}, indent=2), flush=True)
    train(cfg)


if __name__ == "__main__":
    main()
