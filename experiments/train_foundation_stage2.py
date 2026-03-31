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
    "max_test_sequences": 10000,  # subsample test set for speed (full Enformer eval on 40K takes ~50h)
    "rc_aug": True,
    "unfreeze_mode": "transformer",  # "transformer" or "all"
    "grad_clip": 1.0,
    "num_workers": 4,
    "amp_mode": "bfloat16",  # "bfloat16", "fp16", or "off"
    "save_encoder": False,  # skip encoder checkpoint to save disk (~1GB each)
    "borzoi_center_bins": 32,  # pool center N bins (0 = all bins); Borzoi only
    "head_warmup_epochs": 0,  # train head with encoder frozen before unfreezing
    "normalize_embeddings": False,  # L2-normalize encoder embeddings before head
    "cell_line": "k562",
    # LoRA adapter config (Borzoi only)
    "use_lora": False,  # inject LoRA adapters into Borzoi transformer blocks
    "lora_rank": 32,  # bottleneck dimension for adapters
    "lora_blocks": "",  # comma-separated block indices, e.g., "4,5,6,7"
}

CELL_LINE_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
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
    import copy

    from enformer_pytorch import Enformer

    # Fix for transformers>=5.3 which expects all_tied_weights_keys
    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = {}

    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    # Deep copy to prevent safetensors memory-mapping corruption.
    model = copy.deepcopy(model)
    return model, ENFORMER_EMBED_DIM


def _fix_borzoi_attention(model):
    """Fix Borzoi attention: NaN positions + correct relative shift.

    1. Recomputes NaN-corrupted position buffers (``from_pretrained``
       leaves blocks 2-5 with NaN on PyTorch 2.10+).
    2. Replaces ``fast_relative_shift`` (uses ``as_strided`` stride
       tricks that are both incorrect for N≠4096 and give different
       results across PyTorch versions) with a correct ``torch.gather``
       implementation that works for any sequence length.
    """
    from borzoi_pytorch.pytorch_borzoi_transformer import get_positional_embed

    # 1. Recompute NaN positions at seq_len=4096 (matching original training).
    for blk in model.transformer:
        attn = blk[0].fn[1]  # Residual → Sequential[LN, Attention, Dropout]
        device = attn.to_v.weight.device
        attn.positions = get_positional_embed(4096, attn.num_rel_pos_features, device)

    # 2. Monkey-patch with correct relative shift (no as_strided).
    from borzoi_pytorch.pytorch_borzoi_transformer import Attention
    from einops import rearrange
    from torch import einsum

    def _safe_relative_shift_batched(q_bhnd, rel_k_hmd):
        """Compute relative-position logits without vmap (autograd-safe).

        Args:
            q_bhnd: (B, H, N, d) queries (with rel_pos_bias already added)
            rel_k_hmd: (H, M, d) relative-position keys (M = 2*N-1)

        Returns:
            (B, H, N, N) relative logits where [b,h,i,j] = dot(q[b,h,i], rel_k[h, center+j-i])
        """
        B, H, N, d = q_bhnd.shape
        M = rel_k_hmd.shape[1]
        center = M // 2

        # raw[b,h,i,m] = dot(q[b,h,i,:], rel_k[h,m,:])
        raw = einsum("b h i d, h m d -> b h i m", q_bhnd, rel_k_hmd)  # (B, H, N, M)

        # Build Toeplitz indices: col_indices[i,j] = center + j - i
        rows = torch.arange(N, device=q_bhnd.device)
        col_indices = center + rows.unsqueeze(0) - rows.unsqueeze(1)  # (N, N)
        col_indices = col_indices.clamp(0, M - 1)

        # Expand for gather: (1, 1, N, N) → (B, H, N, N)
        idx = col_indices.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
        return torch.gather(raw, 3, idx)  # (B, H, N, N)

    def _patched_forward(self, x):
        n, h = x.shape[-2], self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale

        content_logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)

        # Slice center 2*n-1 positions (correct range for actual seq length)
        pos_center = self.positions.shape[0] // 2  # 4095
        positions = self.pos_dropout(self.positions[pos_center - (n - 1) : pos_center + n])
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = _safe_relative_shift_batched(q + self.rel_pos_bias, rel_k)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out

    Attention.forward = _patched_forward


def _load_borzoi():
    from borzoi_pytorch import Borzoi

    # Fix for transformers>=5.3 which expects all_tied_weights_keys
    if not hasattr(Borzoi, "all_tied_weights_keys"):
        Borzoi.all_tied_weights_keys = {}

    model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    # Fix NaN positions + replace broken fast_relative_shift.
    _fix_borzoi_attention(model)
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


class BottleneckAdapter(nn.Module):
    """Residual bottleneck adapter for LoRA-style fine-tuning."""

    def __init__(self, dim: int, rank: int = 32, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        # Initialize up projection to near-zero so adapter starts as identity
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.up(self.act(self.down(x))))


def inject_lora_adapters(model, block_indices: list[int], rank: int = 32):
    """Inject LoRA adapters into Borzoi transformer blocks.

    Wraps each specified transformer block so the adapter runs after it.
    Returns the list of adapter modules (for optimizer).
    """
    adapters = nn.ModuleList()
    dim = 1536  # Borzoi hidden dim
    new_blocks = list(model.transformer)
    for idx in block_indices:
        adapter = BottleneckAdapter(dim, rank)
        original_block = new_blocks[idx]

        class AdaptedBlock(nn.Module):
            def __init__(self, block, adapt):
                super().__init__()
                self.block = block
                self.adapter = adapt

            def forward(self, x):
                return self.adapter(self.block(x))

        new_blocks[idx] = AdaptedBlock(original_block, adapter)
        adapters.append(adapter)
    model.transformer = nn.Sequential(*new_blocks)
    return adapters


def _forward_borzoi(model, one_hot_batch, center_bins: int = 0, normalize: bool = False):
    """Forward pass through Borzoi: (B, 4, 600) -> (B, 1536).

    Args:
        center_bins: If > 0, pool only center N bins (where the 600bp insert
            maps to ~19 bins at 32bp resolution). This concentrates gradient
            signal on the actual sequence instead of diluting it across all
            6144 bins of mostly zero-padding background.  Set to 0 to mean-pool
            all bins (matches S1 embedding cache, but gradients are 200x weaker).
        normalize: If True, L2-normalize embeddings. This removes the dominant
            common component (~0.9999 cosine similarity) and makes inter-sequence
            differences the primary signal, preventing encoder updates from
            overwhelming the tiny task-relevant variation.
    """
    pad_total = BORZOI_SEQ_LEN - one_hot_batch.shape[2]
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = F.pad(one_hot_batch, (pad_left, pad_right), value=0.0)  # (B, 4, 196608)
    emb = model.get_embs_after_crop(padded)  # (B, 1536, 6144)
    if center_bins > 0:
        total = emb.shape[2]
        c = total // 2
        h = center_bins // 2
        emb = emb[:, :, c - h : c + h]  # (B, 1536, center_bins)
    emb = emb.mean(dim=2)  # (B, 1536)
    if normalize:
        emb = F.normalize(emb, p=2, dim=-1)
    return emb


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
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
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

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                emb_can = forward_fn(encoder_model, can_t)
                emb_rc = forward_fn(encoder_model, rc_t)
            # Head in fp32 to preserve inter-sample embedding differences
            p_can = head(emb_can.float())
            p_rc = head(emb_rc.float())

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
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    cell_line: str = "k562",
    max_test_sequences: int = 0,
) -> dict[str, dict[str, float]]:
    """Evaluate on hashFrag in-dist / SNV / OOD test sets."""
    import pandas as pd

    fc_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    metrics: dict[str, dict[str, float]] = {}

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        df = pd.read_csv(in_path, sep="\t")
        if 0 < max_test_sequences < len(df):
            df = df.sample(n=max_test_sequences, random_state=42)
        pred = _predict_test_sequences(
            encoder_model,
            head,
            forward_fn,
            df["sequence"].tolist(),
            device,
            batch_size,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        true = df[fc_col].to_numpy(dtype=np.float32)
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
            encoder_model,
            head,
            forward_fn,
            df["sequence_ref"].tolist(),
            device,
            batch_size,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        alt_pred = _predict_test_sequences(
            encoder_model,
            head,
            forward_fn,
            df["sequence_alt"].tolist(),
            device,
            batch_size,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        alt_col = f"{fc_col}_alt"
        if alt_col not in df.columns:
            alt_col = "K562_log2FC_alt"  # fallback for K562
        alt_true = df[alt_col].to_numpy(dtype=np.float32)
        metrics["snv_abs"] = {
            "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
            "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            "n": int(len(alt_true)),
        }
        delta_pred = alt_pred - ref_pred
        delta_col = f"delta_{fc_col}"
        if delta_col not in df.columns:
            delta_col = "delta_log2FC"
        delta_true = df[delta_col].to_numpy(dtype=np.float32)
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            "n": int(len(delta_true)),
        }

    # OOD: try cell-line-specific file first, fall back to K562
    ood_path = test_set_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        df = pd.read_csv(ood_path, sep="\t")
        if fc_col in df.columns:
            true = df[fc_col].to_numpy(dtype=np.float32)
        elif "K562_log2FC" in df.columns:
            true = df["K562_log2FC"].to_numpy(dtype=np.float32)
        else:
            true = None
        if true is not None:
            pred = _predict_test_sequences(
                encoder_model,
                head,
                forward_fn,
                df["sequence"].tolist(),
                device,
                batch_size,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )
            metrics["ood"] = {
                "pearson_r": _safe_corr(pred, true, pearsonr),
                "spearman_r": _safe_corr(pred, true, spearmanr),
                "mse": float(np.mean((pred - true) ** 2)),
                "n": int(len(true)),
            }

    return metrics


def save_test_predictions_s2(
    encoder_model,
    head,
    forward_fn,
    test_set_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 4,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    cell_line: str = "k562",
) -> None:
    """Save raw pred/true arrays for scatter plots + back up results."""
    import pandas as pd

    fc_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")

    def _pred(sequences: list[str]) -> np.ndarray:
        return _predict_test_sequences(
            encoder_model,
            head,
            forward_fn,
            sequences,
            device,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )

    arrays = {}
    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        df = pd.read_csv(in_path, sep="\t")
        arrays["in_dist_pred"] = _pred(df["sequence"].tolist())
        arrays["in_dist_true"] = df[fc_col].to_numpy(dtype=np.float32)

    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        df = pd.read_csv(snv_path, sep="\t")
        arrays["snv_ref_pred"] = _pred(df["sequence_ref"].tolist())
        arrays["snv_alt_pred"] = _pred(df["sequence_alt"].tolist())
        alt_col = f"{fc_col}_alt"
        if alt_col not in df.columns:
            alt_col = "K562_log2FC_alt"
        arrays["snv_alt_true"] = df[alt_col].to_numpy(dtype=np.float32)
        arrays["snv_delta_pred"] = arrays["snv_alt_pred"] - arrays["snv_ref_pred"]
        delta_col = f"delta_{fc_col}"
        if delta_col not in df.columns:
            delta_col = "delta_log2FC"
        arrays["snv_delta_true"] = df[delta_col].to_numpy(dtype=np.float32)

    # OOD: try cell-line-specific file first, fall back to K562
    ood_path = test_set_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        df = pd.read_csv(ood_path, sep="\t")
        if fc_col in df.columns:
            ood_true_col = fc_col
        elif "K562_log2FC" in df.columns:
            ood_true_col = "K562_log2FC"
        else:
            ood_true_col = None
        if ood_true_col is not None:
            arrays["ood_pred"] = _pred(df["sequence"].tolist())
            arrays["ood_true"] = df[ood_true_col].to_numpy(dtype=np.float32)

    pred_path = output_dir / "test_predictions.npz"
    np.savez_compressed(pred_path, **arrays)
    print(f"  Saved predictions: {pred_path} ({pred_path.stat().st_size / 1024:.0f} KB)")

    # Back up to permanent directory
    backup_dir = output_dir.parents[0]
    while backup_dir.name != "outputs" and backup_dir != backup_dir.parent:
        backup_dir = backup_dir.parent
    backup_base = backup_dir / "results_backup_DO_NOT_DELETE"
    backup_base.mkdir(parents=True, exist_ok=True)


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
        borzoi_cb = int(cfg.get("borzoi_center_bins", 0))
        borzoi_norm = bool(cfg.get("normalize_embeddings", False))
        if borzoi_cb > 0:
            print(f"Borzoi: center-bin pooling ({borzoi_cb} bins)", flush=True)
        if borzoi_norm:
            print("Borzoi: L2-normalizing embeddings", flush=True)

        def forward_fn(model, batch):
            return _forward_borzoi(model, batch, center_bins=borzoi_cb, normalize=borzoi_norm)

        should_unfreeze = _should_unfreeze_borzoi
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # ── LoRA adapters (Borzoi only) ─────────────────────────────────────────
    lora_adapters = None
    if model_name == "borzoi" and cfg.get("use_lora", False):
        lora_block_str = str(cfg.get("lora_blocks", "4,5,6,7"))
        lora_block_indices = [int(b) for b in lora_block_str.split(",") if b.strip()]
        lora_rank = int(cfg.get("lora_rank", 32))
        lora_adapters = inject_lora_adapters(encoder_model, lora_block_indices, rank=lora_rank)
        lora_params = sum(p.numel() for p in lora_adapters.parameters())
        print(
            f"LoRA adapters injected: rank={lora_rank}, blocks={lora_block_indices}, "
            f"params={lora_params:,}",
            flush=True,
        )

    encoder_model.to(device)
    encoder_model.eval()
    print(f"Encoder loaded in {time.time() - t0:.1f}s", flush=True)

    # ── Load S1 head ─────────────────────────────────────────────────────────
    hidden_dim = int(cfg["hidden_dim"])
    dropout = float(cfg["dropout"])
    head = MLPHead(embed_dim, hidden_dim, dropout)

    s1_dir = cfg["stage1_result_dir"]
    # Skip S1 head for Borzoi center-bin mode: the S1 head was trained on
    # all-bins mean-pool embeddings, which have a different distribution.
    skip_s1 = model_name == "borzoi" and int(cfg.get("borzoi_center_bins", 0)) > 0
    if s1_dir is not None and not skip_s1:
        s1_dir = Path(s1_dir).expanduser().resolve()
        ckpt = torch.load(s1_dir / "best_model.pt", map_location="cpu", weights_only=True)
        head.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Loaded S1 head from {s1_dir / 'best_model.pt'} (epoch {ckpt.get('epoch', '?')})",
            flush=True,
        )
    elif skip_s1:
        print(
            "[INFO] Skipping S1 head (center-bin pooling changes embedding distribution)",
            flush=True,
        )
    else:
        print("[WARN] No stage1_result_dir — starting head from random init", flush=True)

    head.to(device)

    # ── Selective unfreezing ─────────────────────────────────────────────────
    unfreeze_mode = cfg["unfreeze_mode"]
    encoder_params = []
    frozen_count = 0

    if lora_adapters is not None:
        # With LoRA: freeze entire encoder, only adapters are trainable
        for param in encoder_model.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        # Unfreeze adapter parameters
        for adapter in lora_adapters:
            for param in adapter.parameters():
                param.requires_grad = True
                encoder_params.append(param)
                frozen_count -= param.numel()  # don't double-count
    else:
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

    label = "LoRA adapters" if lora_adapters is not None else "encoder"
    print(
        f"Param groups — head: {n_head:,}  "
        f"{label} (trainable): {n_encoder:,}  "
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
    cell_line = cfg.get("cell_line", "k562")
    label_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    train_ds = K562Dataset(data_path=str(data_path), split="train", label_column=label_col)
    val_ds = K562Dataset(data_path=str(data_path), split="val", label_column=label_col)

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
    # AMP mode: "bfloat16" (Enformer), "fp16" (Borzoi — needs GradScaler), "off"
    amp_mode = str(cfg["amp_mode"]).lower()
    if amp_mode == "bfloat16" and device.type == "cuda":
        use_amp = True
        amp_dtype = torch.bfloat16
        scaler = None  # bfloat16 doesn't need GradScaler
    elif amp_mode == "fp16" and device.type == "cuda":
        use_amp = True
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    else:
        use_amp = False
        amp_dtype = torch.float32
        scaler = None
    print(f"Mixed precision: {amp_mode} (enabled={use_amp})", flush=True)
    import copy

    best_val_pearson = -1.0
    best_encoder_state = copy.deepcopy(encoder_model.state_dict())
    epochs_no_improve = 0
    early_stop_patience = int(cfg["early_stop_patience"])
    n_train_batches = len(train_loader)
    all_trainable = list(head.parameters()) + encoder_params
    head_warmup_epochs = int(cfg.get("head_warmup_epochs", 0))
    if head_warmup_epochs > 0:
        print(
            f"Head warmup: {head_warmup_epochs} epochs with encoder frozen (encoder_lr=0)",
            flush=True,
        )

    for epoch in range(int(cfg["epochs"])):
        t_epoch = time.time()

        # Head warmup: freeze encoder for the first N epochs so the head
        # learns to extract signal from stable (pre-trained) embeddings
        # before encoder updates start shifting the embedding space.
        is_warmup = epoch < head_warmup_epochs
        if is_warmup:
            optimizer.param_groups[1]["lr"] = 0.0
            for p in encoder_params:
                p.requires_grad = False
        elif epoch == head_warmup_epochs and head_warmup_epochs > 0:
            optimizer.param_groups[1]["lr"] = encoder_lr
            for p in encoder_params:
                p.requires_grad = True
            print(f"  [Warmup done] Unfreezing encoder (lr={encoder_lr})", flush=True)

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
            # Head + loss in fp32: bfloat16 rounds away the tiny inter-sample
            # embedding differences (cosine sim > 0.999 for Borzoi).
            emb = emb.float()
            pred = head(emb)
            loss = F.mse_loss(pred, labels) / grad_accum_steps

            if torch.isnan(loss):
                print(
                    f"  [WARN] NaN loss at batch {batch_idx + 1}, skipping",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            train_losses.append(loss.item() * grad_accum_steps)

            is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == n_train_batches
            if is_accum_step or is_last_batch:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_trainable, grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                # Head in fp32 (same as training)
                p_can = head(emb_can.float())
                p_rc = head(emb_rc.float())

                avg_pred = ((p_can + p_rc) / 2.0).cpu().float().numpy()
                val_preds.append(avg_pred)
                val_trues.append(labels.numpy())

        val_preds_arr = np.concatenate(val_preds)
        val_trues_arr = np.concatenate(val_trues)

        # Diagnostic: print val prediction stats to help debug constant outputs
        pred_std = float(np.std(val_preds_arr))
        if pred_std < 1e-6:
            print(
                f"  [DIAG] Val preds constant: mean={val_preds_arr.mean():.6f} "
                f"std={pred_std:.2e} min={val_preds_arr.min():.6f} "
                f"max={val_preds_arr.max():.6f}",
                flush=True,
            )

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
            if cfg.get("save_encoder", False):
                torch.save(
                    {
                        "model_state_dict": encoder_model.state_dict(),
                        "epoch": epoch,
                        "seed": seed,
                    },
                    output_dir / "best_encoder.pt",
                )
            # Always keep best encoder state in memory for test eval
            best_encoder_state = copy.deepcopy(encoder_model.state_dict())
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

    if (output_dir / "best_encoder.pt").exists():
        enc_ckpt = torch.load(
            output_dir / "best_encoder.pt", map_location="cpu", weights_only=False
        )
        encoder_model.load_state_dict(enc_ckpt["model_state_dict"])
    else:
        encoder_model.load_state_dict(best_encoder_state)
    encoder_model.to(device)

    print("\n[eval] Evaluating on test sets ...", flush=True)
    test_set_dir = data_path / "test_sets"
    max_test = int(cfg.get("max_test_sequences", 0))
    test_metrics = evaluate_all_test_sets(
        encoder_model,
        head,
        forward_fn,
        test_set_dir,
        device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        cell_line=cell_line,
        max_test_sequences=max_test,
    )

    # Save raw predictions for scatter plots
    print("[eval] Saving test predictions ...", flush=True)
    save_test_predictions_s2(
        encoder_model,
        head,
        forward_fn,
        test_set_dir,
        output_dir,
        device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        cell_line=cell_line,
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
