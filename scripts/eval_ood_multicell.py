#!/usr/bin/env python
"""Evaluate existing model checkpoints on cell-line-specific OOD test sets.

For each model checkpoint directory, loads the model, predicts on the OOD
sequences from data/{cell}/test_sets/test_ood_designed_{cell}.tsv, computes
pearson_r / spearman_r / mse, and patches the existing result.json or
test_metrics.json with the new "ood" key.

Supports all model types used in the ALBench-S2F pipeline:
  - malinois: PyTorch BassetBranched (raw sequences)
  - foundation_s1: PyTorch MLP head + foundation encoder (Enformer/NTv3/Borzoi)
  - enformer_s2 / borzoi_s2: PyTorch encoder + head (fine-tuned)
  - ntv3_s2: JAX NTv3 encoder + PyTorch-format head
  - ag_s1: JAX AlphaGenome (head-only, full encoder for inference)
  - ag_s2: JAX AlphaGenome (fine-tuned encoder + head)

Usage::

    python scripts/eval_ood_multicell.py \
        --cell-line hepg2 \
        --model-type malinois \
        --result-dirs outputs/malinois_hepg2_3seeds/seed_0/seed_0 \
                      outputs/malinois_hepg2_3seeds/seed_1/seed_1

    python scripts/eval_ood_multicell.py \
        --cell-line sknsh \
        --model-type foundation_s1 \
        --encoder-name enformer \
        --result-dirs outputs/enformer_sknsh_cached/seed_0/seed_0

    python scripts/eval_ood_multicell.py \
        --cell-line hepg2 \
        --model-type ag_s1 \
        --result-dirs outputs/ag_hashfrag_hepg2_cached/seed_0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

# ── Label column mapping ────────────────────────────────────────────────────
CELL_LINE_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}


def _safe_corr(pred: np.ndarray, target: np.ndarray, fn) -> float:
    if pred.size < 2 or np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def _compute_ood_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": _safe_corr(pred, true, pearsonr),
        "spearman_r": _safe_corr(pred, true, spearmanr),
        "mse": float(np.mean((pred - true) ** 2)),
        "n": int(len(true)),
    }


def _patch_result_json(result_path: Path, ood_metrics: dict) -> None:
    """Add or update the 'ood' key in a result.json or test_metrics.json."""
    _patch_result_json_key(result_path, "ood", ood_metrics)


def _patch_result_json_key(result_path: Path, key: str, metrics: dict) -> None:
    """Add or update an arbitrary key in a result.json or test_metrics.json."""
    data = json.loads(result_path.read_text())
    if "test_metrics" in data:
        data["test_metrics"][key] = metrics
    else:
        data[key] = metrics
    result_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  Patched {result_path} with {key} metrics", flush=True)


def evaluate_snv_from_predictions(
    ref_preds: np.ndarray,
    alt_preds: np.ndarray,
    snv_df: object,
    cell_line: str,
) -> dict[str, dict[str, float]]:
    """Compute SNV abs and delta metrics using cell-specific labels."""
    fc_col = CELL_LINE_LABEL_COLS[cell_line]
    alt_col = f"{fc_col}_alt"
    delta_col = f"delta_{fc_col}"
    metrics: dict[str, dict[str, float]] = {}

    if alt_col in snv_df.columns:
        alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
        mask = np.isfinite(alt_true)
        if mask.sum() > 0:
            metrics["snv_abs"] = _compute_ood_metrics(alt_preds[mask], alt_true[mask])
    else:
        print(f"    WARNING: {alt_col} not in SNV file")

    if delta_col in snv_df.columns:
        delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
        delta_pred = alt_preds - ref_preds
        mask = np.isfinite(delta_true)
        if mask.sum() > 0:
            metrics["snv_delta"] = _compute_ood_metrics(delta_pred[mask], delta_true[mask])
    else:
        print(f"    WARNING: {delta_col} not in SNV file")

    return metrics


# ── Malinois ────────────────────────────────────────────────────────────────
def eval_malinois(
    result_dir: Path,
    ood_sequences: list[str],
    ood_true: np.ndarray,
    cell_line: str,
) -> dict[str, float] | None:
    """Evaluate Malinois (BassetBranched) on OOD sequences."""
    import torch

    from data.utils import one_hot_encode
    from experiments.train_malinois_k562 import (
        MPRA_DOWNSTREAM,
        MPRA_UPSTREAM,
        _standardize_to_200bp,
    )
    from models.basset_branched import BassetBranched

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find checkpoint
    ckpt_path = result_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] No best_model.pt in {result_dir}", flush=True)
        return None

    # Build model
    model = BassetBranched(
        input_len=600,
        n_outputs=1,
        n_linear_layers=2,
        linear_channels=1000,
        n_branched_layers=1,
        branched_channels=250,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Pre-compute flanks
    left_flank = torch.from_numpy(
        one_hot_encode(MPRA_UPSTREAM[-200:], add_singleton_channel=False)
    ).float()
    right_flank = torch.from_numpy(
        one_hot_encode(MPRA_DOWNSTREAM[:200], add_singleton_channel=False)
    ).float()

    # Encode and predict
    preds = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(ood_sequences), batch_size):
            batch_seqs = ood_sequences[i : i + batch_size]
            tensors = []
            for seq in batch_seqs:
                seq = _standardize_to_200bp(seq)
                oh = torch.from_numpy(one_hot_encode(seq, add_singleton_channel=False)).float()
                padded = torch.cat([left_flank, oh, right_flank], dim=-1)
                tensors.append(padded)
            xb = torch.stack(tensors).to(device)
            out = model(xb).squeeze(-1)
            # RC averaging
            xb_rc = xb.flip(-1)[:, [3, 2, 1, 0], :]
            out_rc = model(xb_rc).squeeze(-1)
            preds.append(((out + out_rc) / 2.0).cpu().numpy().reshape(-1))

    pred = np.concatenate(preds)
    return _compute_ood_metrics(pred, ood_true)


# ── Foundation S1 (cached head + run encoder on the fly) ────────────────────
def eval_foundation_s1(
    result_dir: Path,
    ood_sequences: list[str],
    ood_true: np.ndarray,
    encoder_name: str,
    cell_line: str,
) -> dict[str, float] | None:
    """Evaluate Foundation S1 model (encoder + MLP head) on OOD sequences."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = result_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] No best_model.pt in {result_dir}", flush=True)
        return None

    # Read result.json to get embed_dim and hidden_dim
    result_json = result_dir / "result.json"
    if result_json.exists():
        rdata = json.loads(result_json.read_text())
        embed_dim = rdata.get("embed_dim", 768)
        hidden_dim = rdata.get("hidden_dim", 512)
    else:
        # Defaults per encoder
        embed_dims = {"enformer": 3072, "borzoi": 1536, "nt": 768, "ntv3_post": 1536}
        embed_dim = embed_dims.get(encoder_name, 768)
        hidden_dim = 512

    # Build MLP head (matches train_foundation_cached.py MLPHead)
    # MLPHead wraps nn.Sequential in self.net, so checkpoint keys are net.0.weight, etc.
    from experiments.train_foundation_cached import MLPHead

    head = MLPHead(embed_dim, hidden_dim, dropout=0.1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    head.load_state_dict(ckpt["model_state_dict"])
    head.to(device)
    head.eval()

    # Load encoder and compute embeddings
    if encoder_name == "enformer":
        pred = _predict_enformer_s1(head, ood_sequences, device)
    elif encoder_name in ("borzoi",):
        pred = _predict_borzoi_s1(head, ood_sequences, device)
    elif encoder_name in ("nt", "ntv3_post"):
        pred = _predict_ntv3_s1(head, ood_sequences, device, encoder_name)
    else:
        print(f"  [SKIP] Unknown encoder: {encoder_name}", flush=True)
        return None

    return _compute_ood_metrics(pred, ood_true)


def _predict_enformer_s1(head, sequences, device):
    """Run Enformer encoder + head on raw sequences."""
    import copy

    import torch
    import torch.nn.functional as F

    from data.utils import one_hot_encode
    from experiments.train_foundation_stage2 import (
        ENFORMER_SEQ_LEN,
        ENFORMER_TARGET_LEN,
        MPRA_DOWNSTREAM,
        MPRA_UPSTREAM,
    )

    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]

    from enformer_pytorch import Enformer

    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = {}
    encoder = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    encoder = copy.deepcopy(encoder)
    encoder.to(device)
    encoder.eval()

    preds_fwd, preds_rev = [], []
    batch_size = 2  # Enformer is memory-hungry

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        can_batch, rc_batch = [], []
        for seq in batch_seqs:
            seq = seq.upper()
            if len(seq) < 200:
                pad = 200 - len(seq)
                seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
            elif len(seq) > 200:
                start = (len(seq) - 200) // 2
                seq = seq[start : start + 200]
            full_seq = flank_5 + seq + flank_3  # 600bp
            oh = one_hot_encode(full_seq, add_singleton_channel=False)  # (4, 600)
            can_batch.append(oh.T)  # (600, 4) for channels-last
            rc_batch.append(oh[::-1, ::-1].T.copy())

        can_np = np.stack(can_batch).astype(np.float32)  # (B, 600, 4)
        rc_np = np.stack(rc_batch).astype(np.float32)

        # Pad to 196608
        pad_total = ENFORMER_SEQ_LEN - 600
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        with torch.no_grad():
            for batch_np, preds_list in [(can_np, preds_fwd), (rc_np, preds_rev)]:
                t = torch.from_numpy(batch_np).to(device)
                t = F.pad(t, (0, 0, pad_left, pad_right), value=0.0)
                emb = encoder(t, return_only_embeddings=True)  # (B, 896, 3072)
                center = ENFORMER_TARGET_LEN // 2
                center_emb = emb[:, center - 2 : center + 2, :].mean(dim=1)  # (B, 3072)
                p = head(center_emb).squeeze(-1)
                preds_list.append(p.cpu().numpy())

    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def _predict_borzoi_s1(head, sequences, device):
    """Run Borzoi encoder + head on raw sequences."""
    import torch
    import torch.nn.functional as F

    from data.utils import one_hot_encode
    from experiments.train_foundation_stage2 import (
        BORZOI_SEQ_LEN,
        MPRA_DOWNSTREAM,
        MPRA_UPSTREAM,
        _fix_borzoi_attention,
    )

    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]

    from borzoi_pytorch import Borzoi

    if not hasattr(Borzoi, "all_tied_weights_keys"):
        Borzoi.all_tied_weights_keys = {}
    encoder = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    _fix_borzoi_attention(encoder)
    encoder.to(device)
    encoder.eval()

    preds_fwd, preds_rev = [], []
    batch_size = 2

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        can_batch, rc_batch = [], []
        for seq in batch_seqs:
            seq = seq.upper()
            if len(seq) < 200:
                pad = 200 - len(seq)
                seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
            elif len(seq) > 200:
                start = (len(seq) - 200) // 2
                seq = seq[start : start + 200]
            full_seq = flank_5 + seq + flank_3
            oh = one_hot_encode(full_seq, add_singleton_channel=False)  # (4, 600)
            can_batch.append(oh)
            rc_batch.append(oh[::-1, ::-1].copy())

        pad_total = BORZOI_SEQ_LEN - 600

        with torch.no_grad():
            for batch_arr, preds_list in [(can_batch, preds_fwd), (rc_batch, preds_rev)]:
                t = torch.from_numpy(np.stack(batch_arr).astype(np.float32)).to(device)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                t = F.pad(t, (pad_left, pad_right), value=0.0)  # (B, 4, 196608)
                emb = encoder.get_embs_after_crop(t)  # (B, 1536, 6144)
                emb = emb.mean(dim=2)  # (B, 1536)
                p = head(emb).squeeze(-1)
                preds_list.append(p.cpu().numpy())

    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def _load_ntv3_encoder(variant: str = "post"):
    """Load NTv3 encoder, tokenizer, seq_divisor, and species_token."""
    if variant == "post":
        from nucleotide_transformer_v3.pretrained import get_posttrained_ntv3_model

        model_name = "NTv3_650M_post"
        encoder, tokenizer, ntv3_config = get_posttrained_ntv3_model(
            model_name=model_name,
            use_bfloat16=False,
        )
        species_token = encoder.encode_species("human")
    else:
        from nucleotide_transformer_v3.pretrained import get_pretrained_ntv3_model

        model_name = "NTv3_650M_pre"
        encoder, tokenizer, ntv3_config = get_pretrained_ntv3_model(
            model_name=model_name,
            use_bfloat16=False,
        )
        species_token = None

    seq_divisor = 2**ntv3_config.num_downsamples
    return encoder, tokenizer, seq_divisor, species_token


def _ntv3_encode_sequences(
    encoder,
    tokenizer,
    sequences_str,
    seq_divisor,
    species_token=None,
    batch_size=64,
) -> np.ndarray:
    """Encode DNA strings through NTv3 -> mean-pooled embeddings (N, D)."""
    import jax.numpy as jnp

    from experiments.train_ntv3_stage2 import _pad_to_divisible

    all_embs = []
    for i in range(0, len(sequences_str), batch_size):
        batch = sequences_str[i : i + batch_size]
        padded = [_pad_to_divisible(s, seq_divisor) for s in batch]
        tokens = tokenizer.batch_np_tokenize(padded)
        tokens_jax = jnp.asarray(tokens)

        if species_token is not None:
            sp = jnp.tile(species_token, (tokens_jax.shape[0],))
            outs = encoder(tokens_jax, species_tokens=sp)
        else:
            outs = encoder(tokens_jax)

        embeddings = outs["embedding"]  # (B, T, D)
        pad_mask = jnp.expand_dims(tokens_jax != tokenizer.pad_token_id, axis=-1)
        masked = embeddings * pad_mask
        lens = jnp.sum(pad_mask, axis=1)
        pooled = jnp.sum(masked, axis=1) / jnp.maximum(lens, 1)  # (B, D)
        all_embs.append(np.asarray(pooled))

    return np.concatenate(all_embs, axis=0)


def _predict_ntv3_s1(head, sequences, device, encoder_name):
    """Run NTv3 encoder + PyTorch head on raw sequences (RC-averaged)."""
    import torch

    from experiments.train_foundation_stage2 import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from experiments.train_ntv3_stage2 import _rc_str

    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]

    variant = "post" if "post" in encoder_name else "pre"
    encoder, tokenizer, seq_divisor, species_token = _load_ntv3_encoder(variant)

    # Build 600bp sequences
    can_seqs, rc_seqs = [], []
    for seq in sequences:
        seq = seq.upper()
        if len(seq) < 200:
            pad = 200 - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > 200:
            start = (len(seq) - 200) // 2
            seq = seq[start : start + 200]
        full_seq = flank_5 + seq + flank_3
        can_seqs.append(full_seq)
        rc_seqs.append(_rc_str(full_seq))

    can_embs = _ntv3_encode_sequences(
        encoder,
        tokenizer,
        can_seqs,
        seq_divisor,
        species_token=species_token,
        batch_size=64,
    )
    rc_embs = _ntv3_encode_sequences(
        encoder,
        tokenizer,
        rc_seqs,
        seq_divisor,
        species_token=species_token,
        batch_size=64,
    )

    with torch.no_grad():
        can_t = torch.from_numpy(can_embs.astype(np.float32)).to(device)
        rc_t = torch.from_numpy(rc_embs.astype(np.float32)).to(device)
        preds_can, preds_rc = [], []
        for i in range(0, len(can_t), 512):
            preds_can.append(head(can_t[i : i + 512]).squeeze(-1).cpu().numpy())
            preds_rc.append(head(rc_t[i : i + 512]).squeeze(-1).cpu().numpy())

    return (np.concatenate(preds_can) + np.concatenate(preds_rc)) / 2.0


# ── Enformer / Borzoi S2 ───────────────────────────────────────────────────
def eval_foundation_s2(
    result_dir: Path,
    ood_sequences: list[str],
    ood_true: np.ndarray,
    encoder_name: str,
    cell_line: str,
) -> dict[str, float] | None:
    """Evaluate Foundation S2 (fine-tuned encoder + head) on OOD sequences."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_path = result_dir / "best_model.pt"
    if not head_path.exists():
        print(f"  [SKIP] No best_model.pt in {result_dir}", flush=True)
        return None

    # Read config from result.json
    result_json = result_dir / "result.json"
    if result_json.exists():
        rdata = json.loads(result_json.read_text())
        cfg = rdata.get("config", {})
    else:
        cfg = {}

    hidden_dim = int(cfg.get("hidden_dim", 512))
    dropout = float(cfg.get("dropout", 0.1))

    from experiments.train_foundation_stage2 import (
        MLPHead,
        _forward_borzoi,
        _forward_enformer,
        _load_borzoi,
        _load_enformer,
        _predict_test_sequences,
    )

    # Load encoder
    if encoder_name == "enformer":
        encoder_model, embed_dim = _load_enformer()
        forward_fn = _forward_enformer
    elif encoder_name == "borzoi":
        encoder_model, embed_dim = _load_borzoi()
        center_bins = int(cfg.get("borzoi_center_bins", 0))

        def forward_fn(model, batch):
            return _forward_borzoi(model, batch, center_bins=center_bins)
    else:
        print(f"  [SKIP] Unknown S2 encoder: {encoder_name}", flush=True)
        return None

    # Load fine-tuned encoder weights if available
    encoder_path = result_dir / "best_encoder.pt"
    if encoder_path.exists():
        enc_ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
        encoder_model.load_state_dict(enc_ckpt["model_state_dict"])
        print(f"  Loaded fine-tuned encoder from {encoder_path}", flush=True)
    else:
        print(f"  [WARN] No best_encoder.pt, using pretrained encoder", flush=True)

    encoder_model.to(device)
    encoder_model.eval()

    # Load head
    head = MLPHead(embed_dim, hidden_dim, dropout)
    ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
    head.load_state_dict(ckpt["model_state_dict"])
    head.to(device)
    head.eval()

    # Predict using the existing function
    pred = _predict_test_sequences(
        encoder_model,
        head,
        forward_fn,
        ood_sequences,
        device,
        batch_size=int(cfg.get("batch_size", 4)),
    )

    return _compute_ood_metrics(pred, ood_true)


# ── NTv3 S2 ────────────────────────────────────────────────────────────────
def eval_ntv3_s2(
    result_dir: Path,
    ood_sequences: list[str],
    ood_true: np.ndarray,
    cell_line: str,
) -> dict[str, float] | None:
    """Evaluate NTv3 S2 (fine-tuned JAX encoder + head) on OOD sequences."""
    import pickle

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NTv3 S2 saves: best_head.pt (PyTorch), best_encoder_state.pkl (JAX NNX)
    # Also: best_model.pt may exist from the _save_head_pt function
    head_path = result_dir / "best_head.pt"
    if not head_path.exists():
        head_path = result_dir / "best_model.pt"
    if not head_path.exists():
        print(f"  [SKIP] No best_head.pt or best_model.pt in {result_dir}", flush=True)
        return None

    result_json = result_dir / "result.json"
    if result_json.exists():
        rdata = json.loads(result_json.read_text())
        cfg = rdata.get("config", {})
    else:
        cfg = {}

    embed_dim = int(cfg.get("embed_dim", 1536))
    hidden_dim = int(cfg.get("hidden_dim", 512))
    variant = cfg.get("model_variant", "post")
    use_flanks = str(cfg.get("use_flanks", "True")).lower() in ("true", "1", "yes")

    # Load NTv3 encoder
    encoder, tokenizer, seq_divisor, species_token = _load_ntv3_encoder(variant)

    # Load fine-tuned encoder weights if available
    enc_state_path = result_dir / "best_encoder_state.pkl"
    if enc_state_path.exists():
        import jax
        import jax.numpy as jnp
        from flax import nnx

        with open(enc_state_path, "rb") as f:
            best_encoder_state = pickle.load(f)
        best_enc_jax = jax.tree_util.tree_map(jnp.asarray, best_encoder_state)
        nnx.update(encoder, best_enc_jax)
        print(f"  Loaded fine-tuned NTv3 encoder from {enc_state_path}", flush=True)
    else:
        print("  [WARN] No best_encoder_state.pkl, using pretrained encoder", flush=True)

    # Build PyTorch head and load weights
    head_torch = torch.nn.Sequential(
        torch.nn.LayerNorm(embed_dim),
        torch.nn.Linear(embed_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_dim, 1),
    )
    ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
    # Saved checkpoint may have "net." prefix from MLPHead wrapper; strip it
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.removeprefix("net."): v for k, v in state_dict.items()}
    head_torch.load_state_dict(state_dict)
    head_torch.to(device)
    head_torch.eval()

    # Prepare sequences with flanks
    from experiments.train_foundation_stage2 import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from experiments.train_ntv3_stage2 import _rc_str

    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]

    can_seqs, rc_seqs = [], []
    for seq in ood_sequences:
        seq = seq.upper()
        if len(seq) < 200:
            pad = 200 - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > 200:
            start = (len(seq) - 200) // 2
            seq = seq[start : start + 200]
        if use_flanks:
            full_seq = flank_5 + seq + flank_3
        else:
            full_seq = seq
        can_seqs.append(full_seq)
        rc_seqs.append(_rc_str(full_seq))

    # Encode through NTv3
    can_embs = _ntv3_encode_sequences(
        encoder,
        tokenizer,
        can_seqs,
        seq_divisor,
        species_token=species_token,
        batch_size=64,
    )
    rc_embs = _ntv3_encode_sequences(
        encoder,
        tokenizer,
        rc_seqs,
        seq_divisor,
        species_token=species_token,
        batch_size=64,
    )

    # Predict through PyTorch head
    with torch.no_grad():
        can_t = torch.from_numpy(can_embs.astype(np.float32)).to(device)
        rc_t = torch.from_numpy(rc_embs.astype(np.float32)).to(device)
        preds_can, preds_rc = [], []
        for i in range(0, len(can_t), 512):
            preds_can.append(head_torch(can_t[i : i + 512]).squeeze(-1).cpu().numpy())
            preds_rc.append(head_torch(rc_t[i : i + 512]).squeeze(-1).cpu().numpy())

    pred = (np.concatenate(preds_can) + np.concatenate(preds_rc)) / 2.0
    return _compute_ood_metrics(pred, ood_true)


# ── AlphaGenome S1 / S2 ────────────────────────────────────────────────────
def eval_ag(
    result_dir: Path,
    ood_sequences: list[str],
    ood_true: np.ndarray,
    cell_line: str,
    stage: int = 1,
) -> dict[str, float] | None:
    """Evaluate AlphaGenome S1 or S2 on OOD sequences using full encoder."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head

    # Read existing result to get head config
    for json_name in ("test_metrics.json", "result.json"):
        json_path = result_dir / json_name
        if json_path.exists():
            rdata = json.loads(json_path.read_text())
            break
    else:
        print(f"  [SKIP] No result JSON in {result_dir}", flush=True)
        return None

    head_name = rdata.get("head_name", "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4")
    head_arch = rdata.get("head_arch", "boda-flatten-512-512")
    dropout_rate = 0.1

    # Register head
    register_s2f_head(
        head_name=head_name,
        arch=head_arch,
        task_mode="human",
        num_tracks=1,
        dropout_rate=dropout_rate,
    )

    # Load AG weights
    weights_path = (
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    )
    detach = stage == 1  # S1: head-only, S2: encoder fine-tuned

    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=detach,
    )

    # Load trained checkpoint (both S1 and S2 use save_full_model=True)
    ckpt_path = (result_dir / "best_model" / "checkpoint").resolve()
    if ckpt_path.exists():
        from collections.abc import Mapping

        def _merge(base, override):
            if not isinstance(override, Mapping):
                return override
            if not isinstance(base, Mapping):
                return override
            merged = dict(base)
            for k, v in override.items():
                if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                    merged[k] = _merge(merged[k], v)
                else:
                    merged[k] = v
            return merged

        checkpointer = ocp.StandardCheckpointer()
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(_merge(model._params, loaded_params))
        print(f"  Loaded AG checkpoint from {ckpt_path}", flush=True)
    else:
        print(f"  [SKIP] No orbax checkpoint in {result_dir}", flush=True)
        return None

    # Build predict function
    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    # Use the same prediction function from the AG training script
    from experiments.train_oracle_alphagenome_hashfrag_cached import _predict_sequences

    pred = _predict_sequences(predict_step, model._params, model._state, ood_sequences)

    return _compute_ood_metrics(pred, ood_true)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on OOD test sets")
    parser.add_argument(
        "--cell-line",
        required=True,
        choices=["k562", "hepg2", "sknsh"],
        help="Cell line to evaluate on",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=[
            "malinois",
            "foundation_s1",
            "enformer_s2",
            "borzoi_s2",
            "ntv3_s2",
            "ag_s1",
            "ag_s2",
        ],
        help="Model type determines loading strategy",
    )
    parser.add_argument(
        "--encoder-name",
        default=None,
        choices=["enformer", "borzoi", "nt", "ntv3_post"],
        help="Encoder name (required for foundation_s1)",
    )
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        required=True,
        help="Paths to result directories containing checkpoints",
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="Root directory for data/ (default: current dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, don't load models",
    )
    args = parser.parse_args()

    cell = args.cell_line
    fc_col = CELL_LINE_LABEL_COLS[cell]

    # Load OOD test set
    import pandas as pd

    data_root = Path(args.data_root)
    ood_path = data_root / "data" / cell / "test_sets" / f"test_ood_designed_{cell}.tsv"
    if not ood_path.exists():
        print(f"ERROR: OOD file not found: {ood_path}", file=sys.stderr)
        sys.exit(1)

    ood_df = pd.read_csv(ood_path, sep="\t")
    ood_sequences = ood_df["sequence"].astype(str).tolist()

    if fc_col in ood_df.columns:
        ood_true = ood_df[fc_col].to_numpy(dtype=np.float32)
    elif "K562_log2FC" in ood_df.columns:
        ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
    else:
        print(f"ERROR: No label column ({fc_col}) in {ood_path}", file=sys.stderr)
        sys.exit(1)

    print(
        f"OOD test set: {ood_path} ({len(ood_sequences)} sequences, label col={fc_col})",
        flush=True,
    )

    if args.dry_run:
        for rd in args.result_dirs:
            print(f"  [DRY RUN] Would evaluate: {rd}")
        return

    # Evaluate each result directory
    for rd_str in args.result_dirs:
        rd = Path(rd_str)
        if not rd.exists():
            print(f"  [SKIP] Directory not found: {rd}", flush=True)
            continue

        print(f"\n--- Evaluating {rd} ---", flush=True)

        model_type = args.model_type
        if model_type == "malinois":
            metrics = eval_malinois(rd, ood_sequences, ood_true, cell)
        elif model_type == "foundation_s1":
            if not args.encoder_name:
                print("  ERROR: --encoder-name required for foundation_s1")
                continue
            metrics = eval_foundation_s1(rd, ood_sequences, ood_true, args.encoder_name, cell)
        elif model_type in ("enformer_s2", "borzoi_s2"):
            enc = model_type.replace("_s2", "")
            metrics = eval_foundation_s2(rd, ood_sequences, ood_true, enc, cell)
        elif model_type == "ntv3_s2":
            metrics = eval_ntv3_s2(rd, ood_sequences, ood_true, cell)
        elif model_type == "ag_s1":
            metrics = eval_ag(rd, ood_sequences, ood_true, cell, stage=1)
        elif model_type == "ag_s2":
            metrics = eval_ag(rd, ood_sequences, ood_true, cell, stage=2)
        else:
            print(f"  [SKIP] Unknown model type: {model_type}")
            continue

        if metrics is None:
            continue

        print(
            f"  OOD: pearson_r={metrics['pearson_r']:.4f}  "
            f"spearman_r={metrics['spearman_r']:.4f}  "
            f"mse={metrics['mse']:.4f}  n={metrics['n']}",
            flush=True,
        )

        # Patch result JSON
        for json_name in ("result.json", "test_metrics.json"):
            json_path = rd / json_name
            if json_path.exists():
                _patch_result_json(json_path, metrics)
                break
        else:
            # No existing result file; create a minimal one
            out_path = rd / "ood_metrics.json"
            out_path.write_text(json.dumps({"ood": metrics}, indent=2) + "\n")
            print(f"  Created {out_path}", flush=True)


if __name__ == "__main__":
    main()
