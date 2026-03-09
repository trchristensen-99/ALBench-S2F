#!/usr/bin/env python
"""Stage 2: Fine-tune NTv3 650M encoder + MLP head on K562 hashFrag data.

Loads the best Stage 1 (head-only) checkpoint and selectively unfreezes the
last N transformer blocks for joint fine-tuning using per-group learning rates
via optax.multi_transform.

NTv3 architecture (650M):
  embed_layer → stem → conv_tower_blocks[0-7] → transformer_blocks[0-23]
  → deconv_tower_blocks[0-7] → lm_head

Selective unfreezing: only the last N transformer blocks are trainable
(analogous to AlphaGenome's downres_block selective unfreezing).

Usage::

    python experiments/train_ntv3_stage2.py \
        ++stage1_result_dir=outputs/foundation_grid_search/ntv3/lr0.001_wd1e-06_do0.1/seed_42 \
        ++output_dir=outputs/ntv3_k562_stage2/sweep_elr1e-4_uf4

Run via SLURM::

    sbatch scripts/slurm/ntv3_stage2_sweep.sh
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import nnx
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from data.k562 import K562Dataset
from data.k562_full import K562FullDataset

# ── JAX 0.9 / Flax 0.10 compat fix ──────────────────────────────────────────
# Flax 0.10.4's nnx.jit passes `abstracted_axes` to jax.jit, but JAX 0.9 removed it.
_orig_jax_jit = jax.jit


def _patched_jax_jit(fun, *args, **kwargs):
    kwargs.pop("abstracted_axes", None)
    return _orig_jax_jit(fun, *args, **kwargs)


jax.jit = _patched_jax_jit

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "output_dir": "outputs/ntv3_k562_stage2",
    "stage1_result_dir": None,  # path to best S1 grid search result dir
    "data_path": "data/k562",
    "seed": None,
    "epochs": 50,
    "batch_size": 64,
    "head_lr": 1e-3,
    "encoder_lr": 1e-4,
    "weight_decay": 1e-6,
    "hidden_dim": 512,
    "embed_dim": 1536,
    "dropout": 0.1,
    "early_stop_patience": 10,
    "max_shift": 15,
    "rc_aug": True,
    "unfreeze_blocks": "8,9,10,11",  # last 4 of 12 transformer blocks
    "use_flanks": True,  # 600bp with MPRA flanks (vs 200bp bare)
    "num_workers": 4,
    "use_bfloat16": False,  # f32 for training stability; bf16 OK for inference
    "grad_clip": 1.0,
    "model_variant": "pre",  # "pre" or "post" (post-trained, species-conditioned)
    "model_name": None,  # auto-set from model_variant if None
}

# ── MPRA flanks ──────────────────────────────────────────────────────────────
_MPRA_UPSTREAM = K562FullDataset.MPRA_UPSTREAM
_MPRA_DOWNSTREAM = K562FullDataset.MPRA_DOWNSTREAM
_FLANK_5_STR: str = _MPRA_UPSTREAM[-200:]
_FLANK_3_STR: str = _MPRA_DOWNSTREAM[:200]

_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
_RC_MAP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
NTV3_SEQ_DIVISOR = 128  # U-Net uses 7 downsamples → 2^7 = 128 for 650M (actually 2^8=256)


# ── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    return seed


def _safe_corr(y_pred: np.ndarray, y_true: np.ndarray, fn) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_pred, y_true)[0])


def _onehot_to_str(onehot_5ch: np.ndarray) -> str:
    """Convert (5, L) one-hot tensor to DNA string (uses first 4 channels)."""
    acgt = np.asarray(onehot_5ch)[:4, :]  # (4, L)
    bases = "ACGT"
    result = []
    for j in range(acgt.shape[1]):
        idx = np.argmax(acgt[:, j])
        if acgt[idx, j] > 0:
            result.append(bases[idx])
        else:
            result.append("N")
    return "".join(result)


def _rc_str(seq: str) -> str:
    """Reverse complement a DNA string."""
    return "".join(_RC_MAP.get(c, "N") for c in reversed(seq))


def _pad_to_divisible(seq: str, divisor: int) -> str:
    """N-pad sequence to the next multiple of divisor."""
    remainder = len(seq) % divisor
    if remainder == 0:
        return seq
    pad_total = divisor - remainder
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return "N" * pad_left + seq + "N" * pad_right


# ── Data collation ───────────────────────────────────────────────────────────
def collate_stage2(
    batch: list[tuple],
    augment: bool = True,
    max_shift: int = 15,
    use_flanks: bool = True,
) -> dict[str, np.ndarray | list[str]]:
    """Collate K562Dataset items into DNA strings with optional RC + shift aug.

    Returns dict with 'sequences' (list[str]) and 'targets' (np.ndarray).
    """
    sequences: list[str] = []
    targets = np.zeros(len(batch), dtype=np.float32)

    for i, (seq_5ch, label) in enumerate(batch):
        core_str = _onehot_to_str(seq_5ch)  # 200bp

        if use_flanks:
            full_str = _FLANK_5_STR + core_str + _FLANK_3_STR  # 600bp
        else:
            full_str = core_str  # 200bp

        if augment:
            # RC augmentation
            if np.random.rand() > 0.5:
                full_str = _rc_str(full_str)
            # Shift augmentation (circular roll via string slicing)
            if max_shift > 0 and np.random.rand() > 0.5:
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift != 0:
                    full_str = full_str[shift:] + full_str[:shift]

        sequences.append(full_str)
        targets[i] = float(label.numpy()) if hasattr(label, "numpy") else float(label)

    return {"sequences": sequences, "targets": targets}


# ── PyTorch MLP Head (for loading S1 checkpoint) ────────────────────────────
class _PyTorchMLPHead(torch.nn.Module):
    """Mirror of MLPHead from train_foundation_cached.py for checkpoint loading."""

    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── JAX MLP Head ────────────────────────────────────────────────────────────
class JaxMLPHead(nnx.Module):
    """LayerNorm → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(1)."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.ln = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.linear1 = nnx.Linear(embed_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        x = self.ln(x)
        x = nnx.relu(self.linear1(x))
        x = self.dropout1(x, deterministic=deterministic)
        x = nnx.relu(self.linear2(x))
        x = self.dropout2(x, deterministic=deterministic)
        x = self.linear3(x)
        return x.squeeze(-1)


def _load_s1_head_into_jax(
    s1_dir: Path,
    embed_dim: int,
    hidden_dim: int,
    dropout: float,
    rngs: nnx.Rngs,
) -> JaxMLPHead:
    """Load Stage 1 PyTorch head checkpoint and convert to JAX MLPHead."""
    ckpt_path = s1_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    pt_state = ckpt["model_state_dict"]

    jax_head = JaxMLPHead(embed_dim, hidden_dim, dropout, rngs=rngs)

    # Map PyTorch state_dict keys → JAX head attributes
    # PT: net.0.weight/bias (LayerNorm) → jax_head.ln
    jax_head.ln.scale.value = jnp.array(pt_state["net.0.weight"].numpy())
    jax_head.ln.bias.value = jnp.array(pt_state["net.0.bias"].numpy())
    # PT: net.1.weight/bias (Linear) → jax_head.linear1
    jax_head.linear1.kernel.value = jnp.array(pt_state["net.1.weight"].numpy().T)
    jax_head.linear1.bias.value = jnp.array(pt_state["net.1.bias"].numpy())
    # PT: net.4.weight/bias (Linear) → jax_head.linear2
    jax_head.linear2.kernel.value = jnp.array(pt_state["net.4.weight"].numpy().T)
    jax_head.linear2.bias.value = jnp.array(pt_state["net.4.bias"].numpy())
    # PT: net.7.weight/bias (Linear) → jax_head.linear3
    jax_head.linear3.kernel.value = jnp.array(pt_state["net.7.weight"].numpy().T)
    jax_head.linear3.bias.value = jnp.array(pt_state["net.7.bias"].numpy())

    print(f"Loaded Stage 1 head from {ckpt_path} (epoch {ckpt.get('epoch', '?')})", flush=True)
    return jax_head


# ── Label function for multi_transform ───────────────────────────────────────
def _make_label_fn(unfreeze_set: set[int], unfreeze_all_encoder: bool = False):
    """Create a label function for optax.multi_transform parameter grouping."""

    def _label_fn(path, _leaf):
        key_strs = [str(p.key if hasattr(p, "key") else p) for p in path]
        s = "/".join(key_strs)

        # Head params
        if "mlp_head" in s:
            return "head"

        # Full encoder unfreezing: everything that isn't the head
        if unfreeze_all_encoder:
            return "encoder"

        # Transformer blocks — check if index is in unfreeze set
        if "transformer_blocks" in s:
            for idx_k, k in enumerate(key_strs):
                if k == "transformer_blocks" and idx_k + 1 < len(key_strs):
                    block_idx_str = key_strs[idx_k + 1]
                    if block_idx_str.isdigit() and int(block_idx_str) in unfreeze_set:
                        return "encoder"
                    return "frozen"

        return "frozen"

    return _label_fn


# ── Forward pass ─────────────────────────────────────────────────────────────
def _forward(
    encoder,
    tokenizer,
    head,
    sequences_str,
    seq_divisor,
    deterministic=True,
    species_token=None,
):
    """Full forward: DNA strings → tokenize → encoder → mean-pool → head → scalar."""
    padded = [_pad_to_divisible(s, seq_divisor) for s in sequences_str]
    tokens = tokenizer.batch_np_tokenize(padded)
    tokens_jax = jnp.asarray(tokens)

    if species_token is not None:
        species_batch = jnp.tile(species_token, (tokens_jax.shape[0],))
        outs = encoder(tokens_jax, species_tokens=species_batch)
    else:
        outs = encoder(tokens_jax)
    embeddings = outs["embedding"]  # (B, T, D)

    # Mean-pool excluding padding tokens
    pad_mask = jnp.expand_dims(tokens_jax != tokenizer.pad_token_id, axis=-1)  # (B, T, 1)
    masked_emb = embeddings * pad_mask
    seq_lens = jnp.sum(pad_mask, axis=1)  # (B, 1)
    mean_emb = jnp.sum(masked_emb, axis=1) / jnp.maximum(seq_lens, 1)  # (B, D)

    return head(mean_emb, deterministic=deterministic)


# ── Test-set evaluation ──────────────────────────────────────────────────────
def _predict_test_sequences(
    encoder,
    tokenizer,
    head,
    seqs_str: list[str],
    seq_divisor: int,
    use_flanks: bool,
    batch_size: int = 64,
    species_token=None,
) -> np.ndarray:
    """RC-averaged predictions on raw 200bp test strings."""
    if not seqs_str:
        return np.array([], dtype=np.float32)

    # Build 600bp sequences with flanks
    full_fwd = []
    full_rev = []
    for s in seqs_str:
        s = s.upper()
        if len(s) < 200:
            pad = 200 - len(s)
            s = "N" * (pad // 2) + s + "N" * (pad - pad // 2)
        elif len(s) > 200:
            start = (len(s) - 200) // 2
            s = s[start : start + 200]
        if use_flanks:
            fwd = _FLANK_5_STR + s + _FLANK_3_STR
        else:
            fwd = s
        full_fwd.append(fwd)
        full_rev.append(_rc_str(fwd))

    preds_fwd, preds_rev = [], []
    for i in range(0, len(full_fwd), batch_size):
        batch_fwd = full_fwd[i : i + batch_size]
        batch_rev = full_rev[i : i + batch_size]
        p_fwd = _forward(
            encoder,
            tokenizer,
            head,
            batch_fwd,
            seq_divisor,
            deterministic=True,
            species_token=species_token,
        )
        p_rev = _forward(
            encoder,
            tokenizer,
            head,
            batch_rev,
            seq_divisor,
            deterministic=True,
            species_token=species_token,
        )
        preds_fwd.append(np.asarray(p_fwd).reshape(-1))
        preds_rev.append(np.asarray(p_rev).reshape(-1))

    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def evaluate_all_test_sets(
    encoder,
    tokenizer,
    head,
    test_set_dir: Path,
    seq_divisor: int,
    use_flanks: bool,
    batch_size: int = 64,
    species_token=None,
) -> dict[str, dict[str, float]]:
    """Evaluate on hashFrag in-dist / SNV / OOD test sets."""
    import pandas as pd

    metrics: dict[str, dict[str, float]] = {}

    _pred_kw = dict(species_token=species_token)

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        df = pd.read_csv(in_path, sep="\t")
        pred = _predict_test_sequences(
            encoder,
            tokenizer,
            head,
            df["sequence"].tolist(),
            seq_divisor,
            use_flanks,
            batch_size,
            **_pred_kw,
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
            encoder,
            tokenizer,
            head,
            df["sequence_ref"].tolist(),
            seq_divisor,
            use_flanks,
            batch_size,
            **_pred_kw,
        )
        alt_pred = _predict_test_sequences(
            encoder,
            tokenizer,
            head,
            df["sequence_alt"].tolist(),
            seq_divisor,
            use_flanks,
            batch_size,
            **_pred_kw,
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
            encoder,
            tokenizer,
            head,
            df["sequence"].tolist(),
            seq_divisor,
            use_flanks,
            batch_size,
            **_pred_kw,
        )
        true = df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["ood"] = {
            "pearson_r": _safe_corr(pred, true, pearsonr),
            "spearman_r": _safe_corr(pred, true, spearmanr),
            "mse": float(np.mean((pred - true) ** 2)),
            "n": int(len(true)),
        }

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────
def train(cfg: dict):
    from nucleotide_transformer_v3.pretrained import get_pretrained_ntv3_model

    used_seed = set_seed(int(cfg["seed"]) if cfg["seed"] is not None else None)
    print(f"Seed: {used_seed}", flush=True)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse unfreeze blocks
    uf_spec = str(cfg["unfreeze_blocks"]).strip()
    unfreeze_all_encoder = uf_spec.lower() == "all"
    if unfreeze_all_encoder:
        unfreeze_set = set(range(24))  # all transformer blocks
        print("Unfreezing FULL encoder (all transformer + conv tower)", flush=True)
    else:
        unfreeze_set = {int(b.strip()) for b in uf_spec.split(",")}
        print(f"Unfreezing transformer blocks: {sorted(unfreeze_set)}", flush=True)

    # ── Load NTv3 encoder ────────────────────────────────────────────────────
    model_variant = str(cfg.get("model_variant", "pre"))
    model_name = cfg.get("model_name")
    if model_name is None or model_name == "None":
        model_name = "NTv3_650M_post" if model_variant == "post" else "NTv3_650M_pre"

    variant_label = "post-trained" if model_variant == "post" else "pre-trained"
    print(f"Loading NTv3 650M {variant_label} encoder ({model_name}) ...", flush=True)
    t0 = time.time()

    species_token = None
    if model_variant == "post":
        from nucleotide_transformer_v3.pretrained import get_posttrained_ntv3_model

        encoder, tokenizer, ntv3_config = get_posttrained_ntv3_model(
            model_name=model_name,
            use_bfloat16=bool(cfg["use_bfloat16"]),
        )
        species_token = encoder.encode_species("human")  # shape (1,)
        print(f"  Species token (human): {species_token}", flush=True)
    else:
        encoder, tokenizer, ntv3_config = get_pretrained_ntv3_model(
            model_name=model_name,
            use_bfloat16=bool(cfg["use_bfloat16"]),
        )

    seq_divisor = 2**ntv3_config.num_downsamples
    print(f"Encoder loaded in {time.time() - t0:.1f}s  (seq_divisor={seq_divisor})", flush=True)

    # ── Load Stage 1 head ────────────────────────────────────────────────────
    rngs = nnx.Rngs(used_seed)
    embed_dim = int(cfg["embed_dim"])
    hidden_dim = int(cfg["hidden_dim"])
    dropout = float(cfg["dropout"])

    s1_dir = cfg["stage1_result_dir"]
    if s1_dir is not None:
        s1_dir = Path(s1_dir).expanduser().resolve()
        mlp_head = _load_s1_head_into_jax(s1_dir, embed_dim, hidden_dim, dropout, rngs)
    else:
        print("[WARN] No stage1_result_dir — starting head from random init", flush=True)
        mlp_head = JaxMLPHead(embed_dim, hidden_dim, dropout, rngs=rngs)

    # ── Combine into a single module for nnx.state ───────────────────────────
    class CombinedModel(nnx.Module):
        def __init__(self, encoder, mlp_head):
            self.encoder = encoder
            self.mlp_head = mlp_head

    combined = CombinedModel(encoder, mlp_head)

    # ── Per-group optimizer ──────────────────────────────────────────────────
    head_lr = float(cfg["head_lr"])
    encoder_lr = float(cfg["encoder_lr"])
    wd = float(cfg["weight_decay"])
    grad_clip = float(cfg["grad_clip"])

    label_fn = _make_label_fn(unfreeze_set, unfreeze_all_encoder)
    all_state = nnx.state(combined, nnx.Param)
    param_labels = jax.tree_util.tree_map_with_path(label_fn, all_state)

    optimizer = optax.multi_transform(
        {
            "head": optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.adamw(learning_rate=head_lr, weight_decay=wd),
            ),
            "encoder": optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.adamw(learning_rate=encoder_lr, weight_decay=wd),
            ),
            "frozen": optax.set_to_zero(),
        },
        param_labels,
    )
    opt_state = optimizer.init(all_state)

    # Print per-group param counts
    label_counts: dict[str, int] = {"head": 0, "encoder": 0, "frozen": 0}
    for label, leaf in zip(
        jax.tree_util.tree_leaves(param_labels),
        jax.tree_util.tree_leaves(all_state),
    ):
        label_counts[label] = label_counts.get(label, 0) + leaf.size
    total_params = sum(label_counts.values())
    print(
        f"Param groups — head: {label_counts['head']:,}  "
        f"encoder (trainable): {label_counts['encoder']:,}  "
        f"frozen: {label_counts['frozen']:,}  "
        f"total: {total_params:,}",
        flush=True,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_path = Path(cfg["data_path"])
    train_ds = K562Dataset(data_path=str(data_path), split="train")
    val_ds = K562Dataset(data_path=str(data_path), split="val")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)

    batch_size = int(cfg["batch_size"])
    n_workers = int(cfg["num_workers"])
    use_flanks = bool(cfg["use_flanks"])
    max_shift = int(cfg["max_shift"])

    def _collate_train(batch):
        return collate_stage2(
            batch, augment=cfg["rc_aug"], max_shift=max_shift, use_flanks=use_flanks
        )

    def _collate_eval(batch):
        return collate_stage2(batch, augment=False, max_shift=0, use_flanks=use_flanks)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=_collate_train,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=_collate_eval,
        pin_memory=False,
    )

    # ── Training functions ───────────────────────────────────────────────────
    # Tokenization is Python-side; JIT covers encoder + head + optimizer.
    pad_token_id = tokenizer.pad_token_id

    def _tokenize_batch(sequences_str):
        padded = [_pad_to_divisible(s, seq_divisor) for s in sequences_str]
        tokens = tokenizer.batch_np_tokenize(padded)
        return jnp.asarray(tokens)

    def _encoder_forward(m_encoder, tokens):
        """Encoder forward pass, handling species conditioning for post-trained."""
        if species_token is not None:
            species_batch = jnp.tile(species_token, (tokens.shape[0],))
            return m_encoder(tokens, species_tokens=species_batch)
        return m_encoder(tokens)

    @nnx.jit
    def jit_train_step(model, cur_opt_state, tokens, targets):
        """JIT train step: forward → backward → optimizer update (in-place)."""

        def _loss(m):
            outs = _encoder_forward(m.encoder, tokens)
            embeddings = outs["embedding"]
            mask = jnp.expand_dims(tokens != pad_token_id, axis=-1)
            masked = embeddings * mask
            lens = jnp.sum(mask, axis=1)
            pooled = jnp.sum(masked, axis=1) / jnp.maximum(lens, 1)
            preds = m.mlp_head(pooled, deterministic=False)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = nnx.value_and_grad(_loss)(model)
        # grads has same tree structure as nnx.state(model, nnx.Param)
        params = nnx.state(model, nnx.Param)
        updates, new_opt_state = optimizer.update(grads, cur_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        return loss, new_opt_state

    @nnx.jit
    def jit_eval_step(model, tokens):
        """JIT eval step: forward only, deterministic (no dropout)."""
        outs = _encoder_forward(model.encoder, tokens)
        embeddings = outs["embedding"]
        mask = jnp.expand_dims(tokens != pad_token_id, axis=-1)
        masked = embeddings * mask
        lens = jnp.sum(mask, axis=1)
        pooled = jnp.sum(masked, axis=1) / jnp.maximum(lens, 1)
        return model.mlp_head(pooled, deterministic=True)

    def train_step(batch):
        """Tokenize (Python) → JIT train step."""
        tokens_jax = _tokenize_batch(batch["sequences"])
        targets_jax = jnp.array(batch["targets"])
        nonlocal opt_state
        loss, opt_state = jit_train_step(combined, opt_state, tokens_jax, targets_jax)
        return float(loss)

    def eval_step(batch):
        """Tokenize (Python) → JIT eval step."""
        tokens_jax = _tokenize_batch(batch["sequences"])
        return np.asarray(jit_eval_step(combined, tokens_jax)).reshape(-1)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_pearson = -1.0
    epochs_no_improve = 0
    early_stop_patience = int(cfg["early_stop_patience"])
    best_encoder_state = None
    best_head_state = None

    for epoch in range(int(cfg["epochs"])):
        t_epoch = time.time()
        train_losses: list[float] = []

        for batch_idx, batch in enumerate(train_loader):
            loss_v = train_step(batch)
            train_losses.append(loss_v)
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch + 1} batch {batch_idx + 1}/{len(train_loader)} "
                    f"loss={loss_v:.4f}",
                    flush=True,
                )

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ── Validation ───────────────────────────────────────────────────────
        y_pred_all: list[np.ndarray] = []
        y_true_all: list[np.ndarray] = []
        for batch in val_loader:
            preds = eval_step(batch)
            y_pred_all.append(preds)
            y_true_all.append(batch["targets"])

        y_pred = np.concatenate(y_pred_all)
        y_true = np.concatenate(y_true_all)
        val_pearson = _safe_corr(y_pred, y_true, pearsonr)
        val_spearman = _safe_corr(y_pred, y_true, spearmanr)
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
            # Save best checkpoint (serialise NNX state as numpy)
            best_encoder_state = jax.tree_util.tree_map(
                lambda x: np.asarray(x), nnx.state(combined.encoder, nnx.Param)
            )
            best_head_state = jax.tree_util.tree_map(
                lambda x: np.asarray(x), nnx.state(combined.mlp_head, nnx.Param)
            )
            # Save head as PyTorch checkpoint for compatibility
            _save_head_pt(combined.mlp_head, epoch, used_seed, output_dir / "best_head.pt")
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
    if best_encoder_state is not None:
        best_enc_jax = jax.tree_util.tree_map(jnp.asarray, best_encoder_state)
        best_hd_jax = jax.tree_util.tree_map(jnp.asarray, best_head_state)
        nnx.update(combined.encoder, best_enc_jax)
        nnx.update(combined.mlp_head, best_hd_jax)

    print("\n[eval] Evaluating on test sets ...", flush=True)
    test_set_dir = data_path / "test_sets"
    test_metrics = evaluate_all_test_sets(
        combined.encoder,
        tokenizer,
        combined.mlp_head,
        test_set_dir,
        seq_divisor,
        use_flanks,
        batch_size=batch_size,
        species_token=species_token,
    )

    results = {
        "seed": used_seed,
        "encoder_lr": float(cfg["encoder_lr"]),
        "head_lr": float(cfg["head_lr"]),
        "unfreeze_blocks": sorted(unfreeze_set),
        "best_val_pearson": best_val_pearson,
        "stage1_result_dir": str(cfg["stage1_result_dir"]),
        "param_counts": label_counts,
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


def _save_head_pt(jax_head: JaxMLPHead, epoch: int, seed: int, path: Path):
    """Save JAX MLP head as a PyTorch state_dict for interop."""
    state_dict = {
        "net.0.weight": torch.from_numpy(np.asarray(jax_head.ln.scale.value)),
        "net.0.bias": torch.from_numpy(np.asarray(jax_head.ln.bias.value)),
        "net.1.weight": torch.from_numpy(np.asarray(jax_head.linear1.kernel.value).T),
        "net.1.bias": torch.from_numpy(np.asarray(jax_head.linear1.bias.value)),
        "net.4.weight": torch.from_numpy(np.asarray(jax_head.linear2.kernel.value).T),
        "net.4.bias": torch.from_numpy(np.asarray(jax_head.linear2.bias.value)),
        "net.7.weight": torch.from_numpy(np.asarray(jax_head.linear3.kernel.value).T),
        "net.7.bias": torch.from_numpy(np.asarray(jax_head.linear3.bias.value)),
    }
    torch.save({"model_state_dict": state_dict, "epoch": epoch, "seed": seed}, path)


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
