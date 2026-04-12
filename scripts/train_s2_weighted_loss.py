#!/usr/bin/env python
"""Stage 2 fine-tuning with WEIGHTED LOSS for negative sequences.

Identical to experiments/train_stage2_k562_hashfrag.py except:
  - Adds `neg_weight` config param (default 1.0 = standard MSE)
  - In the train_step loss, sequences with target < 0 get their
    squared error multiplied by `neg_weight`
  - Reports separate train loss for positives vs negatives each epoch
  - Also supports `loss_type` = "mse" | "focal" | "huber_neg"
      mse       — weighted MSE (neg * neg_weight)
      focal     — weight by |pred - target|^focal_gamma (error magnitude)
      huber_neg — Huber loss for negatives, MSE for positives

Usage::

    uv run --no-sync python scripts/train_s2_weighted_loss.py \
        --config-name stage2_k562_oracle \
        ++output_dir=outputs/oracle_neg_sweep/weighted_loss_2x/fold_0 \
        ++stage1_dir=outputs/ag_hashfrag_oracle_cached/oracle_0 \
        ++fold_id=0 \
        ++neg_weight=2.0 \
        ++loss_type=mse

    sbatch scripts/slurm/train_s2_weighted_loss_sweep.sh
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
import torch
import wandb
from alphagenome_ft import create_model_with_heads
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

CELL_LINE_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}

# ── MPRA flanks for 600 bp sequences ──────────────────────────────────────────
_FLANK_5_STR: str = MPRA_UPSTREAM[-200:]
_FLANK_3_STR: str = MPRA_DOWNSTREAM[:200]
_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}

_FLANK_5_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(_FLANK_5_STR):
    if _c in _MAPPING:
        _FLANK_5_ENC[_i, _MAPPING[_c]] = 1.0

_FLANK_3_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(_FLANK_3_STR):
    if _c in _MAPPING:
        _FLANK_3_ENC[_i, _MAPPING[_c]] = 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    return seed


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


def _merge(base, override):
    """Recursively merge *override* into *base*, returning a new dict."""
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


# ── Data collation ─────────────────────────────────────────────────────────────


def collate_stage2(
    batch: list[tuple],
    augment: bool = True,
    max_shift: int = 15,
) -> dict[str, np.ndarray]:
    """Collate K562Dataset items into 600 bp one-hot batches with RC + shift aug."""
    bsz = len(batch)
    x = np.zeros((bsz, 600, 4), dtype=np.float32)
    y = np.zeros(bsz, dtype=np.float32)
    for i, (seq_5ch, label) in enumerate(batch):
        # seq_5ch is (5, 200) tensor; take first 4 channels (ACGT) → (200, 4)
        core = np.asarray(seq_5ch)[:4, :].T
        full = np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)  # (600, 4)
        if augment:
            if np.random.rand() > 0.5:
                full = full[::-1, ::-1]  # RC
            if max_shift > 0 and np.random.rand() > 0.5:
                shift = np.random.randint(-max_shift, max_shift + 1)
                full = np.roll(full, shift, axis=0)
        x[i] = full
        y[i] = float(label.numpy()) if hasattr(label, "numpy") else float(label)
    return {
        "sequences": x,
        "targets": y,
        "organism_index": np.zeros(bsz, dtype=np.int32),
    }


# ── Test-set evaluation helpers ────────────────────────────────────────────────


def _seq_str_to_600bp(seq_str: str) -> np.ndarray:
    seq_str = seq_str.upper()
    target_len = 200
    if len(seq_str) < target_len:
        pad = target_len - len(seq_str)
        seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
    elif len(seq_str) > target_len:
        start = (len(seq_str) - target_len) // 2
        seq_str = seq_str[start : start + target_len]
    core = np.zeros((200, 4), dtype=np.float32)
    for i, c in enumerate(seq_str):
        if c in _MAPPING:
            core[i, _MAPPING[c]] = 1.0
    return np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)  # (600, 4)


def _predict_sequences(
    predict_step_fn,
    model_params,
    model_state,
    seqs_str: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """RC-averaged predictions on raw 200 bp strings via 600 bp context."""
    if not seqs_str:
        return np.array([], dtype=np.float32)
    x_fwd = np.stack([_seq_str_to_600bp(s) for s in seqs_str])
    x_rev = np.stack([_seq_str_to_600bp(s)[::-1, ::-1] for s in seqs_str])
    preds_fwd, preds_rev = [], []
    for i in range(0, len(x_fwd), batch_size):
        preds_fwd.append(
            np.array(
                predict_step_fn(model_params, model_state, jnp.array(x_fwd[i : i + batch_size]))
            ).reshape(-1)
        )
        preds_rev.append(
            np.array(
                predict_step_fn(model_params, model_state, jnp.array(x_rev[i : i + batch_size]))
            ).reshape(-1)
        )
    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def evaluate_all_test_sets(
    model,
    predict_step_fn,
    test_set_dir: Path,
    cell_line: str = "k562",
) -> dict[str, dict[str, float]]:
    """Evaluate on hashFrag in-dist / SNV / OOD test sets using full encoder + RC avg."""
    fc_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    params, state = model._params, model._state
    metrics: dict[str, dict[str, float]] = {}

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        df = pd.read_csv(in_path, sep="\t")
        pred = _predict_sequences(predict_step_fn, params, state, df["sequence"].tolist())
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
        ref_pred = _predict_sequences(predict_step_fn, params, state, df["sequence_ref"].tolist())
        alt_pred = _predict_sequences(predict_step_fn, params, state, df["sequence_alt"].tolist())
        alt_col = f"{fc_col}_alt"
        if alt_col not in df.columns:
            alt_col = "K562_log2FC_alt"
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

    ood_path = test_set_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        df = pd.read_csv(ood_path, sep="\t")
        pred = _predict_sequences(predict_step_fn, params, state, df["sequence"].tolist())
        ood_col = fc_col if fc_col in df.columns else "K562_log2FC"
        true = df[ood_col].to_numpy(dtype=np.float32)
        metrics["ood"] = {
            "pearson_r": _safe_corr(pred, true, pearsonr),
            "spearman_r": _safe_corr(pred, true, spearmanr),
            "mse": float(np.mean((pred - true) ** 2)),
            "n": int(len(true)),
        }

    return metrics


def _save_ag_s2_predictions(
    model, predict_step_fn, test_set_dir: Path, output_dir: Path, cell_line: str = "k562"
):
    """Save raw pred/true arrays for scatter plots."""
    fc_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    params, state = model._params, model._state
    arrays = {}

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        df = pd.read_csv(in_path, sep="\t")
        arrays["in_dist_pred"] = _predict_sequences(
            predict_step_fn, params, state, df["sequence"].tolist()
        )
        arrays["in_dist_true"] = df[fc_col].to_numpy(dtype=np.float32)

    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        df = pd.read_csv(snv_path, sep="\t")
        arrays["snv_ref_pred"] = _predict_sequences(
            predict_step_fn, params, state, df["sequence_ref"].tolist()
        )
        arrays["snv_alt_pred"] = _predict_sequences(
            predict_step_fn, params, state, df["sequence_alt"].tolist()
        )
        alt_col = f"{fc_col}_alt" if f"{fc_col}_alt" in df.columns else "K562_log2FC_alt"
        arrays["snv_alt_true"] = df[alt_col].to_numpy(dtype=np.float32)
        arrays["snv_delta_pred"] = arrays["snv_alt_pred"] - arrays["snv_ref_pred"]
        delta_col = f"delta_{fc_col}" if f"delta_{fc_col}" in df.columns else "delta_log2FC"
        arrays["snv_delta_true"] = df[delta_col].to_numpy(dtype=np.float32)

    ood_path = test_set_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        df = pd.read_csv(ood_path, sep="\t")
        arrays["ood_pred"] = _predict_sequences(
            predict_step_fn, params, state, df["sequence"].tolist()
        )
        ood_col = fc_col if fc_col in df.columns else "K562_log2FC"
        arrays["ood_true"] = df[ood_col].to_numpy(dtype=np.float32)

    pred_path = output_dir / "test_predictions.npz"
    np.savez_compressed(pred_path, **arrays)
    print(f"  Saved predictions: {pred_path} ({pred_path.stat().st_size / 1024:.0f} KB)")


# ── Loss functions ────────────────────────────────────────────────────────────


def weighted_mse_loss(
    pred: jnp.ndarray,
    targets: jnp.ndarray,
    neg_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MSE with neg_weight multiplier for negative targets.

    Returns (total_loss, pos_loss, neg_loss).
    """
    sq_err = (pred - targets) ** 2
    is_neg = targets < 0.0  # shape (B,)
    weights = jnp.where(is_neg, neg_weight, 1.0)
    weighted = sq_err * weights
    total = jnp.mean(weighted)

    # Separate reporting (not used in gradient, just logged)
    n_pos = jnp.sum(~is_neg) + 1e-8
    n_neg = jnp.sum(is_neg) + 1e-8
    pos_loss = jnp.sum(sq_err * (~is_neg)) / n_pos
    neg_loss = jnp.sum(sq_err * is_neg) / n_neg
    return total, pos_loss, neg_loss


def focal_mse_loss(
    pred: jnp.ndarray,
    targets: jnp.ndarray,
    focal_gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Focal-style MSE: weight each sample by |error|^gamma.

    Higher gamma => more focus on hard (high-error) samples.
    Returns (total_loss, pos_loss, neg_loss).
    """
    sq_err = (pred - targets) ** 2
    abs_err = jnp.abs(pred - targets)
    focal_weights = abs_err**focal_gamma  # shape (B,)
    # Normalise to keep gradient scale similar to plain MSE
    total = jnp.mean(sq_err * focal_weights)

    is_neg = targets < 0.0
    n_pos = jnp.sum(~is_neg) + 1e-8
    n_neg = jnp.sum(is_neg) + 1e-8
    pos_loss = jnp.sum(sq_err * (~is_neg)) / n_pos
    neg_loss = jnp.sum(sq_err * is_neg) / n_neg
    return total, pos_loss, neg_loss


def huber_neg_loss(
    pred: jnp.ndarray,
    targets: jnp.ndarray,
    neg_weight: float,
    delta: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Huber loss for negatives (robust to outliers) + weighted MSE for positives.

    For negatives:  huber(error, delta) * neg_weight
    For positives:  MSE (squared error)
    Returns (total_loss, pos_loss, neg_loss).
    """
    sq_err = (pred - targets) ** 2
    abs_err = jnp.abs(pred - targets)
    # Huber: 0.5*err^2 for |err|<=delta, delta*(|err|-0.5*delta) otherwise
    huber = jnp.where(
        abs_err <= delta,
        0.5 * sq_err,
        delta * (abs_err - 0.5 * delta),
    )

    is_neg = targets < 0.0
    per_sample = jnp.where(is_neg, huber * neg_weight, sq_err)
    total = jnp.mean(per_sample)

    n_pos = jnp.sum(~is_neg) + 1e-8
    n_neg = jnp.sum(is_neg) + 1e-8
    pos_loss = jnp.sum(sq_err * (~is_neg)) / n_pos
    neg_loss = jnp.sum(sq_err * is_neg) / n_neg
    return total, pos_loss, neg_loss


# ── Main ───────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="stage2_k562_oracle",
)
def main(cfg: DictConfig) -> None:
    """Stage 2 fine-tuning with weighted loss for negatives."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variant = str(cfg.get("variant", "s2c"))
    encoder_lr = float(cfg.encoder_lr)
    head_lr = float(cfg.head_lr)
    wd = float(cfg.weight_decay)
    dropout_rate = float(cfg.get("dropout_rate", 0.1))
    max_shift = int(cfg.get("max_shift", 15))

    # ── Weighted loss parameters ───────────────────────────────────────────────
    neg_weight = float(cfg.get("neg_weight", 1.0))
    loss_type = str(cfg.get("loss_type", "mse"))  # "mse" | "focal" | "huber_neg"
    focal_gamma = float(cfg.get("focal_gamma", 1.0))
    huber_delta = float(cfg.get("huber_delta", 1.0))

    print(
        f"Loss config: type={loss_type}, neg_weight={neg_weight}, "
        f"focal_gamma={focal_gamma}, huber_delta={huber_delta}",
        flush=True,
    )

    run_name = f"s2_wloss_{loss_type}_negw{neg_weight}_seed{used_seed}"
    wandb.init(
        project="albench-s2f",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["stage2", "alphagenome", "hashfrag", "k562", "weighted_loss", loss_type],
        mode=str(cfg.wandb_mode),
        job_type="stage2_weighted_loss",
    )

    # ── Head registration ──────────────────────────────────────────────────────
    arch_slug = str(cfg.head_arch).replace("-", "_")
    unique_head_name = f"{cfg.head_name}_{arch_slug}_v4"

    register_s2f_head(
        head_name=unique_head_name,
        arch=str(cfg.head_arch),
        task_mode="human",
        num_tracks=int(cfg.num_tracks),
        dropout_rate=dropout_rate,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    weights_path = str(Path(str(cfg.weights_path)).expanduser().resolve())
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {weights_path}")

    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=False,  # allow encoder gradients for Stage 2
    )
    reinit_head_params(model, unique_head_name, num_tokens=5, dim=1536, rng=used_seed)

    # ── Load Stage 1 best checkpoint ──────────────────────────────────────────
    stage1_dir = Path(str(cfg.stage1_dir)).expanduser().resolve()
    s1_ckpt_candidates = [
        stage1_dir / "best_model" / "checkpoint",
        stage1_dir / "best_model",
    ]
    s1_loaded = False
    for s1_ckpt_path in s1_ckpt_candidates:
        if s1_ckpt_path.exists():
            try:
                s1_mgr = ocp.CheckpointManager(str(s1_ckpt_path.resolve()))
                s1_params = s1_mgr.restore(
                    s1_mgr.latest_step(),
                    args=ocp.args.StandardRestore(model._params),
                )
                model._params = jax.device_put(_merge(model._params, s1_params))
                print(f"Loaded Stage 1 head checkpoint from {s1_ckpt_path}", flush=True)
                s1_loaded = True
                break
            except Exception:
                try:
                    checkpointer = ocp.StandardCheckpointer()
                    s1_params, _ = checkpointer.restore(s1_ckpt_path)
                    model._params = jax.device_put(_merge(model._params, s1_params))
                    print(f"Loaded Stage 1 head checkpoint from {s1_ckpt_path}", flush=True)
                    s1_loaded = True
                    break
                except Exception as e2:
                    print(f"[WARN] Failed to load {s1_ckpt_path}: {e2}", flush=True)
    if not s1_loaded:
        print(
            f"[WARN] Stage 1 checkpoint not found at {stage1_dir}/best_model; "
            "starting from random head initialisation.",
            flush=True,
        )

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    print(f"Total parameters: {param_count:,}", flush=True)

    # ── Per-group optimizer ────────────────────────────────────────────────────
    unfreeze_blocks = cfg.get("unfreeze_encoder_blocks", None)
    if unfreeze_blocks is not None:
        unfreeze_set = {f"downres_block_{b}" for b in unfreeze_blocks}
        print(f"Selective encoder freezing: only unfreezing {sorted(unfreeze_set)}", flush=True)
    else:
        unfreeze_set = None

    def _label_fn(path, _leaf):
        key_strs = [p.key if hasattr(p, "key") else str(p) for p in path]
        s = "/".join(str(k) for k in key_strs)
        if unique_head_name in s:
            return "head"
        elif "sequence_encoder" in s:
            if unfreeze_set is None:
                return "encoder"
            for block_name in unfreeze_set:
                if block_name in s:
                    return "encoder"
            return "frozen"
        return "frozen"

    param_labels = jax.tree_util.tree_map_with_path(_label_fn, model._params)
    optimizer = optax.multi_transform(
        {
            "head": optax.adamw(learning_rate=head_lr, weight_decay=wd),
            "encoder": optax.adamw(learning_rate=encoder_lr, weight_decay=wd),
            "frozen": optax.set_to_zero(),
        },
        param_labels,
    )
    opt_state = optimizer.init(model._params)

    label_counts: dict[str, int] = {"head": 0, "encoder": 0, "frozen": 0}
    for label, leaf in zip(
        jax.tree_util.tree_leaves(param_labels),
        jax.tree_util.tree_leaves(model._params),
    ):
        label_counts[label] = label_counts.get(label, 0) + leaf.size
    print(
        f"Param groups — head: {label_counts['head']:,}  "
        f"encoder (trainable): {label_counts['encoder']:,}  "
        f"frozen: {label_counts['frozen']:,}",
        flush=True,
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    use_dedicated_val = bool(cfg.get("use_dedicated_val", False))
    cell_line = str(cfg.get("cell_line", "k562"))
    label_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    print(f"Cell line: {cell_line}, label column: {label_col}", flush=True)

    if use_dedicated_val:
        train_subset = K562Dataset(
            data_path=str(cfg.k562_data_path), split="train", label_column=label_col
        )
        val_subset = K562Dataset(
            data_path=str(cfg.k562_data_path), split="val", label_column=label_col
        )
    else:
        n_folds = int(cfg.get("n_folds", 10))
        fold_id = int(cfg.get("fold_id", 0))
        ds_all = K562Dataset(
            data_path=str(cfg.k562_data_path), split="train", label_column=label_col
        )
        n_total = len(ds_all)
        perm = np.random.default_rng(seed=42).permutation(n_total)
        fold_size = n_total // n_folds
        val_start = fold_id * fold_size
        val_end = val_start + fold_size if fold_id < n_folds - 1 else n_total
        val_idx = perm[val_start:val_end]
        train_idx = np.concatenate([perm[:val_start], perm[val_end:]])
        train_subset = torch.utils.data.Subset(ds_all, train_idx.tolist())
        val_subset = torch.utils.data.Subset(ds_all, val_idx.tolist())

    N_train = len(train_subset)
    N_val = len(val_subset)
    print(
        f"Stage 2 — Train: {N_train:,} | Val: {N_val:,}"
        + (
            f" (dedicated val split)"
            if use_dedicated_val
            else f" (fold {cfg.get('fold_id', 0)}/{cfg.get('n_folds', 10)})"
        ),
        flush=True,
    )

    batch_size = int(cfg.batch_size)
    n_workers = int(cfg.get("num_workers", 8))

    def _collate_train(batch):
        return collate_stage2(batch, augment=True, max_shift=max_shift)

    def _collate_eval(batch):
        return collate_stage2(batch, augment=False, max_shift=0)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=_collate_train,
        pin_memory=True,
        persistent_workers=n_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=_collate_eval,
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )

    # ── JIT steps ──────────────────────────────────────────────────────────────
    # Capture loss params in closure (all are static Python scalars → no retrace)
    _neg_weight = neg_weight
    _loss_type = loss_type
    _focal_gamma = focal_gamma
    _huber_delta = huber_delta

    @jax.jit
    def train_step(params, current_opt_state, batch):
        def loss_func(p):
            preds = model._predict(
                p,
                model._state,
                batch["sequences"],
                batch["organism_index"],
                negative_strand_mask=jnp.zeros(len(batch["sequences"]), dtype=bool),
                strand_reindexing=None,
                requested_outputs=[unique_head_name],
            )[unique_head_name]
            pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
            targets = batch["targets"]

            if _loss_type == "focal":
                total, pos_l, neg_l = focal_mse_loss(pred, targets, _focal_gamma)
            elif _loss_type == "huber_neg":
                total, pos_l, neg_l = huber_neg_loss(pred, targets, _neg_weight, _huber_delta)
            else:  # default: weighted mse
                total, pos_l, neg_l = weighted_mse_loss(pred, targets, _neg_weight)

            return total, (pos_l, neg_l)

        (loss, aux), grads = jax.value_and_grad(loss_func, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss, aux

    @jax.jit
    def eval_step(params, sequences, organism_index):
        preds = model._predict(
            params,
            model._state,
            sequences,
            organism_index,
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
            requested_outputs=[unique_head_name],
        )[unique_head_name]
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
            requested_outputs=[unique_head_name],
        )[unique_head_name]

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_pearson = -1.0
    early_stop_patience = int(cfg.get("early_stop_patience", 10))
    epochs_no_improve = 0

    for epoch in range(int(cfg.epochs)):
        train_losses: list[float] = []
        pos_losses: list[float] = []
        neg_losses: list[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{int(cfg.epochs)}")
        for batch in pbar:
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            model._params, opt_state, loss, (pos_l, neg_l) = train_step(
                model._params, opt_state, batch_jax
            )
            loss_v = float(loss)
            train_losses.append(loss_v)
            pos_losses.append(float(pos_l))
            neg_losses.append(float(neg_l))
            pbar.set_postfix({"loss": f"{loss_v:.4f}"})

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_pos = float(np.mean(pos_losses)) if pos_losses else float("nan")
        avg_neg = float(np.mean(neg_losses)) if neg_losses else float("nan")

        # ── Validation ────────────────────────────────────────────────────────
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        for batch in val_loader:
            seqs = jnp.array(batch["sequences"])
            org_idx = jnp.array(batch["organism_index"])
            preds = eval_step(model._params, seqs, org_idx)
            y_pred_all.append(np.array(preds).reshape(-1))
            y_true_all.append(np.array(batch["targets"]).reshape(-1))

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        pear = _safe_corr(y_true, y_pred, pearsonr)
        spear = _safe_corr(y_true, y_pred, spearmanr)

        print(
            f"Epoch {epoch + 1}: train_loss={avg_train:.4f}  "
            f"pos_loss={avg_pos:.4f}  neg_loss={avg_neg:.4f}  "
            f"val_pearson={pear:.4f}  val_spearman={spear:.4f}",
            flush=True,
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": avg_train,
                "train/pos_loss": avg_pos,
                "train/neg_loss": avg_neg,
                "train/neg_pos_ratio": avg_neg / (avg_pos + 1e-8),
                "val/pearson_r": pear,
                "val/spearman_r": spear,
            }
        )

        if pear > best_val_pearson:
            best_val_pearson = pear
            epochs_no_improve = 0
            model.save_checkpoint(str(output_dir / "best_model"), save_full_model=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping: no improvement for {early_stop_patience} epochs "
                    f"(best val Pearson={best_val_pearson:.4f})",
                    flush=True,
                )
                break

        model.save_checkpoint(str(output_dir / "last_model"), save_full_model=True)

    # ── Post-training test evaluation ──────────────────────────────────────────
    print("\n[eval] Loading best checkpoint for test evaluation ...", flush=True)
    best_ckpt_path = output_dir / "best_model" / "checkpoint"
    if best_ckpt_path.exists():
        checkpointer = ocp.StandardCheckpointer()
        loaded_params, _ = checkpointer.restore(best_ckpt_path)
        model._params = jax.device_put(loaded_params)
    else:
        print("[eval] No best_model checkpoint — using final weights.", flush=True)

    cell_test_dir = Path(f"data/{cell_line}/test_sets")
    test_set_dir = (
        cell_test_dir if cell_test_dir.exists() else Path(str(cfg.k562_data_path)) / "test_sets"
    )
    test_metrics = evaluate_all_test_sets(model, predict_step, test_set_dir, cell_line=cell_line)

    print("[eval] Saving test predictions ...", flush=True)
    _save_ag_s2_predictions(model, predict_step, test_set_dir, output_dir, cell_line=cell_line)

    results = {
        "seed": used_seed,
        "variant": variant,
        "loss_type": loss_type,
        "neg_weight": neg_weight,
        "focal_gamma": focal_gamma,
        "huber_delta": huber_delta,
        "encoder_lr": encoder_lr,
        "head_lr": head_lr,
        "best_val_pearson": best_val_pearson,
        "head_arch": str(cfg.head_arch),
        "head_name": unique_head_name,
        "stage1_dir": str(stage1_dir),
        "test_metrics": test_metrics,
    }
    out_json = output_dir / "test_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Wrote {out_json}", flush=True)

    for test_set, m in test_metrics.items():
        wandb.log({f"test/{test_set}/pearson_r": m.get("pearson_r", 0.0)})
        print(
            f"[eval]   {test_set}: pearson_r={m.get('pearson_r', 0.0):.4f}  "
            f"spearman_r={m.get('spearman_r', 0.0):.4f}  mse={m.get('mse', 0.0):.4f}",
            flush=True,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
