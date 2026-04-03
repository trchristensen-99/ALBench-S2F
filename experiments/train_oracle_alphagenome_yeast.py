#!/usr/bin/env python
"""Train AlphaGenome oracle with frozen encoder on the full yeast S2F dataset.

This is the yeast analogue to ``train_oracle_alphagenome_full.py`` (which targets K562).
Supports the same three augmentation / caching modes:

* ``"full"``     — Full RC + shift augmentation; encoder runs on every batch.
* ``"no_shift"`` — Pre-compute and cache canonical + RC encoder embeddings once, then
                   train only the small head. ~20–50× faster per epoch; ideal for the
                   architecture search described here.
* ``"hybrid"``   — Cache canonical + RC. Each batch randomly chooses cache vs encoder.

Two-stage training (optional):
  Stage 1 — frozen encoder, head only (all modes above supported).
  Stage 2 — full encoder unfrozen, end-to-end fine-tuning at a lower LR.
  Enable with ``second_stage_lr`` (e.g. 1e-5) and ``second_stage_epochs`` (e.g. 50).
  Stage 2 always runs through the encoder with RC; shift magnitude is controlled
  by ``second_stage_max_shift`` (set to 0 to disable shift augmentation).
  Set ``second_stage_lr: null`` to skip Stage 2 entirely.

Yeast-specific details:
* 150bp core sequences padded to 384bp using plasmid flanks (54bp 5' + 89bp 3').
* Objective: 18-bin cross-entropy (KL) on discretised expression levels.
* Metric: Pearson r between predicted expected bin and ground-truth label.
* T=3 encoder tokens (384bp / 128bp stride).

Embedding cache size (yeast ~6M train seqs, T=3, D=1536, float16):
    ≈ 53 GB  — fits in HPC node memory but not GPU VRAM.
"""

from __future__ import annotations

import json
import os
import pickle
import signal
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import wandb
from alphagenome_ft import create_model_with_heads
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.yeast import YeastDataset
from evaluation.yeast_testsets import evaluate_yeast_test_subsets, load_yeast_test_subsets
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import (
    build_embedding_cache,
    build_head_only_predict_fn,
    build_head_only_train_fn,
    load_embedding_cache,
    lookup_cached_batch,
    reinit_head_params,
)

# ── Yeast plasmid flanks ───────────────────────────────────────────────────────
# 54bp 5' and 89bp 3' context from the Weissman lab yeast MPRA plasmid.
_YEAST_FLANK_5 = "GCTAGCGCCGATATCCTAACGAAGTCACTACTACGTACTGCCCTGCACGATAGC"  # 54bp
_YEAST_FLANK_3 = (
    "CCTGCAGCAGACGTCGACACGCGTCGTAAAGTGACGTTGTCCGAAACCCTT"
    "GCATTCGACACCAAACATTCTCTCAGTGCGTGCCCATGAAC"
)  # 92bp

_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_str(s: str) -> np.ndarray:
    out = np.zeros((len(s), 4), dtype=np.float32)
    for i, c in enumerate(s):
        if c in _MAPPING:
            out[i, _MAPPING[c]] = 1.0
    return out


_FLANK5_ENC = _encode_str(_YEAST_FLANK_5)  # (54, 4)
_FLANK3_ENC = _encode_str(_YEAST_FLANK_3)  # (89, 4)

_CANONICAL_TOTAL = _FLANK5_ENC.shape[0] + 150 + _FLANK3_ENC.shape[0]  # 296


# ── Collation helpers ──────────────────────────────────────────────────────────


def _build_yeast_window(core_4ch: np.ndarray, max_len: int, shift: int = 0) -> np.ndarray:
    """Concatenate flanks + core and slice the 384bp window with optional shift."""
    full = np.concatenate([_FLANK5_ENC, core_4ch, _FLANK3_ENC], axis=0)
    full_len = full.shape[0]
    # Buffer large enough to accommodate shift without clipping
    max_shift = max(110, abs(shift) + 10)
    buf_len = max_len + 2 * max_shift
    buf = np.zeros((buf_len, 4), dtype=np.float32)
    base = (buf_len - full_len) // 2
    buf[base : base + full_len, :] = full
    win_base = (buf_len - max_len) // 2
    start = np.clip(win_base + shift, 0, buf_len - max_len)
    return buf[start : start + max_len]


def _shift_existing_window(seq_4ch: np.ndarray, shift: int) -> np.ndarray:
    """Shift a fixed-length window with zero-fill (no wrap-around)."""
    out = np.zeros_like(seq_4ch)
    if shift == 0:
        return seq_4ch
    if shift > 0:
        out[shift:] = seq_4ch[:-shift]
    else:
        s = -shift
        out[:-s] = seq_4ch[s:]
    return out


def collate_yeast(
    batch: list[tuple],
    max_len: int = 384,
    augment: bool = False,
    max_shift: int = 110,
) -> dict[str, np.ndarray]:
    """Collate YeastDataset batches into AlphaGenome inputs."""
    B = len(batch)
    x = np.zeros((B, max_len, 4), dtype=np.float32)
    y = np.zeros(B, dtype=np.float32)
    for i, (seq, label) in enumerate(batch):
        seq_4ch = seq.numpy()[:4, :].T
        shift = int(np.random.randint(-max_shift, max_shift + 1)) if augment else 0
        if seq_4ch.shape[0] == max_len:
            w = _shift_existing_window(seq_4ch, shift) if augment else seq_4ch
        else:
            core = seq_4ch
            if core.shape[0] != 150:
                center = core.shape[0] // 2
                start = max(0, center - 75)
                core = core[start : start + 150]
                if core.shape[0] < 150:
                    pad = np.zeros((150, 4), dtype=np.float32)
                    pad[: core.shape[0], :] = core
                    core = pad
            w = _build_yeast_window(core, max_len, shift)
        if augment and np.random.rand() > 0.5:
            w = w[::-1, ::-1]
        x[i] = w
        y[i] = float(label.numpy()) if hasattr(label, "numpy") else float(label)
    return {
        "sequences": x,
        "targets": y,
        "organism_index": np.zeros(B, dtype=np.int32),
    }


def collate_yeast_indexed(
    batch_items: list[tuple],
    max_len: int = 384,
    augment: bool = False,
    max_shift: int = 110,
) -> dict[str, np.ndarray]:
    """Collate with original dataset indices (for cached embedding lookup)."""
    B = len(batch_items)
    indices = np.empty(B, dtype=np.int64)
    x = np.zeros((B, max_len, 4), dtype=np.float32)
    y = np.zeros(B, dtype=np.float32)
    for i, (orig_idx, seq, label) in enumerate(batch_items):
        indices[i] = orig_idx
        seq_4ch = seq.numpy()[:4, :].T
        shift = int(np.random.randint(-max_shift, max_shift + 1)) if augment else 0
        if seq_4ch.shape[0] == max_len:
            w = _shift_existing_window(seq_4ch, shift) if augment else seq_4ch
        else:
            core = seq_4ch
            if core.shape[0] != 150:
                center = core.shape[0] // 2
                start = max(0, center - 75)
                core = core[start : start + 150]
                if core.shape[0] < 150:
                    pad = np.zeros((150, 4), dtype=np.float32)
                    pad[: core.shape[0], :] = core
                    core = pad
            w = _build_yeast_window(core, max_len, shift)
        if augment and np.random.rand() > 0.5:
            w = w[::-1, ::-1]
        x[i] = w
        y[i] = float(label.numpy()) if hasattr(label, "numpy") else float(label)
    return {
        "indices": indices,
        "sequences": x,
        "targets": y,
        "organism_index": np.zeros(B, dtype=np.int32),
    }


def collate_val_indexed(batch_items: list[tuple]) -> dict[str, np.ndarray]:
    """Minimal collate for cache-based val: indices + targets only."""
    indices = np.array([orig_idx for orig_idx, _seq, _label in batch_items], dtype=np.int64)
    targets = np.array(
        [
            float(lbl.numpy()) if hasattr(lbl, "numpy") else float(lbl)
            for _idx, _seq, lbl in batch_items
        ],
        dtype=np.float32,
    )
    return {"indices": indices, "targets": targets}


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a Dataset to also return the original sample index."""

    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self._ds = dataset
        if isinstance(dataset, torch.utils.data.Subset):
            self._orig: np.ndarray | None = np.array(dataset.indices, dtype=np.int64)
        else:
            self._orig = None

    def __getitem__(self, i: int) -> tuple:
        orig = int(self._orig[i]) if self._orig is not None else i
        return (orig,) + tuple(self._ds[i])

    def __len__(self) -> int:
        return len(self._ds)


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)
    np.random.seed(seed)
    return seed


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


def _reverse_complement_batch(sequences: np.ndarray) -> np.ndarray:
    return sequences[:, ::-1, ::-1].copy()


def _merge_mappings(base, override):
    from collections.abc import Mapping

    if not isinstance(override, Mapping) or not isinstance(base, Mapping):
        return override
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
            merged[k] = _merge_mappings(merged[k], v)
        else:
            merged[k] = v
    return merged


def _create_model(
    *,
    weights_path: Path,
    head_name: str,
    detach_backbone: bool,
) -> object:
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=str(weights_path),
        use_encoder_output=True,
        detach_backbone=detach_backbone,
    )
    reinit_head_params(model, head_name, num_tokens=3, dim=1536)
    model.freeze_except_head(head_name)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="oracle_alphagenome_yeast_cached",
)
def main(cfg: DictConfig) -> None:
    """Train a frozen-encoder AlphaGenome oracle on yeast with optional embedding cache."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── SIGTERM handler for clean shutdown on SLURM preemption ────────────
    # Use --signal=B:TERM@120 in SBATCH to get 120s warning before kill.
    _sigterm_received = False

    def _sigterm_handler(signum, frame):
        nonlocal _sigterm_received
        _sigterm_received = True
        print(
            f"\n[SIGTERM] Received signal {signum}. Will save checkpoint at end of current epoch.",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── Resume detection (for SLURM preemption / --requeue) ───────────────
    _s2_progress_path = output_dir / "s2_progress.json"
    _resume_s1_complete = (
        (output_dir / "stage1_best" / "checkpoint").exists()
        or (output_dir / "last_model_s2" / "checkpoint").exists()
        or (output_dir / "best_model" / "checkpoint").exists()
    )
    _resume_s2 = _s2_progress_path.exists()
    _s2_resume_data: dict | None = None
    if _resume_s2:
        with _s2_progress_path.open() as _rpf:
            _s2_resume_data = json.load(_rpf)
        print(
            f"[RESUME] S2 progress found: epoch {_s2_resume_data['s2_epoch']}, "
            f"best_pearson={_s2_resume_data['best_s2_pearson']:.4f}"
        )
    elif _resume_s1_complete:
        print("[RESUME] Stage-1 complete (stage1_best found). Skipping S1 training.")

    aug_mode = str(cfg.get("aug_mode", "no_shift"))
    if aug_mode not in ("full", "no_shift", "hybrid"):
        raise ValueError(f"aug_mode must be 'full', 'no_shift', or 'hybrid'; got {aug_mode!r}")

    num_tracks = int(cfg.num_tracks)  # 18 for yeast (expression bins)
    dropout_rate = float(cfg.get("dropout_rate", 0.0))
    lr_schedule_type = str(cfg.get("lr_schedule", "plateau"))
    lr_plateau_patience = int(cfg.get("lr_plateau_patience", 5))
    lr_plateau_factor = float(cfg.get("lr_plateau_factor", 0.5))
    early_stop_patience = int(cfg.get("early_stop_patience", 15))
    _use_plateau = lr_schedule_type == "plateau"
    _use_cosine = lr_schedule_type == "cosine"

    # Two-stage config
    _s2_lr_raw = cfg.get("second_stage_lr", None)
    second_stage_lr = float(_s2_lr_raw) if _s2_lr_raw is not None else None
    second_stage_epochs = int(cfg.get("second_stage_epochs", 50))
    second_stage_early_stop = int(cfg.get("second_stage_early_stop_patience", 10))
    second_stage_unfreeze_mode = str(cfg.get("second_stage_unfreeze_mode", "encoder"))
    second_stage_max_shift = int(cfg.get("second_stage_max_shift", 43))
    if second_stage_max_shift < 0:
        raise ValueError("second_stage_max_shift must be >= 0")
    second_stage_full_unfreeze_epoch = cfg.get("second_stage_full_unfreeze_epoch", None)
    if second_stage_full_unfreeze_epoch is not None:
        second_stage_full_unfreeze_epoch = int(second_stage_full_unfreeze_epoch)
    _s2_max_seq_raw = cfg.get("second_stage_max_sequences", None)
    second_stage_max_sequences = int(_s2_max_seq_raw) if _s2_max_seq_raw is not None else None
    detach_backbone = bool(cfg.get("detach_backbone", True))

    # Walltime guard: stop training at max_wall_fraction of max_wall_seconds
    # to leave time for checkpoint saving and test evaluation.
    _wall_raw = cfg.get("max_wall_seconds", None)
    max_wall_seconds = int(_wall_raw) if _wall_raw is not None else None
    max_wall_fraction = float(cfg.get("max_wall_fraction", 0.80))
    _train_start_time = time.monotonic()

    _tags = ["oracle", "alphagenome", "yeast", str(cfg.head_arch), aug_mode]
    if second_stage_lr is not None:
        _tags.append("two_stage")
    wandb.init(
        project="albench-s2f",
        name=f"ag_yeast_{cfg.head_arch}_do{dropout_rate}_{aug_mode}_seed{used_seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=_tags,
        mode=str(cfg.wandb_mode),
        job_type="oracle_training",
    )

    arch_slug = str(cfg.head_arch).replace("-", "_")
    unique_head_name = f"ag_yeast_{arch_slug}_v4"

    _hidden_dims_raw = cfg.get("hidden_dims", None)
    _hidden_dims: list[int] | None = (
        [int(d) for d in _hidden_dims_raw] if _hidden_dims_raw is not None else None
    )
    _activation = str(cfg.get("activation", "relu"))
    register_s2f_head(
        head_name=unique_head_name,
        arch=str(cfg.head_arch),
        task_mode="yeast",
        num_tracks=num_tracks,
        dropout_rate=dropout_rate,
        hidden_dims=_hidden_dims,
        activation=_activation,
    )

    weights_path = Path(str(cfg.weights_path)).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {weights_path}")
    model = _create_model(
        weights_path=weights_path,
        head_name=unique_head_name,
        detach_backbone=detach_backbone,
    )

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    print(f"Total parameters: {n_params:,}")

    loss_fn = model.create_loss_fn_for_head(unique_head_name)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if _use_plateau or _use_cosine:
        optimizer = optax.inject_hyperparams(optax.adamw)(
            learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay)
        )
    elif cfg.get("gradients_clip") is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(cfg.gradients_clip)),
            optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay)),
        )
    else:
        optimizer = optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    opt_state = optimizer.init(model._params)

    # ── Datasets ──────────────────────────────────────────────────────────────
    context_mode = str(cfg.get("context_mode", "alphagenome384"))
    ds_train = YeastDataset(
        data_path=str(cfg.yeast_data_path),
        split="train",
        context_mode=context_mode,
    )
    val_dataset = YeastDataset(
        data_path=str(cfg.yeast_data_path),
        split="val",
        context_mode=context_mode,
    )
    train_dataset: torch.utils.data.Dataset = ds_train

    # ── Embedding cache (no_shift / hybrid) ───────────────────────────────────
    train_canonical = train_rc = val_canonical = val_rc = None
    head_predict_fn = head_train_fn = None
    max_seq_len = int(cfg.max_seq_len)  # 384

    if aug_mode != "full":
        cache_dir = Path(str(cfg.get("cache_dir", output_dir / "embedding_cache")))
        cache_dir = cache_dir.expanduser().resolve()
        # Build cache on the FULL ds_train (not subset) so indices stay valid.
        build_embedding_cache(
            model,
            ds_train,
            cache_dir,
            "train",
            max_seq_len=max_seq_len,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
        )
        build_embedding_cache(
            model,
            val_dataset,
            cache_dir,
            "val",
            max_seq_len=max_seq_len,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
        )
        train_canonical, train_rc = load_embedding_cache(cache_dir, "train", mmap_mode=None)
        val_canonical, val_rc = load_embedding_cache(cache_dir, "val")

        # Auto-detect limited cache: if cache has fewer entries than dataset,
        # truncate dataset to match (assumes cache covers first N sequences).
        cache_size = len(train_canonical)
        if cache_size < len(ds_train):
            print(
                f"[cache] Cache has {cache_size:,} entries but dataset has {len(ds_train):,}."
                f" Limiting training to first {cache_size:,} sequences."
            )
            ds_train = torch.utils.data.Subset(ds_train, range(cache_size))
            train_dataset = ds_train

        head_predict_fn = build_head_only_predict_fn(model, unique_head_name)
        # Verify head shape
        _dummy = jnp.zeros((2, 3, 1536), dtype=jnp.float32)  # T=3 for yeast
        _dummy_org = jnp.zeros((2,), dtype=jnp.int32)
        head_predict_fn(model._params, _dummy, _dummy_org)
        head_train_fn = (
            build_head_only_train_fn(model, unique_head_name) if dropout_rate > 0 else None
        )

    # ── JIT-compiled steps ────────────────────────────────────────────────────
    @jax.jit
    def train_step(params, opt_s, batch):
        def loss_func(p):
            preds = model._predict(
                p,
                model._state,
                batch["sequences"],
                batch["organism_index"],
                negative_strand_mask=jnp.zeros(len(batch["sequences"]), dtype=bool),
                strand_reindexing=None,
                is_training=True,
            )[unique_head_name]
            return loss_fn(preds, batch)["loss"]

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, nxt = optimizer.update(grads, opt_s, params)
        return optax.apply_updates(params, updates), nxt, loss

    @jax.jit
    def eval_step(params, batch):
        preds = model._predict(
            params,
            model._state,
            batch["sequences"],
            batch["organism_index"],
            negative_strand_mask=jnp.zeros(len(batch["sequences"]), dtype=bool),
            strand_reindexing=None,
        )[unique_head_name]
        return preds

    if head_predict_fn is not None:

        @jax.jit
        def cached_train_step(params, opt_s, rng, encoder_output, targets, org_idx):
            def loss_func(p):
                if head_train_fn is not None:
                    preds = head_train_fn(p, rng, encoder_output, org_idx)
                else:
                    preds = head_predict_fn(p, encoder_output, org_idx)
                # Yeast: cross-entropy (soft targets = one-hot of rounded bin)
                bins = jnp.round(jnp.clip(targets, 0.0, 17.0)).astype(jnp.int32)
                target_probs = jax.nn.one_hot(bins, num_tracks)
                log_probs = jax.nn.log_softmax(preds, axis=-1)
                return -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))

            loss, grads = jax.value_and_grad(loss_func)(params)
            updates, nxt = optimizer.update(grads, opt_s, params)
            return optax.apply_updates(params, updates), nxt, loss

        @jax.jit
        def cached_eval_step(params, encoder_output, org_idx):
            return head_predict_fn(params, encoder_output, org_idx)

    # ── Data loaders ──────────────────────────────────────────────────────────
    n_workers = int(cfg.num_workers)

    if aug_mode == "full":

        def collate_train(b):
            return collate_yeast(b, max_seq_len, augment=True)

        def collate_eval(b):
            return collate_yeast(b, max_seq_len, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_train,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
        val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_eval,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
    elif aug_mode == "no_shift":
        # Cached mode: only need indices + targets for training (embedding looked up by index).
        # No need to reconstruct sequences — avoids dependency on hardcoded flanks.
        train_loader = DataLoader(
            IndexedDataset(train_dataset),
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_val_indexed,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
        val_loader = DataLoader(
            IndexedDataset(val_dataset),
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_val_indexed,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_pearson = -1.0
    best_val_spearman = -1.0
    epochs_no_improve = 0
    rng = jax.random.PRNGKey(used_seed)
    current_lr = float(cfg.lr)
    eval_use_reverse_complement = bool(cfg.get("eval_use_reverse_complement", True))
    lr_plateau_best = -float("inf")
    lr_plateau_counter = 0
    _min_lr = float(cfg.lr) * 0.01

    _s1_epochs = 0 if (_resume_s1_complete or _resume_s2) else int(cfg.epochs)
    if _s1_epochs == 0:
        print("[RESUME] Skipping Stage-1 training loop.")
    for epoch in range(_s1_epochs):
        # ── Cosine LR update (applied before each epoch) ──────────────────────
        if _use_cosine:
            max_epochs = max(int(cfg.epochs) - 1, 1)
            progress = epoch / max_epochs
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            current_lr = float(cfg.lr) * (0.01 + 0.99 * cosine_factor)
            opt_state.hyperparams["learning_rate"] = np.float32(current_lr)

        train_losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{int(cfg.epochs)}")

        if aug_mode == "full":
            for batch in pbar:
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}
                model._params, opt_state, loss = train_step(model._params, opt_state, batch_jax)
                train_losses.append(float(loss))
                pbar.set_postfix({"loss": f"{float(loss):.4f}"})

        elif aug_mode == "no_shift":
            for batch in pbar:
                idx = batch["indices"]
                tgt = jnp.array(batch["targets"])
                org = jnp.zeros(len(idx), dtype=jnp.int32)
                # Canonical pass
                rng, srng = jax.random.split(rng)
                emb_c = jnp.array(train_canonical[idx].astype(np.float32))
                model._params, opt_state, loss = cached_train_step(
                    model._params, opt_state, srng, emb_c, tgt, org
                )
                train_losses.append(float(loss))
                # RC pass (same labels)
                rng, srng = jax.random.split(rng)
                emb_r = jnp.array(train_rc[idx].astype(np.float32))
                model._params, opt_state, loss = cached_train_step(
                    model._params, opt_state, srng, emb_r, tgt, org
                )
                train_losses.append(float(loss))
                pbar.set_postfix({"loss": f"{float(loss):.4f}"})

        else:  # hybrid
            for batch in pbar:
                if np.random.rand() > 0.5:
                    idx = batch["indices"]
                    emb = lookup_cached_batch(idx, train_canonical, train_rc)
                    org = jnp.zeros(len(idx), dtype=jnp.int32)
                    rng, srng = jax.random.split(rng)
                    model._params, opt_state, loss = cached_train_step(
                        model._params,
                        opt_state,
                        srng,
                        jnp.array(emb),
                        jnp.array(batch["targets"]),
                        org,
                    )
                else:
                    batch_jax = {k: jnp.array(v) for k, v in batch.items() if k != "indices"}
                    model._params, opt_state, loss = train_step(model._params, opt_state, batch_jax)
                train_losses.append(float(loss))
                pbar.set_postfix({"loss": f"{float(loss):.4f}"})

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")

        # ── Validation ────────────────────────────────────────────────────────
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []

        if aug_mode == "full":
            for batch in val_loader:
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}
                logits_fwd = eval_step(model._params, batch_jax)
                probs_fwd = np.array(jax.nn.softmax(logits_fwd, axis=-1))
                if eval_use_reverse_complement:
                    batch_rc = dict(batch)
                    batch_rc["sequences"] = _reverse_complement_batch(
                        np.asarray(batch["sequences"])
                    )
                    batch_rc_jax = {k: jnp.array(v) for k, v in batch_rc.items()}
                    logits_rev = eval_step(model._params, batch_rc_jax)
                    probs_rev = np.array(jax.nn.softmax(logits_rev, axis=-1))
                    probs = (probs_fwd + probs_rev) / 2.0
                else:
                    probs = probs_fwd
                pred_expr = np.sum(probs * np.arange(num_tracks, dtype=np.float32), axis=-1)
                y_pred_all.append(pred_expr)
                y_true_all.append(np.array(batch["targets"]).reshape(-1))
        else:
            for batch in val_loader:
                idx = batch["indices"]
                emb = jnp.array(val_canonical[idx].astype(np.float32))
                org = jnp.zeros(len(idx), dtype=jnp.int32)
                logits_fwd = cached_eval_step(model._params, emb, org)
                probs_fwd = np.array(jax.nn.softmax(logits_fwd, axis=-1))
                if eval_use_reverse_complement and val_rc is not None:
                    emb_rev = jnp.array(val_rc[idx].astype(np.float32))
                    logits_rev = cached_eval_step(model._params, emb_rev, org)
                    probs_rev = np.array(jax.nn.softmax(logits_rev, axis=-1))
                    probs = (probs_fwd + probs_rev) / 2.0
                else:
                    probs = probs_fwd
                pred_expr = np.sum(probs * np.arange(num_tracks, dtype=np.float32), axis=-1)
                y_pred_all.append(pred_expr)
                y_true_all.append(np.array(batch["targets"]).reshape(-1))

        y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
        y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
        pear = _safe_corr(y_true, y_pred, pearsonr)
        spear = _safe_corr(y_true, y_pred, spearmanr)

        # ── Plateau LR ────────────────────────────────────────────────────────
        if _use_plateau:
            if pear > lr_plateau_best:
                lr_plateau_best = pear
                lr_plateau_counter = 0
            else:
                lr_plateau_counter += 1
                if lr_plateau_counter >= lr_plateau_patience:
                    current_lr = max(current_lr * lr_plateau_factor, _min_lr)
                    opt_state.hyperparams["learning_rate"] = np.float32(current_lr)
                    lr_plateau_counter = 0
                    print(f"  [LR plateau] → {current_lr:.2e}")

        print(f"Epoch {epoch + 1}  val/pearson_r={pear:.4f}  val/spearman_r={spear:.4f}")
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": avg_train,
                "val/pearson_r": pear,
                "val/spearman_r": spear,
                "lr": current_lr,
            }
        )

        if pear > best_val_pearson:
            best_val_pearson = pear
            best_val_spearman = spear
            epochs_no_improve = 0
            try:
                model.save_checkpoint(str(output_dir / "best_model"), save_full_model=False)
            except Exception as e:
                print(f"  [WARN] best_model save failed: {e}")
        else:
            epochs_no_improve += 1

        try:
            model.save_checkpoint(str(output_dir / "last_model"), save_full_model=False)
        except Exception as e:
            print(f"  [WARN] last_model save failed: {e}")

        if _sigterm_received:
            print(
                f"[SIGTERM] Clean shutdown after S1 epoch {epoch + 1}. Exiting.",
                flush=True,
            )
            break

        if epochs_no_improve >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)"
            )
            break

        if max_wall_seconds is not None:
            elapsed = time.monotonic() - _train_start_time
            if elapsed >= max_wall_seconds * max_wall_fraction:
                print(
                    f"Walltime guard: {elapsed / 3600:.1f}h elapsed "
                    f"(>= {max_wall_fraction * 100:.0f}% of {max_wall_seconds / 3600:.1f}h). "
                    f"Stopping S1 to leave time for eval."
                )
                break

    # ── Stage 2: unfreeze encoder, end-to-end fine-tuning ────────────────────
    if second_stage_lr is not None:
        print(
            f"\n=== Stage 2: unfreezing encoder, lr={second_stage_lr:.1e}, "
            f"epochs={second_stage_epochs} ==="
        )
        # Load best Stage-1 checkpoint before unfreezing
        _pretrained_head_dir = cfg.get("pretrained_head_dir", None)
        if _pretrained_head_dir:
            best_s1 = Path(str(_pretrained_head_dir)).expanduser().resolve()
        else:
            best_s1 = output_dir / "best_model"

        if best_s1.exists():
            import orbax.checkpoint as ocp

            ckpt_path = (best_s1 / "checkpoint").resolve()
            if ckpt_path.exists():
                checkpointer = ocp.StandardCheckpointer()
                loaded_params, _ = checkpointer.restore(ckpt_path)
                model._params = jax.device_put(_merge_mappings(model._params, loaded_params))
                print("  Loaded best Stage-1 checkpoint.")
            else:
                print(f"  WARNING: checkpoint directory not found at {ckpt_path}")

        # Save full model (encoder + head) — skip if resuming (best_model already exists)
        if not _resume_s1_complete:
            model.save_checkpoint(str(output_dir / "stage1_best"), save_full_model=True)

        if detach_backbone:
            model_s2 = _create_model(
                weights_path=weights_path,
                head_name=unique_head_name,
                detach_backbone=False,
            )
            model_s2._params = jax.device_put(_merge_mappings(model_s2._params, model._params))
            model_s2._state = model._state
            model = model_s2
            print("  Recreated model with detach_backbone=False for Stage 2.")

        # Clean up stage1_best to free ~870 MB disk (no longer needed after model
        # params are loaded into memory for S2).
        _s1_best_dir = output_dir / "stage1_best"
        if _s1_best_dir.exists():
            import shutil

            shutil.rmtree(_s1_best_dir, ignore_errors=True)
            print("  Cleaned up stage1_best checkpoint (no longer needed).")

        if second_stage_unfreeze_mode not in ("encoder", "backbone", "gradual"):
            raise ValueError(
                "second_stage_unfreeze_mode must be one of: encoder, backbone, gradual"
            )

        model.freeze_except_head(unique_head_name)
        if second_stage_unfreeze_mode == "encoder":
            model.unfreeze_parameters(unfreeze_prefixes=["sequence_encoder"])
        elif second_stage_unfreeze_mode == "backbone":
            model.unfreeze_parameters(
                unfreeze_prefixes=["sequence_encoder", "transformer_tower", "sequence_decoder"]
            )
        else:
            model.unfreeze_parameters(unfreeze_prefixes=["sequence_encoder"])
            if second_stage_full_unfreeze_epoch is None:
                second_stage_full_unfreeze_epoch = max(1, second_stage_epochs // 3)
        n_params_s2 = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
        print(f"  Stage-2 trainable parameters: {n_params_s2:,}")

        s2_optimizer = optax.adamw(
            learning_rate=second_stage_lr,
            weight_decay=float(cfg.get("second_stage_weight_decay", float(cfg.weight_decay))),
        )
        s2_opt_state = s2_optimizer.init(model._params)

        @jax.jit
        def s2_train_step(params, opt_s, batch):
            def loss_func(p):
                preds = model._predict(
                    p,
                    model._state,
                    batch["sequences"],
                    batch["organism_index"],
                    negative_strand_mask=jnp.zeros(len(batch["sequences"]), dtype=bool),
                    strand_reindexing=None,
                    is_training=True,
                )[unique_head_name]
                return loss_fn(preds, batch)["loss"]

            loss, grads = jax.value_and_grad(loss_func)(params)
            updates, nxt = s2_optimizer.update(grads, opt_s, params)
            return optax.apply_updates(params, updates), nxt, loss

        # Stage 2 always runs through the encoder; shift can be disabled via max_shift=0.
        def collate_s2_train(b):
            return collate_yeast(b, max_seq_len, augment=True, max_shift=second_stage_max_shift)

        def collate_s2_eval(b):
            return collate_yeast(b, max_seq_len, augment=False)

        s2_batch_size = int(cfg.get("second_stage_batch_size", cfg.batch_size))

        # Optionally subsample for Stage 2 (full encoder is ~20x slower than cached)
        s2_ds = ds_train
        if second_stage_max_sequences is not None and second_stage_max_sequences < len(ds_train):
            rng_sub = torch.Generator().manual_seed(used_seed)
            s2_indices = (
                torch.randperm(len(ds_train), generator=rng_sub)[:second_stage_max_sequences]
                .sort()
                .values.tolist()
            )
            s2_ds = torch.utils.data.Subset(ds_train, s2_indices)
            print(
                f"  Stage-2 training data: {len(s2_ds):,} of {len(ds_train):,} sequences"
                f" (limited by second_stage_max_sequences)"
            )
        else:
            print(f"  Stage-2 training data: {len(ds_train):,} sequences")

        s2_train_loader = DataLoader(
            s2_ds,
            batch_size=s2_batch_size,
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_s2_train,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
        s2_val_loader = DataLoader(
            val_dataset,
            batch_size=s2_batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_s2_eval,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )

        best_s2_pearson = best_val_pearson
        best_s2_spearman = best_val_spearman
        s2_no_improve = 0

        # ── S2 resume from checkpoint after preemption ────────────────────
        _s2_start_epoch = 0
        if _resume_s2 and _s2_resume_data is not None:
            import orbax.checkpoint as ocp

            _s2_ckpt_path = (output_dir / "last_model_s2" / "checkpoint").resolve()
            if _s2_ckpt_path.exists():
                checkpointer = ocp.StandardCheckpointer()
                loaded_params, _ = checkpointer.restore(_s2_ckpt_path)
                model._params = jax.device_put(loaded_params)
                best_s2_pearson = float(_s2_resume_data["best_s2_pearson"])
                best_s2_spearman = float(_s2_resume_data["best_s2_spearman"])
                s2_no_improve = int(_s2_resume_data["s2_no_improve"])
                _s2_start_epoch = int(_s2_resume_data["s2_epoch"])
                # Restore optimizer state if saved, else reinitialize
                _opt_state_path = output_dir / "last_model_s2" / "opt_state.pkl"
                if _opt_state_path.exists():
                    with open(_opt_state_path, "rb") as _of:
                        s2_opt_state = jax.device_put(pickle.load(_of))
                    print(
                        f"  [RESUME] Loaded S2 epoch {_s2_start_epoch} checkpoint "
                        f"+ optimizer state. Resuming from epoch {_s2_start_epoch + 1}."
                    )
                else:
                    s2_opt_state = s2_optimizer.init(model._params)
                    print(
                        f"  [RESUME] Loaded S2 epoch {_s2_start_epoch} checkpoint "
                        f"(optimizer state not found, reinitializing). "
                        f"Resuming from epoch {_s2_start_epoch + 1}."
                    )
            else:
                print(
                    "  [RESUME] s2_progress.json exists but no checkpoint found. Starting S2 fresh."
                )

        for s2_epoch in range(_s2_start_epoch, second_stage_epochs):
            if (
                second_stage_unfreeze_mode == "gradual"
                and second_stage_full_unfreeze_epoch is not None
                and s2_epoch + 1 == second_stage_full_unfreeze_epoch
            ):
                model.unfreeze_parameters(
                    unfreeze_prefixes=["transformer_tower", "sequence_decoder"]
                )
                print(f"  Gradual Stage-2: unfreezing transformer/decoder at epoch {s2_epoch + 1}")
            s2_losses: list[float] = []
            pbar2 = tqdm(
                s2_train_loader,
                desc=f"S2 Epoch {s2_epoch + 1}/{second_stage_epochs}",
            )
            for batch in pbar2:
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}
                model._params, s2_opt_state, loss = s2_train_step(
                    model._params, s2_opt_state, batch_jax
                )
                s2_losses.append(float(loss))
                pbar2.set_postfix({"loss": f"{float(loss):.4f}"})

            s2_y_true_all: list[np.ndarray] = []
            s2_y_pred_all: list[np.ndarray] = []
            for batch in s2_val_loader:
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}
                logits_fwd = eval_step(model._params, batch_jax)
                probs_fwd = np.array(jax.nn.softmax(logits_fwd, axis=-1))
                if eval_use_reverse_complement:
                    batch_rc = dict(batch)
                    batch_rc["sequences"] = _reverse_complement_batch(
                        np.asarray(batch["sequences"])
                    )
                    batch_rc_jax = {k: jnp.array(v) for k, v in batch_rc.items()}
                    logits_rev = eval_step(model._params, batch_rc_jax)
                    probs_rev = np.array(jax.nn.softmax(logits_rev, axis=-1))
                    probs = (probs_fwd + probs_rev) / 2.0
                else:
                    probs = probs_fwd
                pred_expr = np.sum(probs * np.arange(num_tracks, dtype=np.float32), axis=-1)
                s2_y_pred_all.append(pred_expr)
                s2_y_true_all.append(np.array(batch["targets"]).reshape(-1))

            s2_y_true = np.concatenate(s2_y_true_all) if s2_y_true_all else np.array([])
            s2_y_pred = np.concatenate(s2_y_pred_all) if s2_y_pred_all else np.array([])
            s2_pear = _safe_corr(s2_y_true, s2_y_pred, pearsonr)
            s2_spear = _safe_corr(s2_y_true, s2_y_pred, spearmanr)
            s2_avg_train = float(np.mean(s2_losses)) if s2_losses else float("nan")

            global_epoch = int(cfg.epochs) + s2_epoch + 1
            print(
                f"S2 Epoch {s2_epoch + 1}  val/pearson_r={s2_pear:.4f}  "
                f"val/spearman_r={s2_spear:.4f}"
            )
            wandb.log(
                {
                    "epoch": global_epoch,
                    "stage": 2,
                    "train/loss": s2_avg_train,
                    "val/pearson_r": s2_pear,
                    "val/spearman_r": s2_spear,
                    "lr": second_stage_lr,
                }
            )

            if s2_pear > best_s2_pearson:
                best_s2_pearson = s2_pear
                best_s2_spearman = s2_spear
                s2_no_improve = 0
                try:
                    model.save_checkpoint(str(output_dir / "best_model"), save_full_model=True)
                    print(f"  New best (Stage 2): {best_s2_pearson:.4f}")
                except Exception as e:
                    print(
                        f"  New best (Stage 2): {best_s2_pearson:.4f} [WARN: checkpoint save failed: {e}]"
                    )
            else:
                s2_no_improve += 1

            try:
                model.save_checkpoint(str(output_dir / "last_model_s2"), save_full_model=True)
                # Save optimizer state for resume (preserves Adam momentum/variance)
                _opt_save_path = output_dir / "last_model_s2" / "opt_state.pkl"
                with open(_opt_save_path, "wb") as _of:
                    pickle.dump(jax.device_get(s2_opt_state), _of)
            except Exception as e:
                print(f"  [WARN] last_model_s2 save failed: {e}")

            # Save S2 progress for resume after preemption / --requeue
            with (output_dir / "s2_progress.json").open("w") as _pf:
                json.dump(
                    {
                        "s2_epoch": s2_epoch + 1,
                        "best_s2_pearson": float(best_s2_pearson),
                        "best_s2_spearman": float(best_s2_spearman),
                        "s2_no_improve": int(s2_no_improve),
                    },
                    _pf,
                )

            if _sigterm_received:
                print(
                    f"[SIGTERM] Clean shutdown after epoch {s2_epoch + 1}. "
                    f"Checkpoint saved. Exiting.",
                    flush=True,
                )
                break

            if s2_no_improve >= second_stage_early_stop:
                print(
                    f"Stage-2 early stopping at epoch {s2_epoch + 1} "
                    f"(no improvement for {second_stage_early_stop} epochs)"
                )
                break

            if max_wall_seconds is not None:
                elapsed = time.monotonic() - _train_start_time
                if elapsed >= max_wall_seconds * max_wall_fraction:
                    print(
                        f"Walltime guard: {elapsed / 3600:.1f}h elapsed "
                        f"(>= {max_wall_fraction * 100:.0f}% of {max_wall_seconds / 3600:.1f}h). "
                        f"Stopping S2 to leave time for eval."
                    )
                    break

        print(f"\nFinal best val Pearson: {best_s2_pearson:.4f} (Stage 2)")

    final_best_val_pearson = float(
        best_s2_pearson if second_stage_lr is not None else best_val_pearson
    )
    final_best_val_spearman = float(
        best_val_spearman if second_stage_lr is None else best_s2_spearman
    )

    # Write preliminary summary with val metrics BEFORE test eval
    # (ensures we capture at least val metrics even if test eval or checkpoint save fails)
    _prelim_summary = {
        "seed": int(used_seed),
        "aug_mode": str(aug_mode),
        "second_stage_enabled": bool(second_stage_lr is not None),
        "eval_use_reverse_complement": bool(eval_use_reverse_complement),
        "second_stage_unfreeze_mode": str(second_stage_unfreeze_mode),
        "second_stage_max_shift": int(second_stage_max_shift),
        "best_val_pearson_r": final_best_val_pearson,
        "best_val_spearman_r": final_best_val_spearman,
        "test_metrics": {},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_prelim_summary, f, indent=2)
    print("Saved preliminary summary.json (val metrics only)", flush=True)

    test_metrics: dict[str, dict[str, float]] = {}
    subset_dir = (
        Path(str(cfg.test_subset_dir))
        if cfg.get("test_subset_dir") is not None
        else Path(str(cfg.yeast_data_path)) / "test_subset_ids"
    )
    if subset_dir.exists():
        test_dataset = YeastDataset(
            data_path=str(cfg.yeast_data_path),
            split="test",
            context_mode=str(cfg.context_mode),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(cfg.get("test_batch_size", 256)),
            shuffle=False,
            num_workers=0,
        )

        @jax.jit
        def predict_step(params, x):
            org = jnp.zeros((x.shape[0],), dtype=jnp.int32)
            return model._predict(
                params,
                model._state,
                x,
                org,
                negative_strand_mask=jnp.zeros((x.shape[0],), dtype=bool),
                strand_reindexing=None,
            )[unique_head_name]

        preds_all: list[np.ndarray] = []
        for xb, _ in test_loader:
            x_fwd = xb[:, :4, :].permute(0, 2, 1).cpu().numpy().astype(np.float32)
            x_rev = x_fwd[:, ::-1, ::-1].copy()
            logits_fwd = np.array(predict_step(model._params, jnp.array(x_fwd)))
            logits_rev = np.array(predict_step(model._params, jnp.array(x_rev)))
            probs_fwd = np.array(jax.nn.softmax(logits_fwd, axis=-1))
            probs_rev = np.array(jax.nn.softmax(logits_rev, axis=-1))
            pred_fwd = np.sum(probs_fwd * np.arange(num_tracks, dtype=np.float32), axis=-1)
            pred_rev = np.sum(probs_rev * np.arange(num_tracks, dtype=np.float32), axis=-1)
            preds_all.append((pred_fwd + pred_rev) / 2.0)

        test_preds = np.concatenate(preds_all, axis=0)
        test_subsets = load_yeast_test_subsets(
            subset_dir=subset_dir,
            public_dir=(
                str(cfg.public_leaderboard_dir) if cfg.get("public_leaderboard_dir") else None
            ),
            use_private_only=bool(cfg.get("private_only_test", False)),
        )
        test_metrics = evaluate_yeast_test_subsets(
            predictions=test_preds,
            labels=test_dataset.labels.astype(np.float32),
            subsets=test_subsets,
        )

    summary = {
        "seed": int(used_seed),
        "aug_mode": str(aug_mode),
        "second_stage_enabled": bool(second_stage_lr is not None),
        "eval_use_reverse_complement": bool(eval_use_reverse_complement),
        "second_stage_unfreeze_mode": str(second_stage_unfreeze_mode),
        "second_stage_max_shift": int(second_stage_max_shift),
        "best_val_pearson_r": final_best_val_pearson,
        "best_val_spearman_r": final_best_val_spearman,
        "test_metrics": test_metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    wandb.log(
        {
            "final/best_val_pearson_r": final_best_val_pearson,
            "final/best_val_spearman_r": final_best_val_spearman,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
