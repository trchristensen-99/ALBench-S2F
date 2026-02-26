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
  Stage 2 always runs in ``full`` aug mode (shift + RC via the encoder).
  Set ``second_stage_lr: null`` to skip Stage 2 entirely.

Yeast-specific details:
* 150bp core sequences padded to 384bp using plasmid flanks (54bp 5' + 89bp 3').
* Objective: 18-bin cross-entropy (KL) on discretised expression levels.
* Metric: Pearson r between predicted expected bin and ground-truth label.
* T=3 encoder tokens (384bp / 128bp stride).

Embedding cache size (yeast ~960k train+pool seqs, T=3, D=1536, float16):
    ≈ 8.4 GB  — comfortably fits in HPC scratch storage / H100 VRAM.
"""

from __future__ import annotations

import os
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
)  # 89bp

_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_str(s: str) -> np.ndarray:
    out = np.zeros((len(s), 4), dtype=np.float32)
    for i, c in enumerate(s):
        if c in _MAPPING:
            out[i, _MAPPING[c]] = 1.0
    return out


_FLANK5_ENC = _encode_str(_YEAST_FLANK_5)  # (54, 4)
_FLANK3_ENC = _encode_str(_YEAST_FLANK_3)  # (89, 4)

_CANONICAL_TOTAL = _FLANK5_ENC.shape[0] + 150 + _FLANK3_ENC.shape[0]  # 293


# ── Collation helpers ──────────────────────────────────────────────────────────


def _build_yeast_window(core_4ch: np.ndarray, max_len: int, shift: int = 0) -> np.ndarray:
    """Concatenate flanks + core and slice the 384bp window with optional shift."""
    full = np.concatenate([_FLANK5_ENC, core_4ch, _FLANK3_ENC], axis=0)  # (293, 4)
    # Buffer large enough to accommodate shift without clipping
    max_shift = max(110, abs(shift) + 10)
    buf_len = max_len + 2 * max_shift
    buf = np.zeros((buf_len, 4), dtype=np.float32)
    base = (buf_len - _CANONICAL_TOTAL) // 2
    buf[base : base + _CANONICAL_TOTAL, :] = full
    win_base = (buf_len - max_len) // 2
    start = np.clip(win_base + shift, 0, buf_len - max_len)
    return buf[start : start + max_len]


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
        core = seq.numpy()[:4, :].T  # (150, 4)
        shift = int(np.random.randint(-max_shift, max_shift + 1)) if augment else 0
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
        core = seq.numpy()[:4, :].T
        shift = int(np.random.randint(-max_shift, max_shift + 1)) if augment else 0
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

    # Two-stage config
    _s2_lr_raw = cfg.get("second_stage_lr", None)
    second_stage_lr = float(_s2_lr_raw) if _s2_lr_raw is not None else None
    second_stage_epochs = int(cfg.get("second_stage_epochs", 50))
    second_stage_early_stop = int(cfg.get("second_stage_early_stop_patience", 10))

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
    unique_head_name = f"ag_yeast_{arch_slug}_v1"

    _hidden_dims_raw = cfg.get("hidden_dims", None)
    _hidden_dims: list[int] | None = (
        [int(d) for d in _hidden_dims_raw] if _hidden_dims_raw is not None else None
    )
    register_s2f_head(
        head_name=unique_head_name,
        arch=str(cfg.head_arch),
        task_mode="yeast",
        num_tracks=num_tracks,
        dropout_rate=dropout_rate,
        hidden_dims=_hidden_dims,
    )

    weights_path = Path(str(cfg.weights_path)).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {weights_path}")
    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=str(weights_path),
        use_encoder_output=True,
    )
    # num_tokens=3 for T=3 (384bp / 128bp stride).
    reinit_head_params(model, unique_head_name, num_tokens=3, dim=1536)
    model.freeze_except_head(unique_head_name)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    print(f"Total parameters: {n_params:,}")

    loss_fn = model.create_loss_fn_for_head(unique_head_name)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if _use_plateau:
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
    ds_pool = YeastDataset(
        data_path=str(cfg.yeast_data_path),
        split="pool",
        context_mode=context_mode,
    )
    val_dataset = YeastDataset(
        data_path=str(cfg.yeast_data_path),
        split="val",
        context_mode=context_mode,
    )

    include_pool = bool(cfg.get("include_pool", True))
    if include_pool:
        train_dataset: torch.utils.data.Dataset = torch.utils.data.ConcatDataset(
            [ds_train, ds_pool]
        )
    else:
        train_dataset = ds_train

    # ── Embedding cache (no_shift / hybrid) ───────────────────────────────────
    train_canonical = train_rc = val_canonical = None
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
        # Pool cache (separate split; indexed separately)
        if include_pool:
            build_embedding_cache(
                model,
                ds_pool,
                cache_dir,
                "pool",
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
        # Load all into memory (yeast train+pool cache = ~104 GB total)
        train_can_raw, train_rc_raw = load_embedding_cache(cache_dir, "train", mmap_mode=None)
        if include_pool:
            pool_can_raw, pool_rc_raw = load_embedding_cache(cache_dir, "pool", mmap_mode=None)
            # Concatenate along seq axis so index 0..N_train-1 = train, N_train.. = pool
            train_canonical = np.concatenate([train_can_raw, pool_can_raw], axis=0)
            train_rc = np.concatenate([train_rc_raw, pool_rc_raw], axis=0)
        else:
            train_canonical = train_can_raw
            train_rc = train_rc_raw
        val_canonical, _ = load_embedding_cache(cache_dir, "val")

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
    epochs_no_improve = 0
    rng = jax.random.PRNGKey(used_seed)
    current_lr = float(cfg.lr)
    lr_plateau_best = -float("inf")
    lr_plateau_counter = 0
    _min_lr = float(cfg.lr) * 0.01

    for epoch in range(int(cfg.epochs)):
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
                preds = eval_step(model._params, batch_jax)
                probs = np.array(jax.nn.softmax(preds, axis=-1))
                pred_expr = np.sum(probs * np.arange(num_tracks, dtype=np.float32), axis=-1)
                y_pred_all.append(pred_expr)
                y_true_all.append(np.array(batch["targets"]).reshape(-1))
        else:
            for batch in val_loader:
                idx = batch["indices"]
                emb = jnp.array(val_canonical[idx].astype(np.float32))
                org = jnp.zeros(len(idx), dtype=jnp.int32)
                preds = cached_eval_step(model._params, emb, org)
                probs = np.array(jax.nn.softmax(preds, axis=-1))
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
            epochs_no_improve = 0
            model.save_checkpoint(str(output_dir / "best_model"), save_full_model=False)
        else:
            epochs_no_improve += 1

        model.save_checkpoint(str(output_dir / "last_model"), save_full_model=False)

        if epochs_no_improve >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)"
            )
            break

    # ── Stage 2: unfreeze encoder, end-to-end fine-tuning ────────────────────
    if second_stage_lr is not None:
        print(
            f"\n=== Stage 2: unfreezing encoder, lr={second_stage_lr:.1e}, "
            f"epochs={second_stage_epochs} ==="
        )
        # Load best Stage-1 checkpoint before unfreezing
        best_s1 = output_dir / "best_model"
        if best_s1.exists():
            model.load_checkpoint(str(best_s1))
            print("  Loaded best Stage-1 checkpoint.")

        # Save full model (encoder + head) so Stage 2 can load it later if needed
        model.save_checkpoint(str(output_dir / "stage1_best"), save_full_model=True)

        # Unfreeze everything
        model.unfreeze()
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
                )[unique_head_name]
                return loss_fn(preds, batch)["loss"]

            loss, grads = jax.value_and_grad(loss_func)(params)
            updates, nxt = s2_optimizer.update(grads, opt_s, params)
            return optax.apply_updates(params, updates), nxt, loss

        # Stage 2 always uses full augmentation (encoder runs every step)
        def collate_s2_train(b):
            return collate_yeast(b, max_seq_len, augment=True)

        def collate_s2_eval(b):
            return collate_yeast(b, max_seq_len, augment=False)

        s2_train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_s2_train,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
        s2_val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_s2_eval,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )

        best_s2_pearson = best_val_pearson
        s2_no_improve = 0

        for s2_epoch in range(second_stage_epochs):
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
                preds = eval_step(model._params, batch_jax)
                probs = np.array(jax.nn.softmax(preds, axis=-1))
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
                s2_no_improve = 0
                model.save_checkpoint(str(output_dir / "best_model"), save_full_model=True)
                print(f"  New best (Stage 2): {best_s2_pearson:.4f}")
            else:
                s2_no_improve += 1

            model.save_checkpoint(str(output_dir / "last_model_s2"), save_full_model=True)

            if s2_no_improve >= second_stage_early_stop:
                print(
                    f"Stage-2 early stopping at epoch {s2_epoch + 1} "
                    f"(no improvement for {second_stage_early_stop} epochs)"
                )
                break

        print(f"\nFinal best val Pearson: {best_s2_pearson:.4f} (Stage 2)")

    wandb.finish()


if __name__ == "__main__":
    main()
