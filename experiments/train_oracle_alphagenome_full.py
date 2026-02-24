#!/usr/bin/env python
"""Train AlphaGenome oracle with frozen encoder on the full K562 Malinois replication dataset.

Supports three augmentation / caching modes via the ``aug_mode`` config key:

* ``"full"``     — Original behaviour.  Full RC + shift augmentation; encoder runs on
                   every training batch.  Use for final production runs.
* ``"no_shift"`` — Pre-compute and cache canonical + RC encoder embeddings once, then
                   train only the small head.  RC augmentation is preserved (50 % per
                   sequence); shift augmentation is disabled.  ~20–50× faster per epoch;
                   ideal for rapid architecture search.
* ``"hybrid"``   — Cache canonical + RC embeddings.  At each training batch a coin flip
                   decides: cache path (no shift, random RC) or encoder path (full shift +
                   RC augmentation).  Matches the full augmentation distribution in
                   expectation; ~2× faster than ``"full"``.

Storage for canonical + RC cache (K562, ~700k train seqs, T=5, D=1536, float16):
    ≈ 21.5 GB  — fits in 96 GB H100 NVL VRAM / standard HPC scratch storage.
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

from albench.data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM, K562FullDataset
from albench.data.utils import one_hot_encode
from albench.models.alphagenome_heads import register_s2f_head
from albench.models.embedding_cache import (
    build_embedding_cache,
    build_head_only_predict_fn,
    load_embedding_cache,
    lookup_cached_batch,
    reinit_head_params,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def set_seed(seed: int | None) -> int:
    """Set deterministic seeds when provided, otherwise sample from entropy."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    return seed


def _build_compact_window_string(
    raw_seq: str,
    W: int,
    flank_bp: int,
    shift: int = 0,
) -> str:
    """Build left_flank + var + right_flank (full var, no N's). shift redistributes flank."""
    L = len(raw_seq)
    available = W - L
    if available < 0:
        raise ValueError(f"Compact window W={W} < len(seq)={L}")
    cap = min(200, flank_bp)
    left_len_base = min(cap, available // 2)
    shift_lo = left_len_base - min(cap, available)
    shift_hi = left_len_base - max(0, available - cap)
    s = np.clip(shift, shift_lo, shift_hi)
    left_len = left_len_base - s
    right_len = available - left_len
    left = MPRA_UPSTREAM[-left_len:] if left_len > 0 else ""
    right = MPRA_DOWNSTREAM[:right_len] if right_len > 0 else ""
    return left + raw_seq + right


def collate_compact_k562_full(
    batch: list[tuple],
    W: int,
    flank_bp: int = 200,
    augment: bool = False,
    max_shift: int = 200,
) -> dict[str, np.ndarray]:
    """Collate for compact window: raw (seq_str, label) -> build window with optional shift, encode."""
    batch_size = len(batch)
    x_batch = np.zeros((batch_size, W, 4), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)

    for i, (raw_seq, label) in enumerate(batch):
        label_val = float(label.numpy()) if hasattr(label, "numpy") else float(label)
        shift = 0
        if augment:
            L = len(raw_seq)
            available = W - L
            cap = min(200, flank_bp)
            left_len_base = min(cap, available // 2)
            shift_lo = left_len_base - min(cap, available)
            shift_hi = left_len_base - max(0, available - cap)
            if shift_hi > shift_lo:
                shift = int(np.random.randint(shift_lo, shift_hi + 1))
        built = _build_compact_window_string(raw_seq, W, flank_bp, shift=shift)
        enc = one_hot_encode(built, add_singleton_channel=False)
        seq_np = enc.T
        if augment and np.random.rand() > 0.5:
            seq_np = seq_np[::-1, ::-1]
        x_batch[i] = seq_np
        y_batch[i] = label_val

    return {
        "sequences": x_batch,
        "targets": y_batch,
        "organism_index": np.zeros((batch_size,), dtype=np.int32),
    }


def collate_compact_k562_indexed(
    batch_items: list[tuple],
    W: int,
    flank_bp: int = 200,
    augment: bool = False,
    max_shift: int = 200,
) -> dict[str, np.ndarray]:
    """Collate for indexed compact: (orig_idx, raw_seq, label) -> build window with optional shift, encode."""
    batch_size = len(batch_items)
    indices = np.empty(batch_size, dtype=np.int64)
    x_batch = np.zeros((batch_size, W, 4), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)

    for i, (orig_idx, raw_seq, label) in enumerate(batch_items):
        indices[i] = orig_idx
        label_val = float(label.numpy()) if hasattr(label, "numpy") else float(label)
        shift = 0
        if augment:
            L = len(raw_seq)
            available = W - L
            cap = min(200, flank_bp)
            left_len_base = min(cap, available // 2)
            shift_lo = left_len_base - min(cap, available)
            shift_hi = left_len_base - max(0, available - cap)
            if shift_hi > shift_lo:
                shift = int(np.random.randint(shift_lo, shift_hi + 1))
        built = _build_compact_window_string(raw_seq, W, flank_bp, shift=shift)
        enc = one_hot_encode(built, add_singleton_channel=False)
        seq_np = enc.T
        if augment and np.random.rand() > 0.5:
            seq_np = seq_np[::-1, ::-1]
        x_batch[i] = seq_np
        y_batch[i] = label_val

    return {
        "indices": indices,
        "sequences": x_batch,
        "targets": y_batch,
        "organism_index": np.zeros((batch_size,), dtype=np.int32),
    }


def collate_k562_full(
    batch: list[tuple], max_len: int = 600, augment: bool = False, max_shift: int = 15
) -> dict[str, np.ndarray]:
    """Collate K562 Full dataset batches, applying RC and shift augmentation."""
    batch_size = len(batch)
    x_batch = np.zeros((batch_size, max_len, 4), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)

    for i, (seq, label) in enumerate(batch):
        # seq is (5, L) -> we need (L, 4)
        seq_np = seq.numpy()[:4, :].T

        if augment:
            if np.random.rand() > 0.5:
                seq_np = seq_np[::-1, ::-1]
            if np.random.rand() > 0.5:
                shift = np.random.randint(-max_shift, max_shift + 1)
                seq_np = np.roll(seq_np, shift, axis=0)

        x_batch[i] = seq_np
        y_batch[i] = float(label.numpy())

    return {
        "sequences": x_batch,
        "targets": y_batch,
        "organism_index": np.zeros((batch_size,), dtype=np.int32),
    }


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a Dataset to also return the original (base-dataset) sample index.

    Correctly handles ``torch.utils.data.Subset`` by mapping subset positions
    back to original indices, so cache lookups stay aligned even when
    ``subset_fraction`` is used.
    """

    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self._ds = dataset
        if isinstance(dataset, torch.utils.data.Subset):
            self._orig_indices: np.ndarray | None = np.array(dataset.indices, dtype=np.int64)
        else:
            self._orig_indices = None

    def __getitem__(self, i: int) -> tuple:
        orig_idx = int(self._orig_indices[i]) if self._orig_indices is not None else i
        return (orig_idx,) + tuple(self._ds[i])

    def __len__(self) -> int:
        return len(self._ds)


def collate_k562_indexed(
    batch_items: list[tuple],
    max_len: int = 600,
    augment: bool = False,
    max_shift: int = 15,
) -> dict[str, np.ndarray]:
    """Collate for indexed datasets; includes original dataset indices.

    Used in ``no_shift`` and ``hybrid`` modes so the main loop can look up
    cached embeddings by index.  The ``sequences`` field is always populated
    so the hybrid encoder path can run augmentation on-the-fly.
    """
    batch_size = len(batch_items)
    indices = np.empty(batch_size, dtype=np.int64)
    x_batch = np.zeros((batch_size, max_len, 4), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)

    for i, (orig_idx, seq, label) in enumerate(batch_items):
        indices[i] = orig_idx
        seq_np = seq.numpy()[:4, :].T

        if augment:
            if np.random.rand() > 0.5:
                seq_np = seq_np[::-1, ::-1]
            if np.random.rand() > 0.5:
                shift = np.random.randint(-max_shift, max_shift + 1)
                seq_np = np.roll(seq_np, shift, axis=0)

        x_batch[i] = seq_np
        y_batch[i] = float(label.numpy())

    return {
        "indices": indices,
        "sequences": x_batch,
        "targets": y_batch,
        "organism_index": np.zeros((batch_size,), dtype=np.int32),
    }


def collate_val_indexed(batch_items: list[tuple]) -> dict[str, np.ndarray]:
    """Minimal collate for cached val eval: returns only indices + targets."""
    indices = np.array([orig_idx for orig_idx, _seq, _label in batch_items], dtype=np.int64)
    targets = np.array(
        [float(label.numpy()) for _orig_idx, _seq, label in batch_items], dtype=np.float32
    )
    return {"indices": indices, "targets": targets}


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="oracle_alphagenome_k562_full",
)
def main(cfg: DictConfig) -> None:
    """Train a frozen-encoder AlphaGenome oracle on the full 800k K562 Malinois dataset."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aug_mode = str(cfg.get("aug_mode", "full"))
    if aug_mode not in ("full", "no_shift", "hybrid"):
        raise ValueError(f"aug_mode must be 'full', 'no_shift', or 'hybrid'; got {aug_mode!r}")

    num_tracks = int(cfg.num_tracks)
    if num_tracks <= 0:
        raise ValueError("num_tracks must be > 0")

    wandb.init(
        project="albench-s2f",
        name=f"oracle_alphagenome_{cfg.task_mode}_{cfg.head_arch}_full_seed{used_seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[
            "oracle",
            "alphagenome",
            str(cfg.task_mode),
            str(cfg.head_arch),
            "full_dataset_baseline",
            aug_mode,
        ],
        mode=str(cfg.wandb_mode),
        job_type="oracle_training",
    )

    # Include arch in head name to avoid shape collisions from old checkpoints
    # that may have stored head params under the same name with a different context mode.
    arch_slug = str(cfg.head_arch).replace("-", "_")
    unique_head_name = f"{cfg.head_name}_{arch_slug}_v4"

    register_s2f_head(
        head_name=unique_head_name,
        arch=str(cfg.head_arch),
        task_mode=str(cfg.task_mode),
        num_tracks=num_tracks,
    )

    weights_path = str(Path(str(cfg.weights_path)).expanduser().resolve())
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"AlphaGenome weights path does not exist: {weights_path}. "
            "Expected an absolute path to the local in-repo checkpoint directory."
        )
    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
    )
    # Re-init head so checkpoint head params (e.g. from T=128 run) don't cause shape mismatch.
    reinit_head_params(model, unique_head_name, num_tokens=5, dim=1536)
    model.freeze_except_head(unique_head_name)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    print(f"Total parameters: {param_count:,}")

    loss_fn = model.create_loss_fn_for_head(unique_head_name)
    eval_fn = model.create_loss_fn_for_head(unique_head_name)
    if cfg.gradients_clip is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(cfg.gradients_clip)),
            optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay)),
        )
    else:
        optimizer = optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    opt_state = optimizer.init(model._params)

    # ── Datasets ──────────────────────────────────────────────────────────────
    use_compact_window = bool(cfg.get("use_compact_window", False))
    flank_bp = int(cfg.get("flank_bp", 200))
    compact_window_bp = cfg.get("compact_window_bp", None)
    compact_window_bp = int(compact_window_bp) if compact_window_bp is not None else None

    if use_compact_window:
        ds_train = K562FullDataset(data_path=str(cfg.k562_data_path), split="train", store_raw=True)
        min_var_len = int(np.min(ds_train.raw_lengths))
        ds_train.set_compact_window(min_var_len, flank_bp, window_bp=compact_window_bp)
        effective_max_seq_len = ds_train.sequence_length
        val_dataset = K562FullDataset(
            data_path=str(cfg.k562_data_path), split="val", store_raw=True
        )
        val_dataset.set_compact_window(min_var_len, flank_bp, window_bp=compact_window_bp)
        print(
            f"[Compact window] min_var_len={min_var_len}, flank_bp={flank_bp}, "
            f"W={effective_max_seq_len} (compact_window_bp={compact_window_bp})"
        )
    else:
        ds_train = K562FullDataset(data_path=str(cfg.k562_data_path), split="train")
        val_dataset = K562FullDataset(data_path=str(cfg.k562_data_path), split="val")
        effective_max_seq_len = int(cfg.max_seq_len)

    include_pool = bool(cfg.get("include_pool", False))
    if include_pool:
        if aug_mode != "full":
            raise ValueError(
                "include_pool=True is not supported with cached aug_mode. "
                "Set aug_mode='full' or include_pool=False."
            )
        if use_compact_window:
            raise ValueError("include_pool=True is not supported with use_compact_window.")
        ds_pool = K562FullDataset(data_path=str(cfg.k562_data_path), split="pool")
        train_dataset: torch.utils.data.Dataset = torch.utils.data.ConcatDataset(
            [ds_train, ds_pool]
        )
    else:
        train_dataset = ds_train

    subset_fraction = cfg.get("subset_fraction", None)
    if subset_fraction is not None:
        frac = float(subset_fraction)
        if frac <= 0.0 or frac > 1.0:
            raise ValueError("subset_fraction must be in (0, 1]")
        n_total = len(train_dataset)
        n_take = max(1, int(n_total * frac))
        sub_indices = np.random.choice(n_total, size=n_take, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, sub_indices.tolist())

    # ── Embedding cache (non-full modes) ──────────────────────────────────────
    train_canonical = train_rc = val_canonical = None
    head_predict_fn = None

    if aug_mode != "full":
        raw_cache_dir = cfg.get("cache_dir", None)
        if raw_cache_dir is not None:
            cache_dir = Path(str(raw_cache_dir)).expanduser().resolve()
        else:
            cache_dir = output_dir / (
                "embedding_cache_compact" if use_compact_window else "embedding_cache"
            )
        # Always build cache on the FULL ds_train (not subset) so indices stay valid.
        build_embedding_cache(
            model,
            ds_train,
            cache_dir,
            "train",
            max_seq_len=effective_max_seq_len,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
        )
        build_embedding_cache(
            model,
            val_dataset,
            cache_dir,
            "val",
            max_seq_len=effective_max_seq_len,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
        )
        train_canonical, train_rc = load_embedding_cache(cache_dir, "train")
        val_canonical, _ = load_embedding_cache(cache_dir, "val")
        head_predict_fn = build_head_only_predict_fn(model, unique_head_name)
        # Verify head params shape (catches stale checkpoint head mismatch)
        _dummy_emb = jnp.zeros((2, 5, 1536), dtype=jnp.float32)
        _dummy_org = jnp.zeros((2,), dtype=jnp.int32)
        _ = head_predict_fn(model._params, _dummy_emb, _dummy_org)

    # ── JIT steps ─────────────────────────────────────────────────────────────

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
            )[unique_head_name]
            return loss_fn(preds, batch)["loss"]

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss

    @jax.jit
    def eval_step(params, batch):
        preds = model._predict(
            params,
            model._state,
            batch["sequences"],
            batch["organism_index"],
            negative_strand_mask=jnp.zeros(len(batch["sequences"]), dtype=bool),
            strand_reindexing=None,
        )
        head_preds = preds[unique_head_name]
        return eval_fn(head_preds, batch)["loss"], head_preds

    # Head-only steps (defined only when cache is available)
    if head_predict_fn is not None:

        @jax.jit
        def cached_train_step(params, current_opt_state, encoder_output, targets, organism_index):
            def loss_func(p):
                preds = head_predict_fn(p, encoder_output, organism_index)
                pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
                return jnp.mean((pred - targets) ** 2)

            loss, grads = jax.value_and_grad(loss_func)(params)
            updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state, loss

        @jax.jit
        def cached_eval_step(params, encoder_output, organism_index):
            return head_predict_fn(params, encoder_output, organism_index)

    # When compact + collate-from-raw: dataset must return raw (cache already built with encoded).
    if use_compact_window:
        ds_train.set_compact_return_raw(True)
        val_dataset.set_compact_return_raw(True)

    # ── Data loaders ──────────────────────────────────────────────────────────
    n_workers = int(cfg.num_workers)

    if aug_mode == "full":
        if use_compact_window:

            def collate_fn_train(batch):
                return collate_compact_k562_full(
                    batch, effective_max_seq_len, flank_bp=flank_bp, augment=True
                )

            def collate_fn_eval(batch):
                return collate_compact_k562_full(
                    batch, effective_max_seq_len, flank_bp=flank_bp, augment=False
                )
        else:

            def collate_fn_train(batch):
                return collate_k562_full(batch, effective_max_seq_len, augment=True)

            def collate_fn_eval(batch):
                return collate_k562_full(batch, effective_max_seq_len, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_fn_train,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
        val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=n_workers,
            collate_fn=collate_fn_eval,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
    else:
        # For no_shift / hybrid: train loader returns indices + sequences + targets.
        # no_shift collate: NO augmentation (cache path always used; RC applied in main loop).
        # hybrid  collate: full augmentation (for the 50% encoder-path batches).
        # We always collate with augment=True so encoder-path batches are ready; the
        # cache path ignores the sequences and looks up embeddings instead.
        if use_compact_window:

            def collate_fn_train_indexed(batch):
                return collate_compact_k562_indexed(
                    batch, effective_max_seq_len, flank_bp=flank_bp, augment=True
                )
        else:

            def collate_fn_train_indexed(batch):
                return collate_k562_indexed(batch, effective_max_seq_len, augment=True)

        train_loader = DataLoader(
            IndexedDataset(train_dataset),
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=n_workers,
            collate_fn=collate_fn_train_indexed,
            pin_memory=True,
            persistent_workers=n_workers > 0,
        )
        # Val loader: only needs indices + targets (no sequences; uses cache).
        val_loader = DataLoader(
            IndexedDataset(val_dataset),
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=0,  # lightweight — no sequence loading needed
            collate_fn=collate_val_indexed,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_pearson = -1.0
    early_stop_patience = int(cfg.get("early_stop_patience", 5))
    val_eval_interval = int(cfg.get("val_eval_interval", 1))
    epochs_no_improve = 0

    for epoch in range(int(cfg.epochs)):
        train_losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{int(cfg.epochs)}")

        if aug_mode == "full":
            # ── Full mode: encoder runs every step ────────────────────────────
            for batch in pbar:
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}
                model._params, opt_state, loss = train_step(model._params, opt_state, batch_jax)
                loss_v = float(loss)
                train_losses.append(loss_v)
                pbar.set_postfix({"loss": f"{loss_v:.4f}"})

        elif aug_mode == "no_shift":
            # ── No-shift mode: always use cache, head-only step ───────────────
            for batch in pbar:
                indices = batch["indices"]
                targets = batch["targets"]
                emb = lookup_cached_batch(indices, train_canonical, train_rc)
                org_idx = jnp.zeros(len(indices), dtype=jnp.int32)
                model._params, opt_state, loss = cached_train_step(
                    model._params,
                    opt_state,
                    jnp.array(emb),
                    jnp.array(targets),
                    org_idx,
                )
                loss_v = float(loss)
                train_losses.append(loss_v)
                pbar.set_postfix({"loss": f"{loss_v:.4f}"})

        else:  # aug_mode == "hybrid"
            # ── Hybrid: 50 % cache (no shift) / 50 % encoder (with shift) ────
            for batch in pbar:
                if np.random.rand() > 0.5:
                    # Cache path: per-sequence RC, no shift
                    indices = batch["indices"]
                    targets = batch["targets"]
                    emb = lookup_cached_batch(indices, train_canonical, train_rc)
                    org_idx = jnp.zeros(len(indices), dtype=jnp.int32)
                    model._params, opt_state, loss = cached_train_step(
                        model._params,
                        opt_state,
                        jnp.array(emb),
                        jnp.array(targets),
                        org_idx,
                    )
                else:
                    # Encoder path: collated with full augmentation (RC + shift)
                    batch_jax = {k: jnp.array(v) for k, v in batch.items() if k != "indices"}
                    model._params, opt_state, loss = train_step(model._params, opt_state, batch_jax)
                loss_v = float(loss)
                train_losses.append(loss_v)
                pbar.set_postfix({"loss": f"{loss_v:.4f}"})

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")

        if (epoch + 1) % val_eval_interval != 0:
            wandb.log({"epoch": epoch + 1, "train/loss": avg_train})
            model.save_checkpoint(
                str(output_dir / "last_model"), save_full_model=bool(cfg.save_full_model)
            )
            continue

        # ── Validation eval ───────────────────────────────────────────────────
        val_losses: list[float] = []
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []

        if aug_mode == "full":
            # Encoder-based val eval (augment=False)
            for batch in val_loader:
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}
                val_loss, preds = eval_step(model._params, batch_jax)
                val_losses.append(float(val_loss))
                y_pred_all.append(np.array(preds).reshape(-1))
                y_true_all.append(np.array(batch["targets"]).reshape(-1))
        else:
            # Cache-based val eval: use canonical (no augmentation)
            for batch in val_loader:
                indices = batch["indices"]
                targets = batch["targets"]
                emb = jnp.array(val_canonical[indices].astype(np.float32))
                org_idx = jnp.zeros(len(indices), dtype=jnp.int32)
                preds = cached_eval_step(model._params, emb, org_idx)
                preds_np = np.array(preds).reshape(-1)
                # MSE loss for logging consistency
                val_losses.append(float(np.mean((preds_np - targets) ** 2)))
                y_pred_all.append(preds_np)
                y_true_all.append(np.array(targets).reshape(-1))

        y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
        y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        pear = _safe_corr(y_true, y_pred, pearsonr)
        spear = _safe_corr(y_true, y_pred, spearmanr)

        print(f"Epoch {epoch + 1} val/pearson_r={pear:.4f} val/spearman_r={spear:.4f}")
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": avg_train,
                "val/loss": avg_val,
                "val/pearson_r": pear,
                "val/spearman_r": spear,
            }
        )

        if pear > best_val_pearson:
            best_val_pearson = pear
            epochs_no_improve = 0
            model.save_checkpoint(
                str(output_dir / "best_model"), save_full_model=bool(cfg.save_full_model)
            )
        else:
            epochs_no_improve += 1

        model.save_checkpoint(
            str(output_dir / "last_model"), save_full_model=bool(cfg.save_full_model)
        )

        if epochs_no_improve >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val Pearson improvement for {early_stop_patience} epochs)"
            )
            break

    wandb.finish()


if __name__ == "__main__":
    main()
