#!/usr/bin/env python
"""Train AlphaGenome oracle with frozen encoder and configurable head."""

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

from albench.data.k562 import K562Dataset
from albench.data.yeast import YeastDataset
from albench.models.alphagenome_heads import register_s2f_head

# 150bp yeast plasmid context assembled as 54 + core150 + 89.
FLANK_5_PRIME = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATC"
FLANK_3_PRIME = (
    "GGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA"
)
FLANK_5_ENC = None
FLANK_3_ENC = None


def _init_yeast_flanks() -> tuple[np.ndarray, np.ndarray]:
    """Build one-hot flank tensors once for yeast collation."""
    global FLANK_5_ENC, FLANK_3_ENC
    if FLANK_5_ENC is not None and FLANK_3_ENC is not None:
        return FLANK_5_ENC, FLANK_3_ENC

    def one_hot_encode_sequence(seq: str) -> np.ndarray:
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, char in enumerate(seq):
            if char in mapping:
                arr[i, mapping[char]] = 1.0
        return arr

    FLANK_5_ENC = one_hot_encode_sequence(FLANK_5_PRIME)
    FLANK_3_ENC = one_hot_encode_sequence(FLANK_3_PRIME)
    return FLANK_5_ENC, FLANK_3_ENC


def set_seed(seed: int | None) -> int:
    """Set deterministic seeds when provided, otherwise sample from entropy."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    return seed


def collate_yeast(batch: list[tuple], max_len: int = 384) -> dict[str, np.ndarray]:
    """Build AlphaGenome inputs for yeast using plasmid-style context."""
    flank_5, flank_3 = _init_yeast_flanks()
    batch_size = len(batch)
    x_batch = np.zeros((batch_size, max_len, 4), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)

    len_5 = flank_5.shape[0]
    len_3 = flank_3.shape[0]
    len_core = 150
    total_valid = len_5 + len_core + len_3
    start_idx = (max_len - total_valid) // 2

    idx_5_start = start_idx
    idx_5_end = idx_5_start + len_5
    idx_core_start = idx_5_end
    idx_core_end = idx_core_start + len_core
    idx_3_start = idx_core_end
    idx_3_end = idx_3_start + len_3

    for i, (seq, label) in enumerate(batch):
        seq_np = seq.numpy()
        core = seq_np[:4, :].T
        x_batch[i, idx_5_start:idx_5_end, :] = flank_5
        x_batch[i, idx_core_start:idx_core_end, :] = core
        x_batch[i, idx_3_start:idx_3_end, :] = flank_3
        y_batch[i] = float(label.numpy())

    return {
        "sequences": x_batch,
        "targets": y_batch,
        "organism_index": np.zeros((batch_size,), dtype=np.int32),
    }


def _center_pad(seq: np.ndarray, target_len: int = 384) -> np.ndarray:
    """Center-pad or trim a (L, 4) sequence to target length."""
    curr_len = seq.shape[0]
    if curr_len == target_len:
        return seq
    if curr_len > target_len:
        start = (curr_len - target_len) // 2
        return seq[start : start + target_len]

    pad = np.zeros((target_len, 4), dtype=np.float32)
    left = (target_len - curr_len) // 2
    pad[left : left + curr_len, :] = seq
    return pad


def collate_k562(batch: list[tuple], max_len: int = 384) -> dict[str, np.ndarray]:
    """Build AlphaGenome inputs for K562 using center-padded 200bp cores."""
    batch_size = len(batch)
    x_batch = np.zeros((batch_size, max_len, 4), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)

    for i, (seq, label) in enumerate(batch):
        seq_np = seq.numpy()[:4, :].T
        x_batch[i] = _center_pad(seq_np, target_len=max_len)
        y_batch[i] = float(label.numpy())

    return {
        "sequences": x_batch,
        "targets": y_batch,
        "organism_index": np.zeros((batch_size,), dtype=np.int32),
    }


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="oracle_alphagenome_yeast",
)
def main(cfg: DictConfig) -> None:
    """Train a frozen-encoder AlphaGenome oracle with configurable head."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    num_tracks = int(cfg.num_tracks)
    if num_tracks <= 0:
        raise ValueError("num_tracks must be > 0")

    wandb.init(
        project="albench-s2f",
        name=f"oracle_alphagenome_{cfg.task_mode}_{cfg.head_arch}_seed{used_seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["oracle", "alphagenome", str(cfg.task_mode), str(cfg.head_arch)],
        mode=str(cfg.wandb_mode),
        job_type="oracle_training",
    )

    register_s2f_head(
        head_name=str(cfg.head_name),
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
        heads=[str(cfg.head_name)],
        checkpoint_path=weights_path,
        use_encoder_output=True,
    )
    model.freeze_except_head(str(cfg.head_name))
    loss_fn_head = model.create_loss_fn_for_head(str(cfg.head_name))

    optimizer = optax.chain(
        optax.clip_by_global_norm(float(cfg.gradients_clip)),
        optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay)),
    )
    opt_state = optimizer.init(model._params)

    if str(cfg.task_mode) == "yeast":
        ds_train = YeastDataset(
            data_path=str(cfg.yeast_data_path),
            split="train",
            subset_size=None,
            context_mode=str(cfg.context_mode),
        )
        if bool(cfg.include_pool):
            ds_pool = YeastDataset(
                data_path=str(cfg.yeast_data_path),
                split="pool",
                subset_size=None,
                context_mode=str(cfg.context_mode),
            )
            train_dataset = torch.utils.data.ConcatDataset([ds_train, ds_pool])
        else:
            train_dataset = ds_train
        val_dataset = YeastDataset(
            data_path=str(cfg.yeast_data_path),
            split="val",
            context_mode=str(cfg.context_mode),
        )

        def collate_fn(batch: list[tuple]) -> dict[str, np.ndarray]:
            return collate_yeast(batch, int(cfg.max_seq_len))
    else:
        ds_train = K562Dataset(data_path=str(cfg.k562_data_path), split="train")
        if bool(cfg.include_pool):
            ds_pool = K562Dataset(data_path=str(cfg.k562_data_path), split="pool")
            train_dataset = torch.utils.data.ConcatDataset([ds_train, ds_pool])
        else:
            train_dataset = ds_train
        ds_val = K562Dataset(data_path=str(cfg.k562_data_path), split="val")
        val_dataset = ds_val

        def collate_fn(batch: list[tuple]) -> dict[str, np.ndarray]:
            return collate_k562(batch, int(cfg.max_seq_len))

    subset_fraction = cfg.subset_fraction
    if subset_fraction is not None:
        frac = float(subset_fraction)
        if frac <= 0.0 or frac > 1.0:
            raise ValueError("subset_fraction must be in (0, 1]")
        n_total = len(train_dataset)
        n_take = max(1, int(n_total * frac))
        indices = np.random.choice(n_total, size=n_take, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_fn,
    )

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
            )[str(cfg.head_name)]
            return loss_fn_head(preds, batch)["loss"]

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
        )[str(cfg.head_name)]
        return loss_fn_head(preds, batch)["loss"], preds

    best_val_pearson = -1.0
    bin_values = jnp.arange(18, dtype=jnp.float32)

    for epoch in range(int(cfg.epochs)):
        train_losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{int(cfg.epochs)}")
        for batch in pbar:
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            model._params, opt_state, loss = train_step(model._params, opt_state, batch_jax)
            loss_v = float(loss)
            train_losses.append(loss_v)
            pbar.set_postfix({"loss": f"{loss_v:.4f}"})

        val_losses: list[float] = []
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        for batch in val_loader:
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            val_loss, preds = eval_step(model._params, batch_jax)
            val_losses.append(float(val_loss))

            preds_np = np.array(preds)
            if str(cfg.task_mode) == "yeast":
                probs = jax.nn.softmax(preds, axis=-1)
                pred_expr = np.array(jnp.sum(probs * bin_values, axis=-1))
            else:
                pred_expr = preds_np.reshape(-1)

            y_pred_all.append(pred_expr)
            y_true_all.append(np.array(batch["targets"]).reshape(-1))

        y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
        y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")

        pear = _safe_corr(y_true, y_pred, pearsonr)
        spear = _safe_corr(y_true, y_pred, spearmanr)

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
            model.save_checkpoint(
                str(output_dir / "best_model"), save_full_model=bool(cfg.save_full_model)
            )

        model.save_checkpoint(
            str(output_dir / "last_model"), save_full_model=bool(cfg.save_full_model)
        )

    wandb.finish()


if __name__ == "__main__":
    main()
