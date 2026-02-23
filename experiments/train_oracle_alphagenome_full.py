#!/usr/bin/env python
"""Train AlphaGenome oracle with frozen encoder on the full K562 Malinois replication dataset."""

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

from albench.data.k562_full import K562FullDataset
from albench.models.alphagenome_heads import register_s2f_head


def set_seed(seed: int | None) -> int:
    """Set deterministic seeds when provided, otherwise sample from entropy."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    return seed


def collate_k562_full(
    batch: list[tuple], max_len: int = 600, augment: bool = False
) -> dict[str, np.ndarray]:
    """Collate K562 Full dataset batches, applying RC augmentation."""
    batch_size = len(batch)
    x_batch = np.zeros((batch_size, max_len, 4), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)

    for i, (seq, label) in enumerate(batch):
        # seq is (5, L) -> we need (L, 4)
        seq_np = seq.numpy()[:4, :].T

        if augment and np.random.rand() > 0.5:
            seq_np = seq_np[::-1, ::-1]

        x_batch[i] = seq_np
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
    config_name="oracle_alphagenome_k562_full",
)
def main(cfg: DictConfig) -> None:
    """Train a frozen-encoder AlphaGenome oracle on the full 800k K562 Malinois dataset."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        ],
        mode=str(cfg.wandb_mode),
        job_type="oracle_training",
    )

    # Include arch in head name to avoid shape collisions from old checkpoints
    # that may have stored head params under the same name with a different context mode.
    arch_slug = str(cfg.head_arch).replace("-", "_")
    unique_head_name = f"{cfg.head_name}_{arch_slug}_v3"

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
    model.freeze_except_head(unique_head_name)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    print(f"Total parameters: {param_count:,}")

    loss_fn = model.create_loss_fn_for_head(unique_head_name)
    eval_fn = model.create_loss_fn_for_head(unique_head_name)
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(cfg.gradients_clip)),
        optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay)),
    )
    opt_state = optimizer.init(model._params)

    ds_train = K562FullDataset(data_path=str(cfg.k562_data_path), split="train")
    if bool(cfg.get("include_pool", False)):
        ds_pool = K562FullDataset(data_path=str(cfg.k562_data_path), split="pool")
        train_dataset = torch.utils.data.ConcatDataset([ds_train, ds_pool])
    else:
        train_dataset = ds_train
    val_dataset = K562FullDataset(data_path=str(cfg.k562_data_path), split="val")

    def collate_fn_train(batch: list[tuple]) -> dict[str, np.ndarray]:
        return collate_k562_full(batch, int(cfg.max_seq_len), augment=True)

    def collate_fn_eval(batch: list[tuple]) -> dict[str, np.ndarray]:
        return collate_k562_full(batch, int(cfg.max_seq_len), augment=False)

    subset_fraction = cfg.get("subset_fraction", None)
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
        collate_fn=collate_fn_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_fn_eval,
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

    best_val_pearson = -1.0

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
