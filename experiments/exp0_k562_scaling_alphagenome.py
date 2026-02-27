#!/usr/bin/env python
"""Experiment 0 (K562, AlphaGenome): random downsampling scaling curve.

Trains a frozen-encoder AlphaGenome head (boda-flatten-512-512) on random
subsets of the hashFrag train+pool split (~320 K sequences) and evaluates
each run on the four K562 test sets:
  - In-distribution hashFrag test set
  - SNV absolute expression (alt-allele only)
  - SNV delta variant effect (alt-ref)
  - OOD CRE enhancer set

Run via SLURM array (one task per fraction):
  sbatch scripts/slurm/exp0_k562_scaling_alphagenome.sh
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
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

# ── MPRA flanks (200 bp each, taken from K562FullDataset constants) ──────────
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


# ── Helpers ──────────────────────────────────────────────────────────────────


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    return seed


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


def _build_600bp(seq_tensor: torch.Tensor, max_shift: int = 0, augment: bool = False) -> np.ndarray:
    """Convert a (5, 200) K562Dataset tensor to a (600, 4) MPRA-context array."""
    core = seq_tensor.numpy()[:4, :].T  # (200, 4)
    full_seq = np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)  # (600, 4)
    if augment:
        if np.random.rand() > 0.5:
            full_seq = full_seq[::-1, ::-1]
        if max_shift > 0 and np.random.rand() > 0.5:
            shift = np.random.randint(-max_shift, max_shift + 1)
            full_seq = np.roll(full_seq, shift, axis=0)
    return full_seq


def collate_train(batch: list[tuple], max_shift: int = 15) -> dict[str, np.ndarray]:
    B = len(batch)
    x = np.zeros((B, 600, 4), dtype=np.float32)
    y = np.zeros((B,), dtype=np.float32)
    for i, (seq, label) in enumerate(batch):
        x[i] = _build_600bp(seq, max_shift=max_shift, augment=True)
        y[i] = float(label.numpy())
    return {"sequences": x, "targets": y, "organism_index": np.zeros(B, dtype=np.int32)}


def collate_eval(batch: list[tuple]) -> dict[str, np.ndarray]:
    B = len(batch)
    x = np.zeros((B, 600, 4), dtype=np.float32)
    y = np.zeros((B,), dtype=np.float32)
    for i, (seq, label) in enumerate(batch):
        x[i] = _build_600bp(seq, augment=False)
        y[i] = float(label.numpy())
    return {"sequences": x, "targets": y, "organism_index": np.zeros(B, dtype=np.int32)}


# ── Test-set evaluation ───────────────────────────────────────────────────────


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
    """RC-averaged predictions on raw 200bp sequence strings."""
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


def evaluate_test_sets(
    predict_step_fn,
    model_params,
    model_state,
    test_set_dir: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate on in_distribution, snv_abs (alt-only), snv_delta, and OOD."""
    metrics: dict[str, dict[str, float]] = {}

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        in_df = pd.read_csv(in_path, sep="\t")
        in_pred = _predict_sequences(
            predict_step_fn, model_params, model_state, in_df["sequence"].tolist()
        )
        in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["in_distribution"] = {
            "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
            "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
            "n": int(len(in_true)),
        }

    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        ref_pred = _predict_sequences(
            predict_step_fn, model_params, model_state, snv_df["sequence_ref"].tolist()
        )
        alt_pred = _predict_sequences(
            predict_step_fn, model_params, model_state, snv_df["sequence_alt"].tolist()
        )
        # snv_abs: alt-allele only (ref overlaps in-distribution test set).
        alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
        metrics["snv_abs"] = {
            "pearson_r": _safe_corr(alt_pred, alt_true, pearsonr),
            "spearman_r": _safe_corr(alt_pred, alt_true, spearmanr),
            "mse": float(np.mean((alt_pred - alt_true) ** 2)),
            "n": int(len(alt_true)),
        }
        delta_pred = alt_pred - ref_pred
        delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_pred, delta_true, pearsonr),
            "spearman_r": _safe_corr(delta_pred, delta_true, spearmanr),
            "mse": float(np.mean((delta_pred - delta_true) ** 2)),
            "n": int(len(delta_true)),
        }

    ood_path = test_set_dir / "test_ood_cre.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_pred = _predict_sequences(
            predict_step_fn, model_params, model_state, ood_df["sequence"].tolist()
        )
        ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["ood"] = {
            "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
            "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
            "mse": float(np.mean((ood_pred - ood_true) ** 2)),
            "n": int(len(ood_true)),
        }

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="exp0_k562_scaling_alphagenome",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    fraction = float(cfg.fraction)
    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)

    output_dir = (
        Path(str(cfg.output_dir)).expanduser().resolve()
        / f"fraction_{fraction:.4f}"
        / f"seed_{used_seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    max_shift = int(cfg.get("max_shift", 15))
    dropout_rate = float(cfg.get("dropout_rate", 0.1))

    wandb.init(
        project="albench-s2f",
        name=f"exp0_ag_k562_frac{fraction:.3f}_seed{used_seed}",
        config={**OmegaConf.to_container(cfg, resolve=True), "fraction": fraction},
        tags=["exp0", "k562", "scaling", "alphagenome"],
        mode=str(cfg.wandb_mode),
        job_type="exp0_scaling",
    )

    arch_slug = str(cfg.head_arch).replace("-", "_")
    unique_head_name = f"{cfg.head_name}_{arch_slug}_v4"

    register_s2f_head(
        head_name=unique_head_name,
        arch=str(cfg.head_arch),
        task_mode="human",
        num_tracks=int(cfg.num_tracks),
        dropout_rate=dropout_rate,
    )

    weights_path = str(Path(str(cfg.weights_path)).expanduser().resolve())
    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )
    reinit_head_params(model, unique_head_name, num_tokens=5, dim=1536, rng=used_seed)
    model.freeze_except_head(unique_head_name)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    print(f"Parameters: {param_count:,}", flush=True)

    loss_fn = model.create_loss_fn_for_head(unique_head_name)
    optimizer = optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    opt_state = optimizer.init(model._params)

    # ── Datasets: train+pool subset ──────────────────────────────────────────
    ds_train = K562Dataset(data_path=str(cfg.k562_data_path), split="train")
    ds_pool = K562Dataset(data_path=str(cfg.k562_data_path), split="pool")
    ds_val = K562Dataset(data_path=str(cfg.k562_data_path), split="val")
    full_train = ConcatDataset([ds_train, ds_pool])

    n_total = len(full_train)
    n_samples = max(1, int(n_total * fraction))
    rng_subset = np.random.RandomState(used_seed)
    subset_idx = rng_subset.choice(n_total, size=n_samples, replace=False)
    train_subset = Subset(full_train, subset_idx.tolist())

    print(
        f"Fraction {fraction:.3f}: {n_samples:,}/{n_total:,} training sequences "
        f"| Val: {len(ds_val):,}",
        flush=True,
    )

    n_workers = int(cfg.num_workers)

    def _collate_train(batch):
        return collate_train(batch, max_shift=max_shift)

    train_loader = DataLoader(
        train_subset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=n_workers,
        collate_fn=_collate_train,
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=n_workers,
        collate_fn=collate_eval,
        pin_memory=True,
        persistent_workers=n_workers > 0,
    )

    # ── JIT functions ────────────────────────────────────────────────────────
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
        return optax.apply_updates(params, updates), next_opt_state, loss

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[unique_head_name]

    # ── Training loop ────────────────────────────────────────────────────────
    from tqdm import tqdm

    best_val_pearson = -1.0
    early_stop_patience = int(cfg.get("early_stop_patience", 5))
    epochs_no_improve = 0

    for epoch in range(int(cfg.epochs)):
        train_losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{int(cfg.epochs)}")
        for batch in pbar:
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            model._params, opt_state, loss = train_step(model._params, opt_state, batch_jax)
            loss_v = float(loss)
            train_losses.append(loss_v)
            pbar.set_postfix({"loss": f"{loss_v:.4f}"})

        # Validation
        y_true_all, y_pred_all = [], []
        for batch in val_loader:
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            preds = np.array(
                predict_step(model._params, model._state, batch_jax["sequences"])
            ).reshape(-1)
            y_pred_all.append(preds)
            y_true_all.append(np.array(batch["targets"]).reshape(-1))

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        pear = _safe_corr(y_true, y_pred, pearsonr)
        spear = _safe_corr(y_true, y_pred, spearmanr)

        print(
            f"Epoch {epoch + 1}: train_loss={avg_train:.4f}  val_pearson={pear:.4f}  "
            f"val_spearman={spear:.4f}",
            flush=True,
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": avg_train,
                "val/pearson_r": pear,
                "val/spearman_r": spear,
                "fraction": fraction,
                "n_samples": n_samples,
            }
        )

        if pear > best_val_pearson:
            best_val_pearson = pear
            epochs_no_improve = 0
            model.save_checkpoint(str(output_dir / "best_model"), save_full_model=False)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val Pearson={best_val_pearson:.4f})",
                    flush=True,
                )
                break

        model.save_checkpoint(str(output_dir / "last_model"), save_full_model=False)

    # ── Post-training test evaluation ────────────────────────────────────────
    print("\n[eval] Loading best checkpoint for test evaluation …", flush=True)

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

    ckpt_path = output_dir / "best_model" / "checkpoint"
    if ckpt_path.exists():
        checkpointer = ocp.StandardCheckpointer()
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(_merge(model._params, loaded_params))
    else:
        print("[eval] No best_model checkpoint — using final weights.", flush=True)

    test_set_dir = Path(str(cfg.k562_data_path)) / "test_sets"
    test_metrics = evaluate_test_sets(predict_step, model._params, model._state, test_set_dir)

    results = {
        "fraction": fraction,
        "n_samples": n_samples,
        "n_total": n_total,
        "seed": used_seed,
        "best_val_pearson": best_val_pearson,
        "head_arch": str(cfg.head_arch),
        "test_metrics": test_metrics,
    }

    out_json = output_dir / "result.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Wrote {out_json}", flush=True)

    for ts_name, m in test_metrics.items():
        wandb.log(
            {
                f"test/{ts_name}/pearson_r": m.get("pearson_r", 0.0),
                f"test/{ts_name}/spearman_r": m.get("spearman_r", 0.0),
                "fraction": fraction,
                "n_samples": n_samples,
            }
        )
        print(
            f"[eval] {ts_name}: pearson_r={m.get('pearson_r', 0.0):.4f}  "
            f"spearman_r={m.get('spearman_r', 0.0):.4f}",
            flush=True,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
