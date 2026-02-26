#!/usr/bin/env python
"""Train AlphaGenome oracle on hashFrag K562 splits (10-seed ensemble for AL experiments).

Trains a frozen-encoder AlphaGenome head on the hashFrag train split (~100 K sequences)
and evaluates on:
  - In-distribution hashFrag test set
  - SNV pairs hashFrag test set (absolute expression + delta variant effect)
  - OOD CRE test set

Uses full RC + shift augmentation (aug_mode=full) with detach_backbone=True so the
encoder is properly frozen throughout. Best configuration from architecture search:
boda-flatten-512-512, dropout=0.1, lr=0.001.

Run 10 seeds for ensemble oracle use in AL experiments:

  for seed in 42 43 44 45 46 47 48 49 50 51; do
    uv run python experiments/train_oracle_alphagenome_hashfrag.py \\
        ++seed=$seed ++output_dir=outputs/ag_hashfrag_oracle/seed_$seed
  done
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

# ── MPRA flanks (200 bp each, taken from K562FullDataset constants) ─────────
# 5' flank: last 200 bp of the 300 bp MPRA_UPSTREAM sequence
# 3' flank: first 200 bp of the 300 bp MPRA_DOWNSTREAM sequence
_FLANK_5_STR: str = MPRA_UPSTREAM[-200:]
_FLANK_3_STR: str = MPRA_DOWNSTREAM[:200]

# Pre-encode flanks once at module load time
_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}

_FLANK_5_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(_FLANK_5_STR):
    if _c in _MAPPING:
        _FLANK_5_ENC[_i, _MAPPING[_c]] = 1.0

_FLANK_3_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(_FLANK_3_STR):
    if _c in _MAPPING:
        _FLANK_3_ENC[_i, _MAPPING[_c]] = 1.0


# ── Simple dataset for raw sequence strings ───────────────────────────────────


class RawStringDataset(Dataset):
    """Minimal dataset wrapping raw 200 bp sequence strings and float32 labels.

    Returns ``(tensor(5, 200), tensor(scalar))`` tuples that are fully
    compatible with the existing ``collate_train`` / ``collate_eval`` functions
    (which only use the first 4 channels of the 5-channel tensor).
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = sequences
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        seq_str = str(self.sequences[idx]).upper()
        # Ensure exactly 200 bp (mirrors K562Dataset._standardize_to_200bp)
        if len(seq_str) < 200:
            pad = 200 - len(seq_str)
            seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
        elif len(seq_str) > 200:
            start = (len(seq_str) - 200) // 2
            seq_str = seq_str[start : start + 200]
        # One-hot encode to (4, 200); append zero RC channel → (5, 200)
        enc = np.zeros((5, 200), dtype=np.float32)
        for j, c in enumerate(seq_str):
            if c in _MAPPING:
                enc[_MAPPING[c], j] = 1.0
        return torch.tensor(enc, dtype=torch.float32), torch.tensor(self.labels[idx])


# ── Helpers ───────────────────────────────────────────────────────────────────


def set_seed(seed: int | None) -> int:
    """Set deterministic seeds; sample randomly if seed is None."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)
    np.random.seed(seed)
    return seed


def _build_600bp(seq_tensor: torch.Tensor, max_shift: int = 0, augment: bool = False) -> np.ndarray:
    """Convert a (5, 200) K562Dataset tensor to a (600, 4) MPRA-context array.

    Adds 200 bp MPRA flanks on each side.  With ``augment=True`` applies a
    random RC flip (50 %) and a random shift of up to ``max_shift`` positions.
    """
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
    """Collate K562Dataset batches to 600 bp with RC + shift augmentation."""
    B = len(batch)
    x = np.zeros((B, 600, 4), dtype=np.float32)
    y = np.zeros((B,), dtype=np.float32)
    for i, (seq, label) in enumerate(batch):
        x[i] = _build_600bp(seq, max_shift=max_shift, augment=True)
        y[i] = float(label.numpy())
    return {"sequences": x, "targets": y, "organism_index": np.zeros(B, dtype=np.int32)}


def collate_eval(batch: list[tuple]) -> dict[str, np.ndarray]:
    """Collate K562Dataset batches to 600 bp without augmentation."""
    B = len(batch)
    x = np.zeros((B, 600, 4), dtype=np.float32)
    y = np.zeros((B,), dtype=np.float32)
    for i, (seq, label) in enumerate(batch):
        x[i] = _build_600bp(seq, augment=False)
        y[i] = float(label.numpy())
    return {"sequences": x, "targets": y, "organism_index": np.zeros(B, dtype=np.int32)}


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


# ── Test-set evaluation (600 bp context) ─────────────────────────────────────


def _seq_str_to_600bp(seq_str: str) -> np.ndarray:
    """Encode a raw 200 bp sequence string to a (600, 4) MPRA-context array."""
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
    """Run RC-averaged predictions on a list of raw sequence strings using 600 bp context."""
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
    head_name: str,
    predict_step_fn,
    test_set_dir: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate on all 3 K562 test sets using 600 bp MPRA context.

    Returns metrics for: in_distribution, snv_abs, snv_delta, ood.
    """
    params, state = model._params, model._state
    metrics: dict[str, dict[str, float]] = {}

    # ── In-distribution ──────────────────────────────────────────────────────
    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        in_df = pd.read_csv(in_path, sep="\t")
        in_pred = _predict_sequences(predict_step_fn, params, state, in_df["sequence"].tolist())
        in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["in_distribution"] = {
            "pearson_r": _safe_corr(in_pred, in_true, pearsonr),
            "spearman_r": _safe_corr(in_pred, in_true, spearmanr),
            "mse": float(np.mean((in_pred - in_true) ** 2)),
            "n": int(len(in_true)),
        }
    else:
        print(f"[eval] Missing {in_path} — skipping in_distribution eval", flush=True)

    # ── SNV pairs ────────────────────────────────────────────────────────────
    snv_path = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        ref_pred = _predict_sequences(
            predict_step_fn, params, state, snv_df["sequence_ref"].tolist()
        )
        alt_pred = _predict_sequences(
            predict_step_fn, params, state, snv_df["sequence_alt"].tolist()
        )
        # snv_abs: alt-allele predictions vs alt-allele truth only.
        # Ref sequences largely overlap in-distribution data; excluding them avoids inflating this metric.
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
    else:
        print(f"[eval] Missing {snv_path} — skipping SNV eval", flush=True)

    # ── OOD ─────────────────────────────────────────────────────────────────
    ood_path = test_set_dir / "test_ood_cre.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_pred = _predict_sequences(predict_step_fn, params, state, ood_df["sequence"].tolist())
        ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["ood"] = {
            "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
            "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
            "mse": float(np.mean((ood_pred - ood_true) ** 2)),
            "n": int(len(ood_true)),
        }
    else:
        print(f"[eval] Missing {ood_path} — skipping OOD eval", flush=True)

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="oracle_alphagenome_k562_hashfrag",
)
def main(cfg: DictConfig) -> None:
    """Train AlphaGenome oracle on hashFrag K562 train split."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_shift = int(cfg.get("max_shift", 15))
    dropout_rate = float(cfg.get("dropout_rate", 0.0))

    wandb.init(
        project="albench-s2f",
        name=f"oracle_ag_hashfrag_{cfg.head_arch}_seed{used_seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["oracle", "alphagenome", "hashfrag", str(cfg.head_arch), "full_aug"],
        mode=str(cfg.wandb_mode),
        job_type="oracle_training",
    )

    # Unique head name to avoid stale checkpoint param matching.
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
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {weights_path}")

    model = create_model_with_heads(
        "all_folds",
        heads=[unique_head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,  # critical: keeps encoder frozen during full_aug training
    )
    # Re-init head params with a seed-derived JAX PRNG key so each oracle
    # run starts from a distinct random initialization.
    reinit_head_params(model, unique_head_name, num_tokens=5, dim=1536, rng=used_seed)
    model.freeze_except_head(unique_head_name)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(model._params))
    print(f"Total parameters: {param_count:,}", flush=True)

    loss_fn = model.create_loss_fn_for_head(unique_head_name)

    if cfg.gradients_clip is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(cfg.gradients_clip)),
            optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay)),
        )
    else:
        optimizer = optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    opt_state = optimizer.init(model._params)

    # ── Datasets ──────────────────────────────────────────────────────────────
    use_all_data = bool(cfg.get("use_all_data", False))

    if use_all_data:
        # Combine ALL K562 hashfrag splits (train + pool + val + test) + synthetic sequences
        # so the oracle learns from every available labeled MPRA sequence.
        all_seqs_list: list[np.ndarray] = []
        all_labs_list: list[np.ndarray] = []
        for split_name in ["train", "pool", "val", "test"]:
            ds_split = K562Dataset(data_path=str(cfg.k562_data_path), split=split_name)
            all_seqs_list.append(ds_split.sequences)
            all_labs_list.append(ds_split.labels)
            print(f"  Loaded {split_name}: {len(ds_split.sequences):,}", flush=True)

        # Add synthetic sequences (designed sequences with measured K562_log2FC labels).
        # cre_sequences.tsv is excluded — it is the OOD test set (preserve for honest eval).
        synth_path = Path(str(cfg.k562_data_path)) / "test_sets" / "synthetic_sequences.tsv"
        if synth_path.exists():
            synth_df = pd.read_csv(synth_path, sep="\t")
            all_seqs_list.append(synth_df["sequence"].to_numpy())
            all_labs_list.append(synth_df["K562_log2FC"].to_numpy(dtype=np.float32))
            print(f"  Loaded synthetic: {len(synth_df):,}", flush=True)

        all_seqs = np.concatenate(all_seqs_list)
        all_labs = np.concatenate(all_labs_list)

        # Random 5% holdout for early-stopping / val monitoring.
        # Seed fixed so the split is reproducible across restarts.
        rng_split = np.random.default_rng(seed=42)
        n_total = len(all_seqs)
        val_idx = rng_split.choice(n_total, size=int(n_total * 0.05), replace=False)
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[val_idx] = False

        ds_train = RawStringDataset(all_seqs[train_mask], all_labs[train_mask])
        ds_val = RawStringDataset(all_seqs[val_idx], all_labs[val_idx])
        print(
            f"Full-data oracle — Train: {len(ds_train):,} | Val (5% holdout): {len(ds_val):,}",
            flush=True,
        )
    else:
        ds_train = K562Dataset(data_path=str(cfg.k562_data_path), split="train")
        ds_val = K562Dataset(data_path=str(cfg.k562_data_path), split="val")
        print(f"Train: {len(ds_train):,} | Val: {len(ds_val):,}", flush=True)

    n_workers = int(cfg.num_workers)

    def _collate_train(batch):
        return collate_train(batch, max_shift=max_shift)

    train_loader = DataLoader(
        ds_train,
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

    # ── JIT functions ─────────────────────────────────────────────────────────
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

    # ── Training loop ─────────────────────────────────────────────────────────
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

        # ── Validation ────────────────────────────────────────────────────────
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
                    f"Early stopping: no improvement for {early_stop_patience} epochs "
                    f"(best val Pearson={best_val_pearson:.4f})",
                    flush=True,
                )
                break

        model.save_checkpoint(str(output_dir / "last_model"), save_full_model=False)

    # ── Post-training evaluation on all 3 test sets ───────────────────────────
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
        print("[eval] No best_model checkpoint found — using final weights.", flush=True)

    test_set_dir = Path(str(cfg.k562_data_path)) / "test_sets"
    test_metrics = evaluate_all_test_sets(model, unique_head_name, predict_step, test_set_dir)

    results = {
        "seed": used_seed,
        "best_val_pearson": best_val_pearson,
        "head_arch": str(cfg.head_arch),
        "head_name": unique_head_name,
        "test_metrics": test_metrics,
    }

    out_json = output_dir / "test_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Wrote {out_json}", flush=True)

    for test_set, m in test_metrics.items():
        wandb.log({f"test/{test_set}/pearson_r": m.get("pearson_r", 0.0)})
        print(
            f"[eval] {test_set}: pearson_r={m.get('pearson_r', 0.0):.4f}  "
            f"spearman_r={m.get('spearman_r', 0.0):.4f}  mse={m.get('mse', 0.0):.4f}",
            flush=True,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
