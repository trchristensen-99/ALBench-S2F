#!/usr/bin/env python
"""Train AlphaGenome oracle on hashFrag K562 splits using precomputed embedding cache.

Trains only the small head network on pre-computed encoder embeddings so the
expensive encoder forward pass is never called during training.  Uses the combined
train split (~320K sequences).  RC augmentation is preserved (canonical and RC
embeddings are both included in every epoch).  No shift augmentation.  Typical
speed-up over full-encoder training: 20–50×.

Cache must be pre-built (train / val canonical + RC) before running this script::

    sbatch scripts/slurm/build_hashfrag_embedding_cache.sh

After training the best checkpoint is loaded and the model is evaluated on the
three hashFrag test sets (in_distribution, SNV pairs, OOD designed) using the full
encoder, matching the evaluation protocol in ``train_oracle_alphagenome_hashfrag.py``.

Run 10 seeds for oracle ensemble::

    sbatch scripts/slurm/train_oracle_alphagenome_hashfrag_cached_array.sh
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
import pandas as pd
import wandb
from alphagenome_ft import create_model_with_heads
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import (
    build_head_only_predict_fn,
    build_head_only_train_fn,
    load_embedding_cache,
    reinit_head_params,
)

# ── Cell-line label mapping ──────────────────────────────────────────────────
CELL_LINE_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}

# ── MPRA flanks for 600 bp test-set evaluation ───────────────────────────────
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


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


# ── Test-set evaluation helpers (full encoder) ───────────────────────────────


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
    """RC-averaged predictions on raw 200 bp sequence strings via 600 bp context."""
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
    params, state = model._params, model._state
    metrics: dict[str, dict[str, float]] = {}
    fc_col = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")

    in_path = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_path.exists():
        in_df = pd.read_csv(in_path, sep="\t")
        in_pred = _predict_sequences(predict_step_fn, params, state, in_df["sequence"].tolist())
        in_true = in_df[fc_col].to_numpy(dtype=np.float32)
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
            predict_step_fn, params, state, snv_df["sequence_ref"].tolist()
        )
        alt_pred = _predict_sequences(
            predict_step_fn, params, state, snv_df["sequence_alt"].tolist()
        )
        alt_col = f"{fc_col}_alt"
        if alt_col not in snv_df.columns:
            alt_col = "K562_log2FC_alt"
        alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
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

    # OOD test set is cell-line-specific; skip if file doesn't exist
    ood_path = test_set_dir / f"test_ood_designed_{cell_line}.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_pred = _predict_sequences(predict_step_fn, params, state, ood_df["sequence"].tolist())
        if fc_col in ood_df.columns:
            ood_true = ood_df[fc_col].to_numpy(dtype=np.float32)
        else:
            ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
        metrics["ood"] = {
            "pearson_r": _safe_corr(ood_pred, ood_true, pearsonr),
            "spearman_r": _safe_corr(ood_pred, ood_true, spearmanr),
            "mse": float(np.mean((ood_pred - ood_true) ** 2)),
            "n": int(len(ood_true)),
        }

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="oracle_alphagenome_k562_hashfrag_cached",
)
def main(cfg: DictConfig) -> None:
    """Train AlphaGenome oracle on hashFrag K562 train split using embedding cache."""
    load_dotenv()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))

    used_seed = set_seed(int(cfg.seed) if cfg.seed is not None else None)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dropout_rate = float(cfg.get("dropout_rate", 0.0))

    wandb.init(
        project="albench-s2f",
        name=f"oracle_ag_hashfrag_cached_{cfg.head_arch}_seed{used_seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["oracle", "alphagenome", "hashfrag", "cached", str(cfg.head_arch), "no_shift"],
        mode=str(cfg.wandb_mode),
        job_type="oracle_training",
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
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {weights_path}")

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
    print(f"Total parameters: {param_count:,}", flush=True)

    # ── Load labels and embedding cache ──────────────────────────────────────
    # split="train" contains all ~320K hashFrag training sequences.
    cell_line = str(cfg.get("cell_line", "k562"))
    label_column = CELL_LINE_LABEL_COLS.get(cell_line, "K562_log2FC")
    ds_all_train = K562Dataset(
        data_path=str(cfg.k562_data_path), split="train", label_column=label_column
    )
    all_labels = ds_all_train.labels.astype(np.float32)
    print(f"  Total training labels: {len(all_labels):,}", flush=True)

    cache_dir = Path(str(cfg.cache_dir)).expanduser().resolve()
    print(f"Loading embedding cache from {cache_dir} …", flush=True)
    all_canonical, all_rc = load_embedding_cache(cache_dir, "train")
    print(f"  All embeddings: {all_canonical.shape}", flush=True)

    # ── 10-fold CV split ──────────────────────────────────────────────────────
    n_folds = int(cfg.get("n_folds", 10))
    fold_id = int(cfg.get("fold_id", 0))
    n_total = len(all_labels)
    perm = np.random.default_rng(seed=42).permutation(n_total)
    fold_size = n_total // n_folds
    val_start = fold_id * fold_size
    val_end = val_start + fold_size if fold_id < n_folds - 1 else n_total
    val_idx = perm[val_start:val_end]
    train_idx = np.concatenate([perm[:val_start], perm[val_end:]])

    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]
    train_canonical = all_canonical[train_idx]
    train_rc = all_rc[train_idx]
    val_canonical = all_canonical[val_idx]
    val_rc = all_rc[val_idx]
    N_train = len(train_idx)
    N_val = len(val_idx)
    print(
        f"K-fold oracle (fold {fold_id}/{n_folds}) — Train: {N_train:,} | Val: {N_val:,}",
        flush=True,
    )

    # ── Head-only JIT functions ───────────────────────────────────────────────
    head_predict_fn = build_head_only_predict_fn(model, unique_head_name)
    head_train_fn = (
        build_head_only_train_fn(model, unique_head_name) if dropout_rate > 0.0 else None
    )

    # Verify shapes
    _dummy_emb = jnp.zeros((2, 5, 1536), dtype=jnp.float32)
    _dummy_org = jnp.zeros((2,), dtype=jnp.int32)
    _ = head_predict_fn(model._params, _dummy_emb, _dummy_org)
    print("Head-only predict function compiled OK.", flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optax.adamw(learning_rate=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    opt_state = optimizer.init(model._params)

    # ── JIT steps ─────────────────────────────────────────────────────────────
    @jax.jit
    def cached_train_step(params, current_opt_state, rng, encoder_output, targets, organism_index):
        def loss_func(p):
            if head_train_fn is not None:
                preds = head_train_fn(p, rng, encoder_output, organism_index)
            else:
                preds = head_predict_fn(p, encoder_output, organism_index)
            pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
            return jnp.mean((pred - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    @jax.jit
    def cached_eval_step(params, encoder_output, organism_index):
        preds = head_predict_fn(params, encoder_output, organism_index)
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    # Full-encoder predict for test evaluation (loaded + used only after training)
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

    # ── Training loop (no_shift: canonical + RC pass per batch) ───────────────
    best_val_pearson = -1.0
    early_stop_patience = int(cfg.get("early_stop_patience", 5))
    epochs_no_improve = 0
    rng = jax.random.PRNGKey(used_seed)
    batch_size = int(cfg.batch_size)

    for epoch in range(int(cfg.epochs)):
        perm = np.random.permutation(N_train)
        train_losses: list[float] = []

        pbar = tqdm(
            range(0, N_train, batch_size),
            desc=f"Epoch {epoch + 1}/{int(cfg.epochs)}",
            total=(N_train + batch_size - 1) // batch_size,
        )
        for start in pbar:
            indices = perm[start : start + batch_size]
            targets_jax = jnp.array(train_labels[indices])
            org_idx = jnp.zeros(len(indices), dtype=jnp.int32)

            # Canonical pass
            rng, step_rng = jax.random.split(rng)
            emb_can = jnp.array(train_canonical[indices].astype(np.float32))
            model._params, opt_state, loss = cached_train_step(
                model._params, opt_state, step_rng, emb_can, targets_jax, org_idx
            )
            train_losses.append(float(loss))

            # RC pass (same labels, different embeddings)
            rng, step_rng = jax.random.split(rng)
            emb_rc = jnp.array(train_rc[indices].astype(np.float32))
            model._params, opt_state, loss = cached_train_step(
                model._params, opt_state, step_rng, emb_rc, targets_jax, org_idx
            )
            loss_v = float(loss)
            train_losses.append(loss_v)
            pbar.set_postfix({"loss": f"{loss_v:.4f}"})

        # ── Validation (RC-averaged for accurate early stopping) ────────────
        y_pred_all: list[np.ndarray] = []
        for start in range(0, N_val, batch_size):
            end = min(start + batch_size, N_val)
            indices = np.arange(start, end, dtype=np.int64)
            org_idx = jnp.zeros(len(indices), dtype=jnp.int32)
            emb_can = jnp.array(val_canonical[indices].astype(np.float32))
            emb_rc = jnp.array(val_rc[indices].astype(np.float32))
            preds_can = cached_eval_step(model._params, emb_can, org_idx)
            preds_rc = cached_eval_step(model._params, emb_rc, org_idx)
            y_pred_all.append(
                (np.array(preds_can).reshape(-1) + np.array(preds_rc).reshape(-1)) / 2.0
            )

        y_pred = np.concatenate(y_pred_all)
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        pear = _safe_corr(val_labels, y_pred, pearsonr)
        spear = _safe_corr(val_labels, y_pred, spearmanr)

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

        # Skip last_model save to conserve disk (best_model is sufficient)

    # ── Post-training evaluation on test sets (full encoder) ─────────────────
    print("\n[eval] Loading best checkpoint for test evaluation …", flush=True)

    import orbax.checkpoint as ocp

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
    test_metrics = evaluate_all_test_sets(model, predict_step, test_set_dir, cell_line=cell_line)

    results = {
        "seed": used_seed,
        "best_val_pearson": best_val_pearson,
        "head_arch": str(cfg.head_arch),
        "head_name": unique_head_name,
        "aug_mode": "no_shift_cached",
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
