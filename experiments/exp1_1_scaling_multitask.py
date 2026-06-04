"""AG S2 joint multi-head fine-tuning (K562 + HepG2 + SKNSH).

Trains ONE AG encoder with THREE S2F heads (K562/HepG2/SKNSH), each registered
via ``register_s2f_head`` with ``num_tracks=1``. Encoder is shared and
fine-tuned jointly (top blocks unfrozen). Loss is a NaN-aware sum of per-cell
MSE losses (entries with NaN target for a given cell are ignored).

Best HPs (from K562 single-cell sweep):
  encoder_lr=1.5e-4, head_lr=5e-4, unfreeze_blocks=[0..5],
  warmup_epochs=3, cosine LR after warmup.

Test evaluation runs per cell on chr 7+13 (Reference, Alt, OOD, SNV pairs)
using the existing K562Dataset (chr_split) infrastructure.

Run:
    python experiments/exp1_1_scaling_multitask.py \
        --task k562 --student alphagenome_k562_s2_multitask \
        --reservoir genomic --multitask --chr-split \
        --seed 42 \
        --s1-checkpoint outputs/chr_split_v2/k562/ag_s1_lc/lr1e-4_bs512/genomic/n658000/hp0/seed42/best_model \
        --output-dir outputs/chr_split_v2/joint_multitask/ag_s2/seed_42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logger = logging.getLogger("exp1_1_multitask")


# ---------------------------------------------------------------------------
# Config — mirrors exp1_1_scaling.py S2_CONFIG but with joint multitask
# HPs taken from best single-cell K562 S2 run.
# ---------------------------------------------------------------------------
JOINT_S2_CONFIG = {
    "encoder_lr": 1.5e-4,
    "head_lr": 5.0e-4,
    "weight_decay": 1e-6,
    "unfreeze_blocks": [0, 1, 2, 3, 4, 5],
    "warmup_epochs": 3,
    "epochs": 30,
    "early_stop_patience": 7,
    "max_shift": 15,
    "dropout": 0.1,
    "cosine_lr": True,
}

CELL_LINES = ["k562", "hepg2", "sknsh"]
CELL_LINE_LABEL_COLUMNS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}


@dataclass
class MultitaskRunResult:
    student: str
    reservoir: str
    seed: int
    n_train: int
    val_pearson_per_cell: dict
    val_pearson_mean: float
    test_metrics_per_cell: dict  # {cell: {split: {metric: val}}}


# ---------------------------------------------------------------------------
# Sequence encoding (one-hot with MPRA flanks, identical to exp1_1_scaling)
# ---------------------------------------------------------------------------
def _build_encoder():
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _encode_one(seq_str: str) -> np.ndarray:
        seq_str = seq_str.upper()
        if len(seq_str) < 200:
            pad = 200 - len(seq_str)
            seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
        elif len(seq_str) > 200:
            start = (len(seq_str) - 200) // 2
            seq_str = seq_str[start : start + 200]
        full = flank_5 + seq_str + flank_3
        out = np.zeros((600, 4), dtype=np.float32)
        for i, c in enumerate(full):
            if c in mapping:
                out[i, mapping[c]] = 1.0
        return out

    return _encode_one


# ---------------------------------------------------------------------------
# Data loading — chr-split K562/HepG2/SKNSH labels aligned by sequence
# ---------------------------------------------------------------------------
def _load_joint_train_val(
    chr_split: bool,
    include_alt_alleles: bool,
    seed: int,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Load joint (K562, HepG2, SKNSH) train + val sets with chr-based split.

    Returns:
        (train_seqs, train_labels[N, 3], val_seqs, val_labels[M, 3])
        Labels are float32 with NaN for missing entries.
    """
    from data.k562 import K562Dataset

    train_ds_dict = {}
    val_ds_dict = {}
    for cl, col in CELL_LINE_LABEL_COLUMNS.items():
        train_ds_dict[cl] = K562Dataset(
            data_path=str(REPO / "data" / "k562"),
            split="train",
            label_column=col,
            include_alt_alleles=include_alt_alleles,
        )
        val_ds_dict[cl] = K562Dataset(
            data_path=str(REPO / "data" / "k562"),
            split="val",
            label_column=col,
            include_alt_alleles=include_alt_alleles,
        )

    # Sequences are identical across cell lines (same dataset, different label columns)
    train_seqs = list(train_ds_dict["k562"].sequences)
    val_seqs = list(val_ds_dict["k562"].sequences)

    n_train = len(train_seqs)
    n_val = len(val_seqs)
    train_labels = np.full((n_train, 3), np.nan, dtype=np.float32)
    val_labels = np.full((n_val, 3), np.nan, dtype=np.float32)
    for i, cl in enumerate(CELL_LINES):
        train_labels[:, i] = train_ds_dict[cl].labels.astype(np.float32)
        val_labels[:, i] = val_ds_dict[cl].labels.astype(np.float32)

    logger.info(f"Joint dataset (chr_split={chr_split}): train={n_train:,}, val={n_val:,}")
    for i, cl in enumerate(CELL_LINES):
        n_nan_tr = int(np.isnan(train_labels[:, i]).sum())
        n_nan_v = int(np.isnan(val_labels[:, i]).sum())
        logger.info(
            f"  {cl}: train NaN={n_nan_tr}/{n_train} ({100 * n_nan_tr / n_train:.1f}%), "
            f"val NaN={n_nan_v}/{n_val}"
        )
    return train_seqs, train_labels, val_seqs, val_labels


# ---------------------------------------------------------------------------
# Per-cell test evaluation (chr 7+13 in-dist, SNV pairs, OOD)
# ---------------------------------------------------------------------------
def _evaluate_per_cell_test(
    predict_per_cell_fn,  # (sequences: list[str]) -> dict[str, np.ndarray] mapping cell -> preds
    chr_split: bool,
    include_alt_alleles: bool,
    save_predictions_dir: Path | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-cell test eval on in-dist (chr7+13), SNV pairs, OOD.

    Saves predictions to ``test_predictions.npz`` if ``save_predictions_dir`` set:
      in_dist_pred_{cell}, snv_ref_pred_{cell}, snv_alt_pred_{cell}, ood_pred_{cell}.
    """
    from data.k562 import K562Dataset
    from evaluation.exp1_eval import evaluate_predictions

    per_cell_metrics: dict[str, dict[str, dict[str, float]]] = {cl: {} for cl in CELL_LINES}
    pred_arrays: dict[str, np.ndarray] = {}

    # ── In-distribution (test split of K562 dataset, filtered chr7+13 when chr_split)
    for cl in CELL_LINES:
        try:
            ds = K562Dataset(
                data_path=str(REPO / "data" / "k562"),
                split="test",
                label_column=CELL_LINE_LABEL_COLUMNS[cl],
                include_alt_alleles=include_alt_alleles,
            )
            sequences = list(ds.sequences)
            labels = ds.labels.astype(np.float32)
            preds_dict = predict_per_cell_fn(sequences)
            preds = preds_dict[cl]
            mask = np.isfinite(labels)
            if mask.sum() > 0:
                per_cell_metrics[cl]["in_dist"] = evaluate_predictions(preds[mask], labels[mask])
                logger.info(f"  [{cl}] in_dist: {per_cell_metrics[cl]['in_dist']}")
            pred_arrays[f"in_dist_pred_{cl}"] = preds
            # Save labels for the first cell only (sequences are shared)
            if cl == "k562":
                pred_arrays["in_dist_sequences"] = np.array(sequences, dtype=object)
        except Exception as e:
            logger.error(f"  [{cl}] in_dist eval failed: {e}")
            logger.error(traceback.format_exc())

    # ── SNV pairs ────────────────────────────────────────────────────────
    test_dir = REPO / "data" / "k562" / "test_sets"
    snv_path = test_dir / ("test_snv_pairs.tsv" if chr_split else "test_snv_pairs_hashfrag.tsv")
    if snv_path.exists():
        try:
            snv_df = pd.read_csv(snv_path, sep="\t")
            if chr_split and "IDs_ref" in snv_df.columns:
                test_chrs = {"7", "13", "chr7", "chr13"}
                chroms = snv_df["IDs_ref"].str.split(":", expand=True)[0]
                snv_df = snv_df[chroms.isin(test_chrs)].reset_index(drop=True)
                logger.info(f"  Chr-split SNV filter: kept {len(snv_df)} chr7+13 pairs")

            ref_seqs = snv_df["sequence_ref"].tolist()
            alt_seqs = snv_df["sequence_alt"].tolist()
            ref_preds_dict = predict_per_cell_fn(ref_seqs)
            alt_preds_dict = predict_per_cell_fn(alt_seqs)

            for cl in CELL_LINES:
                label_col = CELL_LINE_LABEL_COLUMNS[cl]
                alt_col_candidates = [f"{label_col}_alt"]
                if cl == "k562":
                    alt_col_candidates.append("K562_log2FC_alt")
                delta_col_candidates = [f"delta_{label_col}"]
                if cl == "k562":
                    delta_col_candidates.append("delta_log2FC")

                pred_arrays[f"snv_ref_pred_{cl}"] = ref_preds_dict[cl]
                pred_arrays[f"snv_alt_pred_{cl}"] = alt_preds_dict[cl]

                # snv_abs: predicted alt vs true alt
                for alt_col in alt_col_candidates:
                    if alt_col in snv_df.columns:
                        alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
                        m = np.isfinite(alt_true)
                        if m.sum() > 0:
                            per_cell_metrics[cl]["snv_abs"] = evaluate_predictions(
                                alt_preds_dict[cl][m], alt_true[m]
                            )
                        break
                # snv_delta
                delta_pred = alt_preds_dict[cl] - ref_preds_dict[cl]
                for delta_col in delta_col_candidates:
                    if delta_col in snv_df.columns:
                        delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
                        m = np.isfinite(delta_true)
                        if m.sum() > 0:
                            per_cell_metrics[cl]["snv_delta"] = evaluate_predictions(
                                delta_pred[m], delta_true[m]
                            )
                        break
                logger.info(
                    f"  [{cl}] snv_abs={per_cell_metrics[cl].get('snv_abs')}, "
                    f"snv_delta={per_cell_metrics[cl].get('snv_delta')}"
                )
        except Exception as e:
            logger.error(f"  SNV eval failed: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"  SNV file not found: {snv_path}")

    # ── OOD designed (per-cell file) ────────────────────────────────────
    for cl in CELL_LINES:
        cell_test_dir = REPO / "data" / cl / "test_sets"
        ood_path = cell_test_dir / f"test_ood_designed_{cl}.tsv"
        if not ood_path.exists():
            ood_path = test_dir / f"test_ood_designed_{cl}.tsv"
        if not ood_path.exists():
            logger.warning(f"  [{cl}] OOD file missing — skipping OOD eval")
            continue
        try:
            ood_df = pd.read_csv(ood_path, sep="\t")
            label_col = CELL_LINE_LABEL_COLUMNS[cl]
            true_col = label_col if label_col in ood_df.columns else None
            if true_col is None and cl == "k562" and "K562_log2FC" in ood_df.columns:
                true_col = "K562_log2FC"
            if true_col is None:
                logger.warning(f"  [{cl}] OOD: no {label_col} column")
                continue
            seqs = ood_df["sequence"].tolist()
            preds_dict = predict_per_cell_fn(seqs)
            preds = preds_dict[cl]
            true = ood_df[true_col].to_numpy(dtype=np.float32)
            m = np.isfinite(true)
            if m.sum() > 0:
                per_cell_metrics[cl]["ood"] = evaluate_predictions(preds[m], true[m])
                logger.info(f"  [{cl}] ood: {per_cell_metrics[cl]['ood']}")
            pred_arrays[f"ood_pred_{cl}"] = preds
        except Exception as e:
            logger.error(f"  [{cl}] OOD eval failed: {e}")
            logger.error(traceback.format_exc())

    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_predictions_dir / "test_predictions.npz"
        np.savez_compressed(save_path, **pred_arrays)
        logger.info(f"  Saved predictions to {save_path} ({len(pred_arrays)} arrays)")

    return per_cell_metrics


# ---------------------------------------------------------------------------
# Joint multi-head AG S2 training
# ---------------------------------------------------------------------------
def train_ag_s2_joint_multihead(
    train_seqs: list[str],
    train_labels: np.ndarray,  # (N, 3) float32 with NaN for missing
    val_seqs: list[str],
    val_labels: np.ndarray,  # (M, 3)
    seed: int,
    s1_checkpoint: str | None,
    batch_size: int = 128,
    config: dict | None = None,
) -> dict[str, Any]:
    """Train AG S2 with 3 separate S2F heads (one per cell line).

    Returns dict with:
      best_params, predict_per_cell_fn (sequences -> {cell: preds}),
      val_pearson_per_cell, val_pearson_mean.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params

    cfg = {**JOINT_S2_CONFIG, **(config or {})}
    encoder_lr = cfg["encoder_lr"]
    head_lr = cfg["head_lr"]
    wd = cfg["weight_decay"]
    unfreeze_blocks = cfg["unfreeze_blocks"]
    warmup_epochs = cfg["warmup_epochs"]
    epochs = cfg["epochs"]
    patience = cfg["early_stop_patience"]
    max_shift_aug = cfg["max_shift"]
    dropout_rate = cfg["dropout"]
    cosine_lr = cfg["cosine_lr"]

    # ── Register 3 heads (one per cell line) ────────────────────────────
    head_names = {cl: f"s2f_joint_s2_{cl}_{seed}" for cl in CELL_LINES}
    for cl in CELL_LINES:
        register_s2f_head(
            head_name=head_names[cl],
            arch="boda-flatten-512-512",
            task_mode="k562",
            num_tracks=1,
            dropout_rate=dropout_rate,
        )

    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_names[cl] for cl in CELL_LINES],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=False,
    )
    for i, cl in enumerate(CELL_LINES):
        reinit_head_params(model, head_names[cl], num_tokens=5, dim=1536, rng=seed + i)

    # ── Warm-start K562 head from S1 checkpoint ─────────────────────────
    if s1_checkpoint:
        s1_path = Path(s1_checkpoint)
        ckpt_path = None
        for candidate in [
            s1_path / "best_model" / "checkpoint",
            s1_path / "checkpoint",
            s1_path,
        ]:
            if candidate.exists() and candidate.is_dir():
                ckpt_path = candidate
                break
        if ckpt_path:
            import orbax.checkpoint as ocp

            checkpointer = ocp.StandardCheckpointer()
            s1_restore = checkpointer.restore(ckpt_path.resolve())
            s1_params = s1_restore[0] if isinstance(s1_restore, (tuple, list)) else s1_restore

            # Find S1 head prefix (any key with exp1_s1 or head_hashfrag)
            s1_prefix = None
            for k in s1_params:
                if "exp1_s1" in k or "head_hashfrag" in k:
                    s1_prefix = k.rsplit("/", 1)[0]
                    break
            # S2 K562 head prefix
            s2_prefix = None
            for k in model._params:
                if head_names["k562"] in k:
                    s2_prefix = k.rsplit("/", 1)[0]
                    break

            if s1_prefix and s2_prefix:
                s2_params = model._params
                n_copied = n_skipped = 0
                for s1_key in list(s1_params.keys()):
                    if s1_key.startswith(s1_prefix):
                        suffix = s1_key[len(s1_prefix) :]
                        s2_key = s2_prefix + suffix
                        if s2_key in s2_params:
                            s1_val = s1_params[s1_key]
                            s2_val = s2_params[s2_key]

                            def _shapes_match(a, b):
                                if hasattr(a, "shape") and hasattr(b, "shape"):
                                    return a.shape == b.shape
                                if isinstance(a, dict) and isinstance(b, dict):
                                    return all(_shapes_match(a[k], b[k]) for k in a if k in b)
                                return True

                            if not _shapes_match(s1_val, s2_val):
                                n_skipped += 1
                                continue
                            s2_params[s2_key] = s1_params[s1_key]
                            n_copied += 1
                model._params = jax.device_put(s2_params)
                logger.info(
                    f"  S2 warm-start (K562 head): copied {n_copied} layers from "
                    f"'{s1_prefix}' -> '{s2_prefix}' (skipped {n_skipped})"
                )
            else:
                logger.warning(
                    f"  S2 warm-start: could not match S1 prefix={s1_prefix} "
                    f"to S2 prefix={s2_prefix}; cold start instead"
                )
        else:
            logger.warning(f"  S2 warm-start: ckpt not found at {s1_path}; cold start")
    else:
        logger.info("  S2: cold start (no S1 checkpoint)")

    # ── Optimizer with differential LRs (head / encoder / frozen) ───────
    unfreeze_set = {f"downres_block_{b}" for b in unfreeze_blocks}
    head_name_strs = list(head_names.values())

    def _label_fn(path, _leaf):
        key_strs = [p.key if hasattr(p, "key") else str(p) for p in path]
        s = "/".join(str(k) for k in key_strs)
        if any(hn in s for hn in head_name_strs):
            return "head"
        if "sequence_encoder" in s:
            for block_name in unfreeze_set:
                if block_name in s:
                    return "encoder"
            return "frozen"
        return "frozen"

    param_labels = jax.tree_util.tree_map_with_path(_label_fn, model._params)

    steps_per_epoch = max(1, len(train_seqs) // batch_size)
    post_warmup_steps = max(1, (epochs - warmup_epochs)) * steps_per_epoch
    if cosine_lr:
        logger.info(f"  S2 cosine LR over {post_warmup_steps} steps")

    def _make_optimizer(enc_lr_val: float):
        if cosine_lr and enc_lr_val > 0:
            enc_schedule = optax.cosine_decay_schedule(
                init_value=enc_lr_val, decay_steps=post_warmup_steps, alpha=0.0
            )
            head_schedule = optax.cosine_decay_schedule(
                init_value=head_lr, decay_steps=post_warmup_steps, alpha=0.0
            )
        else:
            enc_schedule = enc_lr_val
            head_schedule = head_lr
        return optax.multi_transform(
            {
                "head": optax.adamw(learning_rate=head_schedule, weight_decay=wd),
                "encoder": optax.adamw(learning_rate=enc_schedule, weight_decay=wd),
                "frozen": optax.set_to_zero(),
            },
            param_labels,
        )

    optimizer = _make_optimizer(0.0)  # warmup: encoder frozen
    opt_state = optimizer.init(model._params)

    # ── One-hot encoding ────────────────────────────────────────────────
    _encode_one = _build_encoder()
    logger.info(f"  Encoding {len(train_seqs):,} train + {len(val_seqs):,} val sequences...")
    train_onehot = np.stack([_encode_one(s) for s in train_seqs])
    val_onehot = np.stack([_encode_one(s) for s in val_seqs])

    # ── Train + predict steps (jit'd) ───────────────────────────────────
    requested_outputs = head_name_strs

    @jax.jit
    def train_step(params, opt_state_, seqs, targets):
        """targets: (B, 3) with NaN for missing entries per cell."""

        def loss_func(p):
            preds_dict = model._predict(
                p,
                model._state,
                seqs,
                jnp.zeros(len(seqs), dtype=jnp.int32),
                requested_outputs=requested_outputs,
                negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
                strand_reindexing=None,
            )
            total_loss = 0.0
            n_terms = 0.0
            for i, cl in enumerate(CELL_LINES):
                pred = preds_dict[head_names[cl]]
                pred = jnp.squeeze(pred, axis=-1) if pred.ndim > 1 else pred
                target_i = targets[:, i]
                mask = jnp.isfinite(target_i)
                # Replace NaN with 0 in target so multiplication is safe
                target_safe = jnp.where(mask, target_i, 0.0)
                sq = (pred - target_safe) ** 2
                sq = jnp.where(mask, sq, 0.0)
                n_valid = jnp.sum(mask.astype(jnp.float32))
                cell_loss = jnp.where(n_valid > 0, jnp.sum(sq) / jnp.maximum(n_valid, 1.0), 0.0)
                total_loss = total_loss + cell_loss
                n_terms = n_terms + jnp.where(n_valid > 0, 1.0, 0.0)
            return total_loss / jnp.maximum(n_terms, 1.0)

        loss, grads = jax.value_and_grad(loss_func)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state_, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    @jax.jit
    def predict_step(params, seqs):
        preds_dict = model._predict(
            params,
            model._state,
            seqs,
            jnp.zeros(len(seqs), dtype=jnp.int32),
            requested_outputs=requested_outputs,
            negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
            strand_reindexing=None,
        )
        out = {}
        for cl in CELL_LINES:
            p = preds_dict[head_names[cl]]
            out[cl] = jnp.squeeze(p, axis=-1) if p.ndim > 1 else p
        return out

    def _rc_onehot_batch(x):
        rc = x[:, ::-1, :]
        rc = rc.at[:, :, :4].set(rc[:, :, [3, 2, 1, 0]])
        return rc

    def _shift_onehot_batch(x, max_shift):
        if max_shift <= 0:
            return x
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift != 0:
            return jnp.roll(x, shift, axis=1)
        return x

    # ── Training loop ───────────────────────────────────────────────────
    rng_perm = np.random.default_rng(seed + 1)
    n_train = len(train_seqs)
    n_val = len(val_seqs)
    best_val_mean = -float("inf")
    best_val_per_cell: dict[str, float] = {}
    best_params = jax.device_get(model._params)
    patience_counter = 0

    from scipy.stats import pearsonr as _pr

    for epoch in range(epochs):
        if epoch == warmup_epochs:
            logger.info(f"  S2: Unfreezing blocks {unfreeze_blocks} (epoch {epoch + 1})")
            optimizer = _make_optimizer(encoder_lr)
            opt_state = optimizer.init(model._params)

        perm = rng_perm.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            seqs = jnp.array(train_onehot[idx])
            if np.random.rand() > 0.5:
                seqs = _rc_onehot_batch(seqs)
            if max_shift_aug > 0 and np.random.rand() > 0.5:
                seqs = _shift_onehot_batch(seqs, max_shift_aug)
            targets = jnp.array(train_labels[idx])
            model._params, opt_state, loss = train_step(model._params, opt_state, seqs, targets)
            epoch_losses.append(float(loss))

        # ── Validation: RC-averaged Pearson per cell ────────────────────
        val_preds_per_cell = {cl: [] for cl in CELL_LINES}
        for i in range(0, n_val, 64):
            batch_v = jnp.array(val_onehot[i : i + 64])
            p_fwd = predict_step(model._params, batch_v)
            p_rc = predict_step(model._params, _rc_onehot_batch(batch_v))
            for cl in CELL_LINES:
                p_avg = (np.array(p_fwd[cl]) + np.array(p_rc[cl])) / 2.0
                val_preds_per_cell[cl].append(p_avg.reshape(-1))

        val_pearson_per_cell: dict[str, float] = {}
        for i, cl in enumerate(CELL_LINES):
            preds = np.concatenate(val_preds_per_cell[cl])
            true = val_labels[:, i]
            mask = np.isfinite(true)
            if mask.sum() > 10 and np.std(preds[mask]) > 0:
                val_pearson_per_cell[cl] = float(_pr(preds[mask], true[mask])[0])
            else:
                val_pearson_per_cell[cl] = 0.0
        val_mean = float(np.mean(list(val_pearson_per_cell.values())))

        if val_mean > best_val_mean:
            best_val_mean = val_mean
            best_val_per_cell = dict(val_pearson_per_cell)
            best_params = jax.device_get(model._params)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= warmup_epochs:
                logger.info(
                    f"  Early stop at epoch {epoch + 1} (best mean val={best_val_mean:.4f})"
                )
                break

        msg = (
            f"  Epoch {epoch + 1}: train_loss={np.mean(epoch_losses):.4f}, val_mean={val_mean:.4f}"
        )
        for cl in CELL_LINES:
            msg += f", {cl}={val_pearson_per_cell[cl]:.4f}"
        logger.info(msg)

    logger.info(f"  Best val mean Pearson: {best_val_mean:.4f}, per-cell: {best_val_per_cell}")

    # ── Build predict_per_cell_fn (RC-averaged) for test eval ───────────
    def predict_per_cell_fn(sequences: list[str]) -> dict[str, np.ndarray]:
        x = np.stack([_encode_one(s) for s in sequences])
        preds_per_cell: dict[str, list[np.ndarray]] = {cl: [] for cl in CELL_LINES}
        for i in range(0, len(sequences), 64):
            batch_x = jnp.array(x[i : i + 64])
            p_fwd = predict_step(best_params, batch_x)
            p_rc = predict_step(best_params, _rc_onehot_batch(batch_x))
            for cl in CELL_LINES:
                p_avg = (np.array(p_fwd[cl]) + np.array(p_rc[cl])) / 2.0
                preds_per_cell[cl].append(p_avg.reshape(-1))
        return {cl: np.concatenate(preds_per_cell[cl]) for cl in CELL_LINES}

    return {
        "best_params": best_params,
        "predict_per_cell_fn": predict_per_cell_fn,
        "val_pearson_per_cell": best_val_per_cell,
        "val_pearson_mean": best_val_mean,
        "head_names": head_names,
    }


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------
def _save_joint_checkpoint(best_params: dict, run_dir: Path) -> None:
    try:
        import shutil

        import orbax.checkpoint as ocp

        ckpt_dir = (run_dir / "best_model" / "checkpoint").resolve()
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(ckpt_dir, best_params)
        logger.info(f"  Saved joint AG S2 checkpoint to {ckpt_dir}")
    except Exception as e:
        logger.warning(f"  Joint AG S2 ckpt save failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Exp1.1 joint multi-head AG S2 training (K562+HepG2+SKNSH)."
    )
    parser.add_argument(
        "--task", required=True, choices=["k562"], help="Only k562 supported for joint multitask."
    )
    parser.add_argument(
        "--student",
        required=True,
        choices=["alphagenome_k562_s2_multitask"],
        help="Joint multitask AG S2 student.",
    )
    parser.add_argument("--reservoir", default="genomic")
    parser.add_argument(
        "--multitask",
        action="store_true",
        required=True,
        help="Required flag for joint multi-head training (sanity).",
    )
    parser.add_argument("--chr-split", action="store_true", default=True)
    parser.add_argument("--include-alt-alleles", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--s1-checkpoint",
        type=str,
        default=None,
        help="Path to AG S1 K562 checkpoint dir for warm start of K562 head.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encoder-lr", type=float, default=JOINT_S2_CONFIG["encoder_lr"])
    parser.add_argument("--head-lr", type=float, default=JOINT_S2_CONFIG["head_lr"])
    parser.add_argument("--epochs", type=int, default=JOINT_S2_CONFIG["epochs"])
    parser.add_argument(
        "--early-stop-patience", type=int, default=JOINT_S2_CONFIG["early_stop_patience"]
    )
    parser.add_argument("--save-predictions", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Sanity: --multitask must accompany alphagenome_k562_s2_multitask
    if args.student == "alphagenome_k562_s2_multitask" and not args.multitask:
        parser.error("--multitask is required with --student alphagenome_k562_s2_multitask")

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== AG S2 joint multi-head | seed={args.seed} ===")
    logger.info(f"Output: {output_dir}")
    logger.info(f"S1 ckpt: {args.s1_checkpoint}")
    logger.info(f"Config: {JOINT_S2_CONFIG}")

    # ── Load data ───────────────────────────────────────────────────────
    train_seqs, train_labels, val_seqs, val_labels = _load_joint_train_val(
        chr_split=args.chr_split,
        include_alt_alleles=args.include_alt_alleles,
        seed=args.seed,
    )

    # ── Train ───────────────────────────────────────────────────────────
    config = dict(JOINT_S2_CONFIG)
    config["encoder_lr"] = args.encoder_lr
    config["head_lr"] = args.head_lr
    config["epochs"] = args.epochs
    config["early_stop_patience"] = args.early_stop_patience

    t0 = time.perf_counter()
    train_out = train_ag_s2_joint_multihead(
        train_seqs=train_seqs,
        train_labels=train_labels,
        val_seqs=val_seqs,
        val_labels=val_labels,
        seed=args.seed,
        s1_checkpoint=args.s1_checkpoint,
        batch_size=args.batch_size,
        config=config,
    )
    train_time = time.perf_counter() - t0
    logger.info(f"Training done in {train_time / 60:.1f} min")

    # ── Save checkpoint ─────────────────────────────────────────────────
    _save_joint_checkpoint(train_out["best_params"], output_dir)

    # ── Test eval (per-cell) ───────────────────────────────────────────
    logger.info("Running per-cell test evaluation...")
    test_metrics = _evaluate_per_cell_test(
        predict_per_cell_fn=train_out["predict_per_cell_fn"],
        chr_split=args.chr_split,
        include_alt_alleles=args.include_alt_alleles,
        save_predictions_dir=output_dir if args.save_predictions else None,
    )

    # ── Save result.json ───────────────────────────────────────────────
    result = MultitaskRunResult(
        student=args.student,
        reservoir=args.reservoir,
        seed=args.seed,
        n_train=len(train_seqs),
        val_pearson_per_cell=train_out["val_pearson_per_cell"],
        val_pearson_mean=train_out["val_pearson_mean"],
        test_metrics_per_cell=test_metrics,
    )
    (output_dir / "result.json").write_text(json.dumps(asdict(result), indent=2, default=str))
    logger.info(f"Saved result.json to {output_dir}")

    # Console summary
    logger.info("=== Per-cell test summary ===")
    for cl in CELL_LINES:
        m = test_metrics.get(cl, {})
        ind = m.get("in_dist", {})
        ood = m.get("ood", {})
        logger.info(
            f"  {cl}: in_dist Pearson={ind.get('pearson_r', 'n/a')}, "
            f"OOD Pearson={ood.get('pearson_r', 'n/a')}"
        )


if __name__ == "__main__":
    main()
