#!/usr/bin/env python
"""Generate K562 LegNet oracle pseudo-labels via 10-fold cross-validation.

Trains 10 LegNet models (one per fold) on K562 hashFrag data, then generates:
  - Out-of-fold predictions for the entire training pool
  - Ensemble predictions for val and all test sets (in_dist, snv, ood, random)

Usage::

    uv run --no-sync python experiments/generate_oracle_pseudolabels_k562_legnet.py

Output::

    outputs/oracle_legnet_k562_ensemble/
        oracle_0/ ... oracle_9/   — per-fold model checkpoints
        train_oracle_labels.npz   — oracle_mean, oracle_std, oof_oracle, true_label
        val_oracle_labels.npz     — oracle_mean, oracle_std, true_label
        test_in_dist_oracle_labels.npz
        test_snv_oracle_labels.npz
        test_ood_oracle_labels.npz
        test_random_10k_oracle_labels.npz
        summary.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # noqa: E402

from data.k562 import K562Dataset  # noqa: E402
from models.legnet import LegNet, one_hot_encode_batch  # noqa: E402
from models.training import train_model_optimized  # noqa: E402


class _LegNetWithCheckpoint(LegNet):
    """LegNet subclass with save_checkpoint for train_model_optimized compat."""

    def save_checkpoint(self, path: str, **kwargs) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"model_state_dict": self.state_dict(), **kwargs}
        torch.save(checkpoint, path)

    def get_model_info(self) -> dict:
        return {
            "model_class": "LegNet",
            "task_mode": self.task_mode,
            "block_sizes": self.block_sizes,
        }


# ── Hyperparameters ──────────────────────────────────────────────────────────
N_FOLDS = 10
SEED = 42
BATCH_SIZE = 1024
EPOCHS = 80
LR = 0.005
WEIGHT_DECAY = 0.01
PCT_START = 0.3
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 2
USE_AMP = True
USE_COMPILE = True


# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe_corr(a: np.ndarray, b: np.ndarray, fn) -> float:
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(fn(a, b)[0])


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _oracle_fold_indices(n_total: int, n_folds: int = 10) -> dict[int, np.ndarray]:
    """Deterministic fold split matching the AG oracle approach (seed=42)."""
    perm = np.random.default_rng(seed=42).permutation(n_total)
    fold_size = n_total // n_folds
    fold_val_idx: dict[int, np.ndarray] = {}
    for fold_id in range(n_folds):
        val_start = fold_id * fold_size
        val_end = val_start + fold_size if fold_id < n_folds - 1 else n_total
        fold_val_idx[fold_id] = perm[val_start:val_end]
    return fold_val_idx


def _encode_sequences(sequences: np.ndarray, seq_len: int = 200) -> np.ndarray:
    """Encode string sequences to (N, 4, L) float32 array."""
    standardized: list[str] = []
    for seq in sequences:
        seq = str(seq).upper()
        if len(seq) < seq_len:
            pad = seq_len - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > seq_len:
            start = (len(seq) - seq_len) // 2
            seq = seq[start : start + seq_len]
        standardized.append(seq)
    return one_hot_encode_batch(standardized, seq_len=seq_len)


def _predict_batch(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run predictions on encoded sequences. Returns (N,) array."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i : i + batch_size]).float().to(device)
            out = model(batch).detach().cpu().numpy().reshape(-1)
            preds.append(out)
    return np.concatenate(preds)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = Path(
        os.environ.get(
            "OUTPUT_DIR",
            str(REPO / "outputs" / "oracle_legnet_k562_ensemble"),
        )
    )
    data_dir = Path(os.environ.get("K562_DATA_PATH", str(REPO / "data" / "k562")))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    ds_train = K562Dataset(data_path=str(data_dir), split="train")
    ds_val = K562Dataset(data_path=str(data_dir), split="val")

    train_seqs_raw = ds_train.sequences  # np array of strings
    train_labels = ds_train.labels.astype(np.float32)
    val_seqs_raw = ds_val.sequences
    val_labels = ds_val.labels.astype(np.float32)
    n_train = len(train_labels)
    n_val = len(val_labels)

    print(f"Train: {n_train:,}  Val: {n_val:,}")

    # Pre-encode all sequences
    print("Encoding training sequences...", flush=True)
    train_x = _encode_sequences(train_seqs_raw)
    print("Encoding validation sequences...", flush=True)
    val_x = _encode_sequences(val_seqs_raw)

    # ── Load test sets ───────────────────────────────────────────────────────
    test_dir = data_dir / "test_sets"

    in_dist_df = pd.read_csv(test_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_dist_seqs = in_dist_df["sequence"].astype(str).tolist()
    in_dist_labels = in_dist_df["K562_log2FC"].to_numpy(dtype=np.float32)
    in_dist_x = _encode_sequences(np.array(in_dist_seqs))

    snv_df = pd.read_csv(test_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    snv_ref_seqs = snv_df["sequence_ref"].astype(str).tolist()
    snv_alt_seqs = snv_df["sequence_alt"].astype(str).tolist()
    snv_delta_labels = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
    snv_ref_x = _encode_sequences(np.array(snv_ref_seqs))
    snv_alt_x = _encode_sequences(np.array(snv_alt_seqs))

    ood_df = pd.read_csv(test_dir / "test_ood_designed_k562.tsv", sep="\t")
    ood_seqs = ood_df["sequence"].astype(str).tolist()
    ood_labels = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
    ood_x = _encode_sequences(np.array(ood_seqs))

    # Random 10K
    rng = np.random.default_rng(42)
    nucleotides = np.array(list("ACGT"))
    indices = rng.integers(0, 4, size=(10_000, 200))
    random_seqs = ["".join(nucleotides[row]) for row in indices]
    random_x = _encode_sequences(np.array(random_seqs))

    print(
        f"Test sets loaded: in_dist={len(in_dist_labels):,}  "
        f"snv={len(snv_delta_labels):,} pairs  ood={len(ood_labels):,}  "
        f"random=10,000"
    )

    # ── Fold split ───────────────────────────────────────────────────────────
    fold_val_idx = _oracle_fold_indices(n_train, N_FOLDS)

    # ── Accumulators ─────────────────────────────────────────────────────────
    def _init(n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    train_sum, train_sumsq = _init(n_train)
    val_sum, val_sumsq = _init(n_val)
    in_dist_sum, in_dist_sumsq = _init(len(in_dist_labels))
    snv_ref_sum, snv_ref_sumsq = _init(len(snv_delta_labels))
    snv_alt_sum, snv_alt_sumsq = _init(len(snv_delta_labels))
    ood_sum, ood_sumsq = _init(len(ood_labels))
    random_sum, random_sumsq = _init(10_000)
    train_oof = np.full(n_train, np.nan, dtype=np.float32)

    trained_models: list[nn.Module] = []

    # ── Train k-fold models ──────────────────────────────────────────────────
    for fold_id in range(N_FOLDS):
        fold_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_id}/{N_FOLDS}")
        print(f"{'=' * 60}")

        fold_dir = output_dir / f"oracle_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Check if already trained
        best_ckpt = fold_dir / "best_model.pt"
        if best_ckpt.exists():
            print(f"  Loading existing checkpoint from {best_ckpt}")
            model = _LegNetWithCheckpoint(in_channels=4, task_mode="k562").to(device)
            state = torch.load(best_ckpt, map_location="cpu")
            model.load_state_dict(state["model_state_dict"], strict=True)
            model.eval()
        else:
            # Split train data into fold-train and fold-val
            val_idx = fold_val_idx[fold_id]
            all_idx = np.arange(n_train)
            train_mask = np.ones(n_train, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_idx[train_mask]

            fold_train_x = torch.from_numpy(train_x[train_idx])
            fold_train_y = torch.from_numpy(train_labels[train_idx])
            fold_val_x = torch.from_numpy(train_x[val_idx])
            fold_val_y = torch.from_numpy(train_labels[val_idx])

            print(f"  Fold train: {len(train_idx):,}  Fold val: {len(val_idx):,}")

            train_dataset = TensorDataset(fold_train_x, fold_train_y)
            val_dataset = TensorDataset(fold_val_x, fold_val_y)

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
            )

            # Create model with a unique seed per fold
            _set_seed(SEED + fold_id)
            model = _LegNetWithCheckpoint(in_channels=4, task_mode="k562").to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * EPOCHS
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=LR,
                total_steps=total_steps,
                pct_start=PCT_START,
            )
            criterion = nn.MSELoss()

            history = train_model_optimized(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=EPOCHS,
                device=device,
                scheduler=scheduler,
                checkpoint_dir=fold_dir,
                use_reverse_complement=False,
                early_stopping_patience=EARLY_STOP_PATIENCE,
                metric_for_best="pearson_r",
                use_amp=USE_AMP,
                use_compile=USE_COMPILE,
            )

            # Reload best checkpoint
            state = torch.load(best_ckpt, map_location="cpu")
            model = _LegNetWithCheckpoint(in_channels=4, task_mode="k562").to(device)
            model.load_state_dict(state["model_state_dict"], strict=True)
            model.eval()

            best_val_r = max(history["val_pearson_r"]) if history["val_pearson_r"] else 0.0
            print(f"  Best val Pearson: {best_val_r:.4f}")

            # Save fold summary
            fold_summary = {
                "fold_id": fold_id,
                "best_val_pearson_r": best_val_r,
                "epochs_run": len(history["val_loss"]),
                "n_fold_train": len(train_idx),
                "n_fold_val": len(val_idx),
                "training_time_s": time.time() - fold_start,
            }
            (fold_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2))

        trained_models.append(model)

        # ── Predictions for this fold ────────────────────────────────────────
        # Full training set predictions (for ensemble + OOF)
        train_preds = _predict_batch(model, train_x, device)
        train_sum += train_preds
        train_sumsq += train_preds.astype(np.float64) ** 2

        # OOF: this model predicts its held-out fold
        oof_idx = fold_val_idx[fold_id]
        train_oof[oof_idx] = train_preds[oof_idx]

        # Val predictions
        val_preds = _predict_batch(model, val_x, device)
        val_sum += val_preds
        val_sumsq += val_preds.astype(np.float64) ** 2

        # Test set predictions
        in_dist_preds = _predict_batch(model, in_dist_x, device)
        in_dist_sum += in_dist_preds
        in_dist_sumsq += in_dist_preds.astype(np.float64) ** 2

        snv_ref_preds = _predict_batch(model, snv_ref_x, device)
        snv_ref_sum += snv_ref_preds
        snv_ref_sumsq += snv_ref_preds.astype(np.float64) ** 2

        snv_alt_preds = _predict_batch(model, snv_alt_x, device)
        snv_alt_sum += snv_alt_preds
        snv_alt_sumsq += snv_alt_preds.astype(np.float64) ** 2

        ood_preds = _predict_batch(model, ood_x, device)
        ood_sum += ood_preds
        ood_sumsq += ood_preds.astype(np.float64) ** 2

        random_preds = _predict_batch(model, random_x, device)
        random_sum += random_preds
        random_sumsq += random_preds.astype(np.float64) ** 2

        fold_elapsed = time.time() - fold_start
        print(
            f"  Fold {fold_id} done ({fold_elapsed:.0f}s): "
            f"val_pearson={_safe_corr(val_labels, val_preds, pearsonr):.4f}  "
            f"in_dist_pearson={_safe_corr(in_dist_labels, in_dist_preds, pearsonr):.4f}  "
            f"ood_pearson={_safe_corr(ood_labels, ood_preds, pearsonr):.4f}"
        )

    # ── Finalize ensemble statistics ─────────────────────────────────────────
    n_models = N_FOLDS

    def _finalize(s: np.ndarray, s2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = (s / n_models).astype(np.float32)
        var = np.maximum((s2 / n_models) - mean.astype(np.float64) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        return mean, std

    train_mean, train_std = _finalize(train_sum, train_sumsq)
    val_mean, val_std = _finalize(val_sum, val_sumsq)
    in_dist_mean, in_dist_std = _finalize(in_dist_sum, in_dist_sumsq)
    snv_ref_mean, _ = _finalize(snv_ref_sum, snv_ref_sumsq)
    snv_alt_mean, snv_alt_std = _finalize(snv_alt_sum, snv_alt_sumsq)
    snv_delta_mean = snv_alt_mean - snv_ref_mean
    ood_mean, ood_std = _finalize(ood_sum, ood_sumsq)
    random_mean, random_std = _finalize(random_sum, random_sumsq)

    # ── Save .npz files ──────────────────────────────────────────────────────
    np.savez_compressed(
        output_dir / "train_oracle_labels.npz",
        oracle_mean=train_mean,
        oracle_std=train_std,
        oof_oracle=train_oof,
        true_label=train_labels,
    )
    np.savez_compressed(
        output_dir / "val_oracle_labels.npz",
        oracle_mean=val_mean,
        oracle_std=val_std,
        true_label=val_labels,
    )
    np.savez_compressed(
        output_dir / "test_in_dist_oracle_labels.npz",
        oracle_mean=in_dist_mean,
        oracle_std=in_dist_std,
        true_label=in_dist_labels,
    )
    np.savez_compressed(
        output_dir / "test_snv_oracle_labels.npz",
        ref_oracle_mean=snv_ref_mean,
        alt_oracle_mean=snv_alt_mean,
        delta_oracle_mean=snv_delta_mean,
        alt_oracle_std=snv_alt_std,
        true_delta=snv_delta_labels,
    )
    np.savez_compressed(
        output_dir / "test_ood_oracle_labels.npz",
        oracle_mean=ood_mean,
        oracle_std=ood_std,
        true_label=ood_labels,
    )
    np.savez_compressed(
        output_dir / "test_random_10k_oracle_labels.npz",
        oracle_mean=random_mean,
        oracle_std=random_std,
    )
    print(f"\nWrote .npz files to {output_dir}")

    # ── Also create Exp 1 test set NPZs ──────────────────────────────────────
    test_npz_dir = data_dir / "test_sets_legnet"
    test_npz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating Exp 1 test set NPZs in {test_npz_dir}")

    np.savez_compressed(
        test_npz_dir / "genomic_oracle.npz",
        sequences=np.array(in_dist_seqs),
        oracle_labels=in_dist_mean,
        oracle_std=in_dist_std,
        true_labels=in_dist_labels,
    )
    print(f"  genomic_oracle.npz: {len(in_dist_labels):,} sequences")

    np.savez_compressed(
        test_npz_dir / "snv_oracle.npz",
        ref_sequences=np.array(snv_ref_seqs),
        alt_sequences=np.array(snv_alt_seqs),
        ref_oracle_labels=snv_ref_mean,
        alt_oracle_labels=snv_alt_mean,
        delta_oracle_labels=snv_delta_mean,
        alt_oracle_std=snv_alt_std,
        true_delta=snv_delta_labels,
    )
    print(f"  snv_oracle.npz: {len(snv_delta_labels):,} pairs")

    np.savez_compressed(
        test_npz_dir / "ood_oracle.npz",
        sequences=np.array(ood_seqs),
        oracle_labels=ood_mean,
        oracle_std=ood_std,
        true_labels=ood_labels,
    )
    print(f"  ood_oracle.npz: {len(ood_labels):,} sequences")

    np.savez_compressed(
        test_npz_dir / "random_10k_oracle.npz",
        sequences=np.array(random_seqs),
        oracle_labels=random_mean,
        oracle_std=random_std,
    )
    print(f"  random_10k_oracle.npz: 10,000 sequences")

    # ── Summary metrics ──────────────────────────────────────────────────────
    oof_mask = np.isfinite(train_oof)
    summary = {
        "n_oracle_models": n_models,
        "oracle_folds": list(range(N_FOLDS)),
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "pct_start": PCT_START,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "seed": SEED,
        },
        "train_pool": {
            "n": int(n_train),
            "ensemble_pearson_r": _safe_corr(train_labels, train_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(train_labels, train_mean, spearmanr),
            "oof_covered": int(np.sum(oof_mask)),
            "oof_pearson_r": _safe_corr(train_labels[oof_mask], train_oof[oof_mask], pearsonr),
            "oof_spearman_r": _safe_corr(train_labels[oof_mask], train_oof[oof_mask], spearmanr),
        },
        "val": {
            "n": int(n_val),
            "ensemble_pearson_r": _safe_corr(val_labels, val_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(val_labels, val_mean, spearmanr),
        },
        "test_in_distribution": {
            "n": int(len(in_dist_labels)),
            "ensemble_pearson_r": _safe_corr(in_dist_labels, in_dist_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(in_dist_labels, in_dist_mean, spearmanr),
        },
        "test_snv_delta": {
            "n": int(len(snv_delta_labels)),
            "ensemble_pearson_r": _safe_corr(snv_delta_labels, snv_delta_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(snv_delta_labels, snv_delta_mean, spearmanr),
        },
        "test_ood": {
            "n": int(len(ood_labels)),
            "ensemble_pearson_r": _safe_corr(ood_labels, ood_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(ood_labels, ood_mean, spearmanr),
        },
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote summary to {output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))
    print(f"\nAll done. Test NPZs in {test_npz_dir}")


if __name__ == "__main__":
    main()
