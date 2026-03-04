#!/usr/bin/env python
"""Generate DREAM-RNN yeast oracle pseudo-labels for train/val/test.

The full train.txt (~6M sequences) is loaded as split="train". Oracle models
were trained on a 100K k-fold subset — for those indices, out-of-fold (OOF)
predictions are recorded; the remaining ~5.9M sequences are OOF by
construction (never seen during training).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.yeast import YeastDataset
from evaluation.yeast_testsets import evaluate_yeast_test_subsets, load_yeast_test_subsets
from models.dream_rnn import create_dream_rnn


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, fn: object) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(fn(y_true, y_pred)[0])


def _kfold_val_indices(n: int, n_folds: int, fold_seed: int) -> dict[int, np.ndarray]:
    perm = np.random.RandomState(fold_seed).permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=np.int64)
    fold_sizes[: n % n_folds] += 1
    out: dict[int, np.ndarray] = {}
    start = 0
    for fold in range(n_folds):
        end = start + int(fold_sizes[fold])
        out[fold] = perm[start:end]
        start = end
    return out


def _predict_dataset(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_reverse_complement: bool,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yhat = model.predict(xb, use_reverse_complement=use_reverse_complement)
            preds.append(yhat.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="generate_oracle_pseudolabels_yeast_dream",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    device = torch.device(f"cuda:{int(cfg.gpu)}" if torch.cuda.is_available() else "cpu")
    oracle_dir = Path(str(cfg.oracle_dir)).expanduser().resolve()
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_runs = []
    for run_dir in sorted(oracle_dir.glob("oracle_*")):
        ckpt = run_dir / "best_model.pt"
        summ = run_dir / "summary.json"
        if not ckpt.exists() or not summ.exists():
            continue
        fold_id = int(json.loads(summ.read_text()).get("fold_id", int(run_dir.name.split("_")[-1])))
        oracle_runs.append((fold_id, run_dir))
    if not oracle_runs:
        raise FileNotFoundError(f"No oracle checkpoints with summaries found in {oracle_dir}")

    print(f"Found {len(oracle_runs)} oracle folds in {oracle_dir}")

    ds_train = YeastDataset(
        data_path=str(cfg.data_path),
        split="train",
        context_mode=str(cfg.context_mode),
    )
    ds_val = YeastDataset(
        data_path=str(cfg.data_path),
        split="val",
        context_mode=str(cfg.context_mode),
    )
    ds_test = YeastDataset(
        data_path=str(cfg.data_path),
        split="test",
        context_mode=str(cfg.context_mode),
    )
    n_train = len(ds_train)

    fold_val_idx = _kfold_val_indices(
        n=n_train,
        n_folds=int(cfg.n_folds),
        fold_seed=int(cfg.fold_split_seed),
    )

    model = create_dream_rnn(
        input_channels=6,
        sequence_length=ds_train.get_sequence_length(),
        task_mode="yeast",
        hidden_dim=int(cfg.hidden_dim),
        cnn_filters=int(cfg.cnn_filters),
        dropout_cnn=float(cfg.dropout_cnn),
        dropout_lstm=float(cfg.dropout_lstm),
    ).to(device)

    def init_accum(n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    train_sum, train_sumsq = init_accum(n_train)
    val_sum, val_sumsq = init_accum(len(ds_val))
    test_sum, test_sumsq = init_accum(len(ds_test))
    train_oof = np.full(n_train, np.nan, dtype=np.float32)

    for fold_id, run_dir in tqdm(oracle_runs, desc="Oracle folds"):
        ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)

        train_preds = _predict_dataset(
            model,
            ds_train,
            device,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
            use_reverse_complement=bool(cfg.use_reverse_complement),
        ).astype(np.float32)
        val_preds = _predict_dataset(
            model,
            ds_val,
            device,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
            use_reverse_complement=bool(cfg.use_reverse_complement),
        ).astype(np.float32)
        test_preds = _predict_dataset(
            model,
            ds_test,
            device,
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
            use_reverse_complement=bool(cfg.use_reverse_complement),
        ).astype(np.float32)

        train_sum += train_preds
        train_sumsq += train_preds.astype(np.float64) ** 2
        val_sum += val_preds
        val_sumsq += val_preds.astype(np.float64) ** 2
        test_sum += test_preds
        test_sumsq += test_preds.astype(np.float64) ** 2

        if fold_id in fold_val_idx:
            idx = fold_val_idx[fold_id]
            train_oof[idx] = train_preds[idx]

    n_models = len(oracle_runs)

    def finalize(sum_arr: np.ndarray, sumsq_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = (sum_arr / n_models).astype(np.float32)
        var = np.maximum((sumsq_arr / n_models) - (mean.astype(np.float64) ** 2), 0.0)
        std = np.sqrt(var).astype(np.float32)
        return mean, std

    train_mean, train_std = finalize(train_sum, train_sumsq)
    val_mean, val_std = finalize(val_sum, val_sumsq)
    test_mean, test_std = finalize(test_sum, test_sumsq)

    train_labels = ds_train.labels.astype(np.float32)
    val_labels = ds_val.labels.astype(np.float32)
    test_labels = ds_test.labels.astype(np.float32)

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
        output_dir / "test_oracle_labels.npz",
        oracle_mean=test_mean,
        oracle_std=test_std,
        true_label=test_labels,
    )

    subsets = load_yeast_test_subsets(
        subset_dir=Path(str(cfg.data_path)) / "test_subset_ids",
        public_dir=None,
        use_private_only=False,
    )
    subset_metrics = evaluate_yeast_test_subsets(
        predictions=test_mean,
        labels=test_labels,
        subsets=subsets,
    )

    oof_mask = np.isfinite(train_oof)
    summary = {
        "n_oracle_models": n_models,
        "train": {
            "n": int(len(train_labels)),
            "ensemble_pearson_r": _safe_corr(train_labels, train_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(train_labels, train_mean, spearmanr),
            "oof_covered": int(np.sum(oof_mask)),
            "oof_pearson_r": _safe_corr(train_labels[oof_mask], train_oof[oof_mask], pearsonr),
            "oof_spearman_r": _safe_corr(train_labels[oof_mask], train_oof[oof_mask], spearmanr),
        },
        "val": {
            "n": int(len(val_labels)),
            "ensemble_pearson_r": _safe_corr(val_labels, val_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(val_labels, val_mean, spearmanr),
        },
        "test_full": {
            "n": int(len(test_labels)),
            "ensemble_pearson_r": _safe_corr(test_labels, test_mean, pearsonr),
            "ensemble_spearman_r": _safe_corr(test_labels, test_mean, spearmanr),
        },
        "test_subsets": subset_metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote pseudo-labels to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
