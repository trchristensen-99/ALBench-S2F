#!/usr/bin/env python
"""Eval-only: load trained DREAM-RNN oracle checkpoint and run test evaluation.

For folds that finished training (best_model.pt exists) but timed out before
test evaluation completed (no summary.json).  Writes/updates summary.json with
test_metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.yeast import YeastDataset
from evaluation.yeast_testsets import (
    evaluate_yeast_test_subsets,
    load_yeast_test_subsets,
)
from models.dream_rnn import create_dream_rnn


def _predict_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_reverse_complement: bool,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yhat = model.predict(xb, use_reverse_complement=use_reverse_complement)
            preds.append(yhat.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Eval-only for trained DREAM-RNN oracle folds")
    parser.add_argument("--oracle-dir", type=Path, required=True, help="Root oracle output dir")
    parser.add_argument("--fold-id", type=int, required=True, help="Fold to evaluate")
    parser.add_argument("--data-path", type=Path, default=Path("data/yeast"))
    parser.add_argument("--context-mode", type=str, default="dream150")
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--cnn-filters", type=int, default=256)
    parser.add_argument("--dropout-cnn", type=float, default=0.2)
    parser.add_argument("--dropout-lstm", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-reverse-complement", action="store_true", default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--fold-split-seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fold_dir = args.oracle_dir / f"oracle_{args.fold_id}"
    best_ckpt = fold_dir / "best_model.pt"

    if not best_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {best_ckpt}")

    print(f"Evaluating fold {args.fold_id} from {fold_dir}")
    print(f"Device: {device}")

    # Load test dataset
    test_dataset = YeastDataset(
        data_path=str(args.data_path),
        split="test",
        context_mode=args.context_mode,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model and load checkpoint
    seq_len = test_dataset.get_sequence_length()
    model = create_dream_rnn(
        input_channels=6,
        sequence_length=seq_len,
        task_mode="yeast",
        hidden_dim=args.hidden_dim,
        cnn_filters=args.cnn_filters,
        dropout_cnn=args.dropout_cnn,
        dropout_lstm=args.dropout_lstm,
    ).to(device)

    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    print("Loaded checkpoint")

    # Load test subsets
    subset_dir = args.data_path / "test_subset_ids"
    if not subset_dir.exists():
        raise FileNotFoundError(f"Test subset dir not found: {subset_dir}")

    subsets = load_yeast_test_subsets(subset_dir=subset_dir)

    # Run predictions
    preds = _predict_test(
        model, test_loader, device, use_reverse_complement=args.use_reverse_complement
    )
    test_metrics = evaluate_yeast_test_subsets(
        predictions=preds,
        labels=test_dataset.labels.astype(np.float32),
        subsets=subsets,
    )

    print("Test metrics:")
    for subset_name, m in test_metrics.items():
        print(f"  {subset_name}: pearson_r={m['pearson_r']:.4f}  spearman_r={m['spearman_r']:.4f}")

    # Load existing summary or create minimal one
    summary_path = fold_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {
            "fold_id": args.fold_id,
            "n_folds": args.n_folds,
            "fold_split_seed": args.fold_split_seed,
        }

    summary["test_metrics"] = test_metrics

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
