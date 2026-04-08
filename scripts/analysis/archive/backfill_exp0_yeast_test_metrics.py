#!/usr/bin/env python3
"""Backfill yeast random/SNV/genomic test metrics into existing exp0 result.json files."""

from __future__ import annotations

import argparse
import json
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


def predict_test(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yhat = model.predict(xb, use_reverse_complement=True)
            preds.append(yhat.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-path", type=str, default="data/yeast")
    parser.add_argument("--subset-dir", type=Path, default=None)
    parser.add_argument("--public-leaderboard-dir", type=Path, default=None)
    parser.add_argument("--private-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subset_dir = args.subset_dir or (Path(args.data_path) / "test_subset_ids")
    if not subset_dir.exists():
        raise SystemExit(f"Subset directory not found: {subset_dir}")

    result_paths = sorted(args.output_root.glob("fraction_*/result.json"))
    if not result_paths:
        raise SystemExit(f"No result.json files found under {args.output_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = YeastDataset(data_path=args.data_path, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_labels = test_dataset.labels.astype(np.float32)
    subsets = load_yeast_test_subsets(
        subset_dir=subset_dir,
        public_dir=args.public_leaderboard_dir,
        use_private_only=args.private_only,
    )

    for result_path in result_paths:
        payload = json.loads(result_path.read_text())
        if (not args.force) and payload.get("test_metrics", {}).get("random"):
            print(f"Skipping (already has test_metrics): {result_path}")
            continue

        ckpt_path = result_path.parent / "best_model.pt"
        if not ckpt_path.exists():
            print(f"Skipping (missing best_model.pt): {result_path.parent}")
            continue

        model = create_dream_rnn(
            input_channels=6,
            sequence_length=150,
            task_mode="yeast",
            hidden_dim=320,
            cnn_filters=160,
            dropout_cnn=0.1,
            dropout_lstm=0.1,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)

        preds = predict_test(model, test_loader, device)
        metrics = evaluate_yeast_test_subsets(preds, test_labels, subsets)
        payload["test_metrics"] = metrics
        result_path.write_text(json.dumps(payload, indent=2))

        print(
            f"Updated {result_path}: "
            f"random={metrics['random']['pearson_r']:.4f}, "
            f"snv={metrics['snv']['pearson_r']:.4f}, "
            f"genomic={metrics['genomic']['pearson_r']:.4f}"
        )


if __name__ == "__main__":
    main()
