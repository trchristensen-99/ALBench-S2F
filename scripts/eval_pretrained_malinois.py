#!/usr/bin/env python
"""Evaluate the official pretrained Malinois model on our test sets.

This loads the boda2 pretrained weights (3-output K562/HepG2/SknSh model)
and evaluates on chr-split and/or hashfrag test sets without any training.

Usage:
    python scripts/eval_pretrained_malinois.py [--chr-split] [--hashfrag] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.k562 import K562Dataset
from experiments.train_malinois_k562 import (
    CELL_LINE_LABEL_COLS,
    K562MalinoisDataset,
    _evaluate_chr_split_test,
    _predict_sequences,
    evaluate_test_sets,
)
from models.basset_branched import BassetBranched


def load_pretrained_malinois(
    checkpoint_path: str = "data/pretrained/malinois_trained/torch_checkpoint.pt",
    device: torch.device = torch.device("cpu"),
) -> BassetBranched:
    """Load the official pretrained Malinois model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract hyperparams from checkpoint
    hp = vars(ckpt["model_hparams"]) if "model_hparams" in ckpt else {}

    model = BassetBranched(
        input_len=hp.get("input_len", 600),
        conv1_channels=hp.get("conv1_channels", 300),
        conv1_kernel_size=hp.get("conv1_kernel_size", 19),
        conv2_channels=hp.get("conv2_channels", 200),
        conv2_kernel_size=hp.get("conv2_kernel_size", 11),
        conv3_channels=hp.get("conv3_channels", 200),
        conv3_kernel_size=hp.get("conv3_kernel_size", 7),
        n_linear_layers=hp.get("n_linear_layers", 1),
        linear_channels=hp.get("linear_channels", 1000),
        linear_dropout_p=hp.get("linear_dropout_p", 0.116),
        n_branched_layers=hp.get("n_branched_layers", 3),
        branched_channels=hp.get("branched_channels", 140),
        branched_dropout_p=hp.get("branched_dropout_p", 0.576),
        n_outputs=hp.get("n_outputs", 3),
    )

    # Load state dict
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded pretrained Malinois: {n_params:,} parameters")
    print(f"  Architecture: {hp}")
    return model


def evaluate_pretrained(
    model: BassetBranched,
    device: torch.device,
    output_dir: Path,
    do_chr_split: bool = True,
    do_hashfrag: bool = True,
) -> dict:
    """Evaluate pretrained model on all cell types and test sets."""
    results = {}
    cell_types = [("k562", 0), ("hepg2", 1), ("sknsh", 2)]

    cfg = {
        "use_reverse_complement": True,
        "cell_line": "k562",
    }

    for ct_name, ct_idx in cell_types:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {ct_name.upper()} (output index {ct_idx})")
        print(f"{'=' * 60}")
        ct_results = {}

        if do_chr_split:
            print(f"\n--- Chr-split test ---")
            label_col = CELL_LINE_LABEL_COLS.get(ct_name, "K562_log2FC")
            try:
                test_ds = K562MalinoisDataset(
                    K562Dataset(
                        data_path="data/k562",
                        split="test",
                        label_column=label_col,
                        use_hashfrag=False,
                        use_chromosome_fallback=True,
                        include_alt_alleles=True,
                    )
                )
                ct_cfg = {**cfg, "cell_line": ct_name}

                # Evaluate with the right output column
                from torch.utils.data import DataLoader

                loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
                preds, trues = [], []
                model.eval()
                with torch.no_grad():
                    for x, y in loader:
                        x = x.to(device)
                        out = model(x)[:, ct_idx]
                        if cfg["use_reverse_complement"]:
                            x_rc = x.flip(-1)[:, [3, 2, 1, 0], :]
                            out_rc = model(x_rc)[:, ct_idx]
                            out = (out + out_rc) / 2.0
                        preds.append(out.cpu().numpy())
                        trues.append(y.numpy())
                pred = np.concatenate(preds)
                true = np.concatenate(trues)
                from scipy.stats import pearsonr, spearmanr

                ct_results["chr_split_in_dist"] = {
                    "pearson_r": float(pearsonr(pred, true)[0]),
                    "spearman_r": float(spearmanr(pred, true)[0]),
                    "mse": float(np.mean((pred - true) ** 2)),
                    "n": len(true),
                }
                print(f"  in_dist Pearson: {ct_results['chr_split_in_dist']['pearson_r']:.4f}")

                # SNV and OOD via evaluate_test_sets with cell_type_idx
                test_set_dir = Path("data/k562/test_sets")
                if test_set_dir.exists():
                    snv_ood = evaluate_test_sets(
                        model, device, test_set_dir, ct_cfg, cell_type_idx=ct_idx
                    )
                    for key in ["snv_abs", "snv_delta", "ood"]:
                        if key in snv_ood:
                            ct_results[f"chr_split_{key}"] = snv_ood[key]
                            print(f"  {key} Pearson: {snv_ood[key]['pearson_r']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

        if do_hashfrag:
            print(f"\n--- HashFrag test ---")
            try:
                test_dir = Path("data/k562/test_sets")
                ct_cfg = {**cfg, "cell_line": ct_name}
                hf_metrics = evaluate_test_sets(
                    model, device, test_dir, ct_cfg, cell_type_idx=ct_idx
                )
                for key, val in hf_metrics.items():
                    ct_results[f"hashfrag_{key}"] = val
                    if isinstance(val, dict) and "pearson_r" in val:
                        print(f"  {key} Pearson: {val['pearson_r']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

        results[ct_name] = ct_results

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pretrained_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="data/pretrained/malinois_trained/torch_checkpoint.pt",
    )
    parser.add_argument("--output-dir", default="outputs/malinois_pretrained_eval")
    parser.add_argument("--chr-split", action="store_true", default=True)
    parser.add_argument("--no-chr-split", dest="chr_split", action="store_false")
    parser.add_argument("--hashfrag", action="store_true", default=True)
    parser.add_argument("--no-hashfrag", dest="hashfrag", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_pretrained_malinois(args.checkpoint, device)
    evaluate_pretrained(
        model,
        device,
        Path(args.output_dir),
        do_chr_split=args.chr_split,
        do_hashfrag=args.hashfrag,
    )


if __name__ == "__main__":
    main()
