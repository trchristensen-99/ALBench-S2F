#!/usr/bin/env python
"""Save complete predictions.npz for from-scratch models in chr_split.

Generates in_dist, SNV, and OOD predictions for DREAM-RNN and DREAM-CNN
models. Existing predictions.npz files (which may only contain in_dist)
are overwritten when --force is used.

Usage::

    python scripts/save_from_scratch_chr_split_predictions.py --cell k562 --model dream_rnn --force
    python scripts/save_from_scratch_chr_split_predictions.py --cell hepg2 --model dream_cnn --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.k562 import K562Dataset  # noqa: E402
from data.utils import one_hot_encode  # noqa: E402

CELL_LINE_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}


def _safe_corr(pred, true, fn):
    mask = np.isfinite(pred) & np.isfinite(true)
    return float(fn(pred[mask], true[mask])[0]) if mask.sum() >= 3 else 0.0


# ── Sequence encoding helpers ────────────────────────────────────────────────


def _encode_200bp(seq_str: str, n_channels: int = 4) -> np.ndarray:
    """Encode a raw sequence to (n_channels, 200) array.

    For 5-channel models (DREAM-RNN), adds a zero RC flag channel.
    """
    seq_str = seq_str.upper()
    target = 200
    if len(seq_str) < target:
        pad = target - len(seq_str)
        seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
    elif len(seq_str) > target:
        start = (len(seq_str) - target) // 2
        seq_str = seq_str[start : start + target]
    base = one_hot_encode(seq_str, add_singleton_channel=False)  # (4, L)
    if n_channels == 5:
        rc_flag = np.zeros((1, target), dtype=np.float32)
        return np.concatenate([base, rc_flag], axis=0)
    return base


def _predict_sequences_dream_rnn(
    models: list[torch.nn.Module],
    sequences: list[str],
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """RC-averaged predictions for DREAM-RNN (5-channel, 200bp).

    Averages across all ensemble members.
    """
    if not sequences:
        return np.array([], dtype=np.float32)

    encoded = np.stack([_encode_200bp(s, n_channels=5) for s in sequences])
    x = torch.from_numpy(encoded).float()

    ensemble_preds = []
    for model in models:
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i : i + batch_size].to(device)
                p = model.predict(batch, use_reverse_complement=True)
                preds.append(p.cpu().numpy().reshape(-1))
        ensemble_preds.append(np.concatenate(preds))

    return np.mean(np.stack(ensemble_preds, axis=0), axis=0)


def _predict_sequences_dream_cnn(
    models: list[torch.nn.Module],
    sequences: list[str],
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Forward-pass predictions for DREAM-CNN (4-channel, 200bp).

    DREAM-CNN does not use RC-averaging (was not trained with RC flag).
    We still do RC-averaging at inference: forward(x) + forward(RC(x)) / 2.
    Averages across all ensemble members.
    """
    if not sequences:
        return np.array([], dtype=np.float32)

    encoded = np.stack([_encode_200bp(s, n_channels=4) for s in sequences])
    x = torch.from_numpy(encoded).float()

    ensemble_preds = []
    for model in models:
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i : i + batch_size].to(device)
                out = model(batch)

                # RC averaging: reverse sequence, swap ACGT
                batch_rc = batch.flip(-1)[:, [3, 2, 1, 0], :]
                out_rc = model(batch_rc)
                avg = ((out + out_rc) / 2.0).cpu().numpy().reshape(-1)
                preds.append(avg)
        ensemble_preds.append(np.concatenate(preds))

    return np.mean(np.stack(ensemble_preds, axis=0), axis=0)


# ── Test data loading ────────────────────────────────────────────────────────


def load_test_data(cell_line: str):
    """Load chr-split test data (in_dist, SNV, OOD)."""
    label_col = CELL_LINE_LABEL_COLS[cell_line]
    test_dir = REPO / "data" / cell_line / "test_sets"
    if not test_dir.exists():
        test_dir = REPO / "data" / "k562" / "test_sets"

    result = {}

    # In-distribution (chr7+13)
    ds = K562Dataset(
        data_path=str(REPO / "data" / "k562"),
        split="test",
        label_column=label_col,
        use_hashfrag=False,
        use_chromosome_fallback=True,
        include_alt_alleles=True,
    )
    result["in_dist"] = {
        "sequences": list(ds.sequences),
        "labels": ds.labels.astype(np.float32),
    }

    # SNV pairs
    snv_path = test_dir / "test_snv_pairs_hashfrag.tsv"
    if not snv_path.exists():
        snv_path = REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        # Filter to chr7+13 for chr-split
        if "IDs_ref" in snv_df.columns:
            test_chrs = {"7", "13", "chr7", "chr13"}
            chroms = snv_df["IDs_ref"].str.split(":", expand=True)[0]
            snv_df = snv_df[chroms.isin(test_chrs)].reset_index(drop=True)

        alt_col = f"{label_col}_alt"
        if alt_col not in snv_df.columns:
            alt_col = "K562_log2FC_alt"
        delta_col = f"delta_{label_col}"
        if delta_col not in snv_df.columns:
            delta_col = "delta_log2FC"

        result["snv"] = {
            "ref_sequences": snv_df["sequence_ref"].tolist(),
            "alt_sequences": snv_df["sequence_alt"].tolist(),
            "alt_true": snv_df[alt_col].to_numpy(dtype=np.float32)
            if alt_col in snv_df.columns
            else None,
            "delta_true": snv_df[delta_col].to_numpy(dtype=np.float32)
            if delta_col in snv_df.columns
            else None,
        }

    # OOD
    ood_path = test_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = REPO / "data" / "k562" / "test_sets" / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists() and cell_line == "k562":
        ood_path = REPO / "data" / "k562" / "test_sets" / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_col = label_col if label_col in ood_df.columns else "K562_log2FC"
        if ood_col in ood_df.columns:
            result["ood"] = {
                "sequences": ood_df["sequence"].tolist(),
                "labels": ood_df[ood_col].to_numpy(dtype=np.float32),
            }

    return result


# ── Model loading ────────────────────────────────────────────────────────────


def _extract_state_dicts(ckpt_path: Path) -> list[dict]:
    """Extract model state dict(s) from a checkpoint file.

    Handles both formats:
    - Ensemble format (from exp1_1_scaling.py): has ``model_state_dicts`` key (list)
    - Legacy single-model format: has ``model_state_dict`` key or is a raw state dict
    """
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Ensemble format: {"student_type": ..., "ensemble_size": N, "model_state_dicts": [...]}
    if "model_state_dicts" in state:
        sds = state["model_state_dicts"]
        print(f"    Ensemble checkpoint: {len(sds)} model(s)")
        return sds

    # Legacy single-model format
    sd = state.get("model_state_dict", state)
    return [sd]


def load_dream_rnn(ckpt_path: Path, device: torch.device):
    """Load DREAM-RNN model(s) from checkpoint.

    Returns a list of models for ensemble averaging.
    """
    from models.dream_rnn import create_dream_rnn

    state_dicts = _extract_state_dicts(ckpt_path)

    # Detect input channels from first state dict
    sd0 = state_dicts[0]
    first_conv_key = "conv1_short.0.weight"
    if first_conv_key in sd0:
        in_channels = sd0[first_conv_key].shape[1]
    else:
        in_channels = 5

    models = []
    for sd in state_dicts:
        model = create_dream_rnn(input_channels=in_channels, sequence_length=200, task_mode="k562")
        model.load_state_dict(sd)
        model.to(device).eval()
        models.append(model)

    return models, in_channels


def load_dream_cnn(ckpt_path: Path, device: torch.device):
    """Load DREAM-CNN model(s) from checkpoint.

    Returns a list of models for ensemble averaging.
    """
    from models.dream_cnn import DREAMCNN

    state_dicts = _extract_state_dicts(ckpt_path)

    # Detect input channels from first state dict
    sd0 = state_dicts[0]
    stem_key = "stem.conv_short.0.weight"
    if stem_key in sd0:
        in_channels = sd0[stem_key].shape[1]
    else:
        in_channels = 4

    models = []
    for sd in state_dicts:
        model = DREAMCNN(in_channels=in_channels, task_mode="k562")
        model.load_state_dict(sd)
        model.to(device).eval()
        models.append(model)

    return models, in_channels


# ── Main ─────────────────────────────────────────────────────────────────────


def generate_predictions(
    models: list,
    model_type: str,
    in_channels: int,
    test_data: dict,
    device: torch.device,
    batch_size: int = 512,
):
    """Generate predictions for all test sets."""

    def predict_fn(seqs):
        if model_type == "dream_rnn":
            return _predict_sequences_dream_rnn(models, seqs, device, batch_size)
        return _predict_sequences_dream_cnn(models, seqs, device, batch_size)

    arrays = {}
    metrics = {}

    # In-distribution
    if "in_dist" in test_data:
        td = test_data["in_dist"]
        print(f"    Predicting in_dist ({len(td['sequences'])} seqs)...", flush=True)
        pred = predict_fn(td["sequences"])
        true = td["labels"]
        arrays["in_dist_pred"] = pred
        arrays["in_dist_true"] = true
        metrics["in_dist"] = {
            "pearson_r": _safe_corr(pred, true, pearsonr),
            "spearman_r": _safe_corr(pred, true, spearmanr),
            "mse": float(np.mean((pred - true) ** 2)),
            "n": len(true),
        }
        print(
            f"      in_dist: pearson={metrics['in_dist']['pearson_r']:.4f}, "
            f"mse={metrics['in_dist']['mse']:.4f}"
        )

    # SNV
    if "snv" in test_data:
        td = test_data["snv"]
        print(f"    Predicting SNV ref ({len(td['ref_sequences'])} seqs)...", flush=True)
        ref_pred = predict_fn(td["ref_sequences"])
        print(f"    Predicting SNV alt ({len(td['alt_sequences'])} seqs)...", flush=True)
        alt_pred = predict_fn(td["alt_sequences"])
        arrays["snv_ref_pred"] = ref_pred
        arrays["snv_alt_pred"] = alt_pred

        if td["alt_true"] is not None:
            arrays["snv_alt_true"] = td["alt_true"]
            metrics["snv_abs"] = {
                "pearson_r": _safe_corr(alt_pred, td["alt_true"], pearsonr),
                "spearman_r": _safe_corr(alt_pred, td["alt_true"], spearmanr),
                "mse": float(np.mean((alt_pred - td["alt_true"]) ** 2)),
                "n": len(td["alt_true"]),
            }
            print(f"      snv_abs: pearson={metrics['snv_abs']['pearson_r']:.4f}")

        delta_pred = alt_pred - ref_pred
        arrays["snv_delta_pred"] = delta_pred
        if td["delta_true"] is not None:
            arrays["snv_delta_true"] = td["delta_true"]
            metrics["snv_delta"] = {
                "pearson_r": _safe_corr(delta_pred, td["delta_true"], pearsonr),
                "spearman_r": _safe_corr(delta_pred, td["delta_true"], spearmanr),
                "mse": float(np.mean((delta_pred - td["delta_true"]) ** 2)),
                "n": len(td["delta_true"]),
            }
            print(f"      snv_delta: pearson={metrics['snv_delta']['pearson_r']:.4f}")

    # OOD
    if "ood" in test_data:
        td = test_data["ood"]
        print(f"    Predicting OOD ({len(td['sequences'])} seqs)...", flush=True)
        pred = predict_fn(td["sequences"])
        true = td["labels"]
        arrays["ood_pred"] = pred
        arrays["ood_true"] = true
        metrics["ood"] = {
            "pearson_r": _safe_corr(pred, true, pearsonr),
            "spearman_r": _safe_corr(pred, true, spearmanr),
            "mse": float(np.mean((pred - true) ** 2)),
            "n": len(true),
        }
        print(
            f"      ood: pearson={metrics['ood']['pearson_r']:.4f}, mse={metrics['ood']['mse']:.4f}"
        )

    return arrays, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Save complete predictions for from-scratch models in chr_split"
    )
    parser.add_argument("--cell", required=True, choices=["k562", "hepg2", "sknsh"])
    parser.add_argument("--model", required=True, choices=["dream_rnn", "dream_cnn"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing predictions.npz")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Cell: {args.cell}")
    print(f"Model: {args.model}")
    print(f"Force: {args.force}")

    # Load test data once
    print("\nLoading test data...", flush=True)
    test_data = load_test_data(args.cell)
    for name, data in test_data.items():
        if "sequences" in data:
            print(f"  {name}: {len(data['sequences'])} sequences")
        elif "ref_sequences" in data:
            print(f"  {name}: {len(data['ref_sequences'])} pairs")

    # Find model directories
    model_dir = REPO / "outputs" / "chr_split" / args.cell / args.model
    if not model_dir.exists():
        print(f"\nERROR: {model_dir} does not exist")
        sys.exit(1)

    # Find all run directories with checkpoints
    ckpt_dirs = []
    for rj in sorted(model_dir.rglob("result.json")):
        run_dir = rj.parent
        ckpt_path = run_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt_dirs.append(run_dir)

    if not ckpt_dirs:
        print(f"\nNo checkpoints found in {model_dir}")
        sys.exit(0)

    print(f"\nFound {len(ckpt_dirs)} runs with checkpoints:")
    for d in ckpt_dirs:
        pred_exists = (d / "predictions.npz").exists()
        status = " [EXISTS]" if pred_exists else ""
        print(f"  {d.relative_to(REPO)}{status}")

    for i, run_dir in enumerate(ckpt_dirs):
        pred_path = run_dir / "predictions.npz"
        if pred_path.exists() and not args.force:
            # Check if existing predictions are complete (have SNV + OOD)
            existing = np.load(str(pred_path))
            if "snv_ref_pred" in existing and "ood_pred" in existing:
                print(f"\n[{i + 1}/{len(ckpt_dirs)}] SKIP (complete): {run_dir.relative_to(REPO)}")
                continue
            else:
                print(
                    f"\n[{i + 1}/{len(ckpt_dirs)}] Regenerating (incomplete): "
                    f"{run_dir.relative_to(REPO)} — has keys: {list(existing.keys())}"
                )
        elif pred_path.exists() and args.force:
            print(f"\n[{i + 1}/{len(ckpt_dirs)}] Overwriting: {run_dir.relative_to(REPO)}")
        else:
            print(f"\n[{i + 1}/{len(ckpt_dirs)}] Generating: {run_dir.relative_to(REPO)}")

        ckpt_path = run_dir / "best_model.pt"
        try:
            if args.model == "dream_rnn":
                models, in_ch = load_dream_rnn(ckpt_path, device)
            else:
                models, in_ch = load_dream_cnn(ckpt_path, device)
            print(f"    Loaded {len(models)} model(s) ({in_ch} channels)")

            arrays, metrics = generate_predictions(
                models, args.model, in_ch, test_data, device, args.batch_size
            )

            if arrays:
                np.savez_compressed(str(pred_path), **arrays)
                print(f"    Saved: {pred_path} ({pred_path.stat().st_size / 1024:.0f} KB)")
                print(f"    Keys: {sorted(arrays.keys())}")

                # Update result.json
                result_path = run_dir / "result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    result["test_metrics"] = metrics
                else:
                    result = {
                        "model": args.model,
                        "cell": args.cell,
                        "test_metrics": metrics,
                    }
                result_path.write_text(json.dumps(result, indent=2, default=str))

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
