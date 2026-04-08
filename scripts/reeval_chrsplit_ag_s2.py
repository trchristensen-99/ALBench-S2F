#!/usr/bin/env python
"""Re-evaluate LegNet models on chr-split AG S2 oracle test sets.

Walks through result.json files in exp1_1 output directories, loads the
corresponding saved model checkpoint, and re-evaluates on chr-split test
sequences labeled by the AG S2 oracle ensemble.

Overwrites test_metrics in result.json with chr-split AG S2 metrics.

Usage:
    uv run --no-sync python scripts/reeval_chrsplit_ag_s2.py \
        --results-dir outputs/exp1_1/k562/legnet_ag_s2

    # Dry run (show what would be re-evaluated):
    uv run --no-sync python scripts/reeval_chrsplit_ag_s2.py \
        --results-dir outputs/exp1_1/k562/legnet_ag_s2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TEST_DIR = REPO / "data" / "k562" / "test_sets_ag_s2_chrsplit"


def _safe_corr(a, b, fn):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    return float(fn(a[mask], b[mask])[0])


def _predict_legnet(model, sequences, device, batch_size=512):
    """Predict with LegNet (4-channel, 200bp, RC-averaged)."""
    from data.utils import one_hot_encode

    def _encode(seq):
        seq = seq.upper()
        if len(seq) < 200:
            pad = 200 - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > 200:
            start = (len(seq) - 200) // 2
            seq = seq[start : start + 200]
        return one_hot_encode(seq, add_singleton_channel=False)  # (4, 200)

    encoded = np.stack([_encode(s) for s in sequences])
    x = torch.from_numpy(encoded).float()
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(device)
            out = model(batch)
            batch_rc = batch.flip(-1)[:, [3, 2, 1, 0], :]
            out_rc = model(batch_rc)
            avg = ((out + out_rc) / 2.0).cpu().numpy().reshape(-1)
            preds.append(avg)
    return np.concatenate(preds)


def evaluate_model(model, device, test_dir):
    """Evaluate model on all chr-split AG S2 test sets."""
    metrics = {}

    # In-distribution
    f = test_dir / "genomic_oracle.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        seqs = d["sequences"].tolist()
        oracle = d["oracle_mean"]
        preds = _predict_legnet(model, seqs, device)
        metrics["in_dist"] = {
            "pearson_r": _safe_corr(preds, oracle, pearsonr),
            "spearman_r": _safe_corr(preds, oracle, spearmanr),
            "mse": float(np.mean((preds - oracle) ** 2)),
            "n": len(seqs),
        }
        # Also compute against real labels if available
        if "true_label" in d:
            true = d["true_label"]
            metrics["in_dist_real"] = {
                "pearson_r": _safe_corr(preds, true, pearsonr),
                "spearman_r": _safe_corr(preds, true, spearmanr),
                "mse": float(np.mean((preds - true) ** 2)),
                "n": len(seqs),
            }

    # SNV
    f = test_dir / "snv_oracle.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        ref_seqs = d["ref_sequences"].tolist()
        alt_seqs = d["alt_sequences"].tolist()
        alt_oracle = d["alt_mean"]
        delta_oracle = d["delta_mean"]

        ref_preds = _predict_legnet(model, ref_seqs, device)
        alt_preds = _predict_legnet(model, alt_seqs, device)
        delta_preds = alt_preds - ref_preds

        metrics["snv_abs"] = {
            "pearson_r": _safe_corr(alt_preds, alt_oracle, pearsonr),
            "spearman_r": _safe_corr(alt_preds, alt_oracle, spearmanr),
            "mse": float(np.mean((alt_preds - alt_oracle) ** 2)),
            "n": len(alt_seqs),
        }
        metrics["snv_delta"] = {
            "pearson_r": _safe_corr(delta_preds, delta_oracle, pearsonr),
            "spearman_r": _safe_corr(delta_preds, delta_oracle, spearmanr),
            "mse": float(np.mean((delta_preds - delta_oracle) ** 2)),
            "n": len(alt_seqs),
        }

    # OOD
    f = test_dir / "ood_oracle.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        seqs = d["sequences"].tolist()
        oracle = d["oracle_mean"]
        preds = _predict_legnet(model, seqs, device)
        metrics["ood"] = {
            "pearson_r": _safe_corr(preds, oracle, pearsonr),
            "spearman_r": _safe_corr(preds, oracle, spearmanr),
            "mse": float(np.mean((preds - oracle) ** 2)),
            "n": len(seqs),
        }
        if "true_label" in d:
            true = d["true_label"]
            metrics["ood_real"] = {
                "pearson_r": _safe_corr(preds, true, pearsonr),
                "n": len(seqs),
            }

    return metrics


def load_legnet_from_checkpoint(ckpt_path, device):
    """Load LegNet model(s) from checkpoint, return first model."""
    from models.legnet import LegNet

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "model_state_dicts" in state:
        sd = state["model_state_dicts"][0]
    elif "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state

    # Detect block_sizes from checkpoint weights
    block_sizes = None
    if "block_sizes" in state:
        block_sizes = state["block_sizes"]

    model = LegNet(in_channels=4, block_sizes=block_sizes) if block_sizes else LegNet(in_channels=4)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Re-eval even if chr-split metrics exist"
    )
    args = parser.parse_args()

    if not args.test_dir.exists():
        logger.error(f"Test dir not found: {args.test_dir}")
        logger.error("Run: python scripts/generate_ag_s2_test_labels.py first")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Results dir: {args.results_dir}")
    logger.info(f"Test dir: {args.test_dir}")

    result_files = sorted(args.results_dir.rglob("result.json"))
    logger.info(f"Found {len(result_files)} result.json files")

    n_reeval = 0
    n_skip = 0
    for rj in result_files:
        # Check if already has chr-split metrics
        d = json.loads(rj.read_text())
        if not args.force and d.get("test_split") == "chr_split_ag_s2":
            n_skip += 1
            continue

        # Find checkpoint
        run_dir = rj.parent
        ckpt = run_dir / "best_model.pt"
        if not ckpt.exists():
            continue

        if args.dry_run:
            logger.info(f"  WOULD re-eval: {run_dir.relative_to(args.results_dir)}")
            n_reeval += 1
            continue

        try:
            model = load_legnet_from_checkpoint(ckpt, device)
            metrics = evaluate_model(model, device, args.test_dir)

            # Update result.json
            d["test_metrics"] = metrics
            d["test_split"] = "chr_split_ag_s2"
            rj.write_text(json.dumps(d, indent=2, default=str))
            n_reeval += 1

            if n_reeval % 20 == 0:
                id_r = metrics.get("in_dist", {}).get("pearson_r", -1)
                logger.info(f"  Re-evaluated {n_reeval}: id={id_r:.4f} ({run_dir.name})")

        except Exception as e:
            logger.warning(f"  Failed: {run_dir.name}: {e}")

    logger.info(f"\nDone: {n_reeval} re-evaluated, {n_skip} skipped")


if __name__ == "__main__":
    main()
