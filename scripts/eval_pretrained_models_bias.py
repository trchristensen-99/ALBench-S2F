#!/usr/bin/env python
"""Evaluate ALL available pretrained MPRA models on random DNA & controls.

Tests pretrained models that we did NOT train ourselves to show the
systematic bias is a field-wide problem, not specific to our training.

Models tested:
  1. Malinois (pretrained, Gosai et al. 2024) — 3-output BassetBranched
  2. PARM (van Steensel lab, Nature 2025) — K562 promoter activity model
  3. MPRA-LegNet (Agarwal et al. 2025) — if weights available

Evaluates on:
  - 250 Agarwal dinucleotide-shuffled controls (real measured: mean=-0.53)
  - 200 Agarwal intergenic sequences (subsample)
  - 1000 random 200bp DNA sequences
  - K562 chr-split SNV pairs (delta effect prediction)

Usage:
    uv run --no-sync python scripts/eval_pretrained_models_bias.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def one_hot_encode(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in mapping:
            arr[mapping[c], i] = 1.0
    return arr


def pad_to_600bp(ohe_4x200: np.ndarray) -> np.ndarray:
    from experiments.train_malinois_k562 import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    left = one_hot_encode(MPRA_UPSTREAM[-200:])
    right = one_hot_encode(MPRA_DOWNSTREAM[:200])
    return np.concatenate([left, ohe_4x200, right], axis=1)


def predict_malinois(seqs, device):
    """Pretrained Malinois (K562 = output 0), 600bp padded."""
    from scripts.eval_pretrained_malinois import load_pretrained_malinois

    model = load_pretrained_malinois(
        str(REPO / "data" / "pretrained" / "malinois_trained" / "torch_checkpoint.pt"), device
    )
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            batch = seqs[i : i + 256]
            encoded = []
            for s in batch:
                ohe = one_hot_encode(s[:200])  # ensure exactly 200bp
                if ohe.shape[1] < 200:
                    pad = np.zeros((4, 200 - ohe.shape[1]), dtype=np.float32)
                    ohe = np.concatenate([ohe, pad], axis=1)
                encoded.append(pad_to_600bp(ohe))
            x = torch.from_numpy(np.stack(encoded)).float().to(device)
            out = model(x)[:, 0]
            x_rc = x.flip(-1)[:, [3, 2, 1, 0], :]
            out_rc = model(x_rc)[:, 0]
            preds.append(((out + out_rc) / 2).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(preds)


def predict_parm(seqs, device):
    """PARM K562 (5-fold ensemble), pads 200bp to 600bp with zeros."""
    sys.path.insert(0, str(REPO / "external" / "PARM"))
    from PARM.PARM_utils_load_model import load_PARM

    preds_all = []
    for fold in range(5):
        weight_path = (
            REPO / "external" / "PARM" / "pre_trained_models" / "K562" / f"K562_fold{fold}.parm"
        )
        model = load_PARM(weight_file=str(weight_path), L_max=600, train=False)
        model = model.to(device).eval()

        fold_preds = []
        with torch.no_grad():
            for i in range(0, len(seqs), 256):
                batch = seqs[i : i + 256]
                # PARM expects (B, 4, L) one-hot, pad to 600bp with zeros
                encoded = []
                for s in batch:
                    ohe = one_hot_encode(s)  # (4, 200)
                    padded = np.zeros((4, 600), dtype=np.float32)
                    # Center the 200bp in the 600bp window
                    start = (600 - len(s)) // 2
                    padded[:, start : start + len(s)] = ohe
                    encoded.append(padded)
                x = torch.from_numpy(np.stack(encoded)).float().to(device)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                fold_preds.append(out.squeeze().cpu().numpy())
        preds_all.append(np.concatenate(fold_preds))
        del model

    torch.cuda.empty_cache()
    return np.mean(preds_all, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = REPO / "outputs" / "pretrained_bias_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    # 1. Random DNA
    rng = np.random.default_rng(42)
    random_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(1000)]

    # 2. Agarwal shuffled controls
    shuffled_seqs = []
    controls_path = REPO / "data" / "agarwal_2025" / "k562_all_controls_200bp.tsv"
    with open(controls_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["category"] == "shuffled_negative":
                shuffled_seqs.append(row["sequence"])

    # 3. SNV pairs for delta
    import pandas as pd

    snv_path = REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        ref_seqs = snv_df["sequence_ref"].tolist()[:2000]
        alt_seqs = snv_df["sequence_alt"].tolist()[:2000]
        true_delta = snv_df["delta_log2FC"].values[:2000] if "delta_log2FC" in snv_df else None
    else:
        ref_seqs, alt_seqs, true_delta = [], [], None

    logger.info(
        "Sequences: %d random, %d shuffled, %d SNV pairs",
        len(random_seqs),
        len(shuffled_seqs),
        len(ref_seqs),
    )

    results = {}

    # ═══════════════════════════════════════════════
    # 1. Malinois (pretrained)
    # ═══════════════════════════════════════════════
    logger.info("Testing Malinois (pretrained)...")
    try:
        mal_random = predict_malinois(random_seqs, device)
        mal_shuffled = predict_malinois(shuffled_seqs, device)
        results["Malinois"] = {
            "random_mean": float(np.mean(mal_random)),
            "random_std": float(np.std(mal_random)),
            "shuffled_mean": float(np.mean(mal_shuffled)),
            "shuffled_std": float(np.std(mal_shuffled)),
        }
        if ref_seqs:
            mal_ref = predict_malinois(ref_seqs, device)
            mal_alt = predict_malinois(alt_seqs, device)
            mal_delta = mal_alt - mal_ref
            if true_delta is not None:
                r = float(pearsonr(mal_delta, true_delta)[0])
                results["Malinois"]["snv_delta_r"] = r
                results["Malinois"]["snv_delta_std_ratio"] = float(
                    np.std(mal_delta) / np.std(true_delta)
                )
        logger.info(
            "  Malinois: random=%.3f, shuffled=%.3f", np.mean(mal_random), np.mean(mal_shuffled)
        )
    except Exception as e:
        logger.warning("Malinois failed: %s", e)

    # ═══════════════════════════════════════════════
    # 2. PARM (K562, pretrained)
    # ═══════════════════════════════════════════════
    logger.info("Testing PARM (K562)...")
    try:
        parm_random = predict_parm(random_seqs, device)
        parm_shuffled = predict_parm(shuffled_seqs, device)
        results["PARM"] = {
            "random_mean": float(np.mean(parm_random)),
            "random_std": float(np.std(parm_random)),
            "shuffled_mean": float(np.mean(parm_shuffled)),
            "shuffled_std": float(np.std(parm_shuffled)),
        }
        if ref_seqs:
            parm_ref = predict_parm(ref_seqs, device)
            parm_alt = predict_parm(alt_seqs, device)
            parm_delta = parm_alt - parm_ref
            if true_delta is not None:
                r = float(pearsonr(parm_delta, true_delta)[0])
                results["PARM"]["snv_delta_r"] = r
                results["PARM"]["snv_delta_std_ratio"] = float(
                    np.std(parm_delta) / np.std(true_delta)
                )
        logger.info(
            "  PARM: random=%.3f, shuffled=%.3f", np.mean(parm_random), np.mean(parm_shuffled)
        )
    except Exception as e:
        logger.warning("PARM failed: %s", e)
        import traceback

        traceback.print_exc()

    # Save results
    with open(out_dir / "pretrained_bias_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PRETRAINED MODEL BIAS EVALUATION")
    print("=" * 70)
    print(f"\n{'Model':<20s} {'Random Mean':>12s} {'Shuffled Mean':>14s} {'SNV Delta r':>12s}")
    print("-" * 60)
    for model, vals in results.items():
        snv_r = vals.get("snv_delta_r", "N/A")
        snv_str = f"{snv_r:.4f}" if isinstance(snv_r, float) else snv_r
        print(
            f"{model:<20s} {vals['random_mean']:>12.3f} {vals['shuffled_mean']:>14.3f} {snv_str:>12s}"
        )
    print(f"\nReal MPRA shuffled controls: mean = -0.53")
    print(f"Expected for random DNA: ≈ -0.53 (based on shuffled controls)")
    print(f"\nSaved to {out_dir / 'pretrained_bias_results.json'}")


if __name__ == "__main__":
    main()
