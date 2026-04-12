#!/usr/bin/env python
"""Test post-hoc CpG correction on the oracle.

Applies correction: corrected = predicted - EXCESS_SLOPE * (seq_cpg - BASELINE_CPG)

This removes the spurious CpG→activity association without retraining.
Tests on: in-dist, OOD, SNV, random DNA, shuffled controls, intergenic.

Usage (on HPC):
    uv run --no-sync python scripts/analysis/test_cpg_correction.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# CpG correction parameters (from titration experiment)
# Oracle slope: 13.76 log2FC per unit CpG
# Real slope: ~0.9 (from Gosai ctrl_neg)
# Excess: 12.9
EXCESS_CPG_SLOPE = 12.9
BASELINE_CPG = 0.01  # typical genomic CpG frequency


def cpg_freq(seq: str) -> float:
    """Compute CpG dinucleotide frequency."""
    seq = seq.upper()
    if len(seq) < 2:
        return 0.0
    n_cpg = sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G")
    return n_cpg / (len(seq) - 1)


def apply_cpg_correction(predictions: np.ndarray, sequences: list[str]) -> np.ndarray:
    """Apply post-hoc CpG correction to oracle predictions."""
    cpg_freqs = np.array([cpg_freq(s) for s in sequences])
    corrections = EXCESS_CPG_SLOPE * (cpg_freqs - BASELINE_CPG)
    return predictions - corrections


def main():
    import glob

    import pandas as pd

    # Load oracle
    print("Loading oracle model...")
    from scripts.eval_neg_sweep_random_dna import load_s2_model, predict_sequences

    ckpt_dir = None
    for pattern in [
        "outputs/oracle_neg_sweep/baseline/fold_0",
        "outputs/ag_hashfrag_oracle_cached/oracle_0",
        "outputs/oracle_neg_sweep/frac005_elr1/fold_0",
    ]:
        p = REPO / pattern
        if (p / "best_model").exists():
            ckpt_dir = str(p)
            break
    if ckpt_dir is None:
        for p in sorted(glob.glob(str(REPO / "outputs/oracle_neg_sweep/*/fold_0/best_model"))):
            ckpt_dir = str(Path(p).parent)
            break

    if ckpt_dir is None:
        print("ERROR: No oracle checkpoint found")
        sys.exit(1)

    print(f"  Using: {ckpt_dir}")
    model, predict_step_fn, head_name = load_s2_model(ckpt_dir, "baseline")

    # ── Test sets ──────────────────────────────────────────────────
    results = {}

    # 1. Random DNA (500 seqs)
    rng = np.random.RandomState(42)
    random_seqs = ["".join("ACGT"[i] for i in rng.randint(0, 4, 200)) for _ in range(500)]
    raw = predict_sequences(model, predict_step_fn, random_seqs)
    corrected = apply_cpg_correction(raw, random_seqs)
    results["random_dna"] = {
        "raw_mean": float(np.mean(raw)),
        "corrected_mean": float(np.mean(corrected)),
        "raw_std": float(np.std(raw)),
        "corrected_std": float(np.std(corrected)),
        "n": len(raw),
    }

    # 2. Agarwal controls
    controls_path = REPO / "data/agarwal_2025/k562_all_controls_200bp.tsv"
    if controls_path.exists():
        ctrl_df = pd.read_csv(controls_path, sep="\t")
        for cat_name, key in [
            ("shuffled_negative", "shuffled"),
            ("ernst_negative", "intergenic"),
        ]:
            cat_seqs = ctrl_df[ctrl_df["category"] == cat_name]["sequence"].tolist()
            if cat_seqs:
                raw = predict_sequences(model, predict_step_fn, cat_seqs)
                corrected = apply_cpg_correction(raw, cat_seqs)
                results[key] = {
                    "raw_mean": float(np.mean(raw)),
                    "corrected_mean": float(np.mean(corrected)),
                    "raw_std": float(np.std(raw)),
                    "corrected_std": float(np.std(corrected)),
                    "n": len(raw),
                }

    # 3. Gosai ctrl_neg
    gosai = pd.read_csv(
        REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt",
        sep="\t",
        low_memory=False,
    )
    ctrl_neg = gosai[gosai["class"] == "ctrl_neg"]
    ctrl_seqs = ctrl_neg["sequence"].dropna().tolist()
    ctrl_real = ctrl_neg["K562_log2FC"].dropna().values
    raw = predict_sequences(model, predict_step_fn, ctrl_seqs)
    corrected = apply_cpg_correction(raw, ctrl_seqs)
    from scipy.stats import pearsonr

    r_raw, _ = pearsonr(ctrl_real[: len(raw)], raw[: len(ctrl_real)])
    r_corr, _ = pearsonr(ctrl_real[: len(corrected)], corrected[: len(ctrl_real)])
    results["gosai_ctrl_neg"] = {
        "raw_mean": float(np.mean(raw)),
        "corrected_mean": float(np.mean(corrected)),
        "real_mean": float(np.mean(ctrl_real)),
        "pearson_raw": float(r_raw),
        "pearson_corrected": float(r_corr),
        "n": len(raw),
    }

    # 4. In-dist test set
    test_dir = REPO / "data/k562/test_sets"
    for test_name, test_file in [
        ("in_dist", "test_in_dist_k562.tsv"),
        ("ood", "test_ood_designed_k562.tsv"),
    ]:
        tf = test_dir / test_file
        if not tf.exists():
            continue
        tdf = pd.read_csv(tf, sep="\t")
        test_seqs = tdf["sequence"].tolist()
        test_labels = tdf["K562_log2FC"].values
        raw = predict_sequences(model, predict_step_fn, test_seqs)
        corrected = apply_cpg_correction(raw, test_seqs)
        r_raw, _ = pearsonr(test_labels, raw)
        r_corr, _ = pearsonr(test_labels, corrected)
        results[test_name] = {
            "pearson_raw": float(r_raw),
            "pearson_corrected": float(r_corr),
            "delta": float(r_corr - r_raw),
            "n": len(raw),
        }

    # 5. SNV test set
    snv_file = test_dir / "test_snv_k562.tsv"
    if snv_file.exists():
        sdf = pd.read_csv(snv_file, sep="\t")
        ref_seqs = sdf["sequence_ref"].tolist()
        alt_seqs = sdf["sequence_alt"].tolist()
        ref_labels = sdf["K562_log2FC_ref"].values
        alt_labels = sdf["K562_log2FC_alt"].values

        ref_raw = predict_sequences(model, predict_step_fn, ref_seqs)
        alt_raw = predict_sequences(model, predict_step_fn, alt_seqs)
        ref_corr = apply_cpg_correction(ref_raw, ref_seqs)
        alt_corr = apply_cpg_correction(alt_raw, alt_seqs)

        # Absolute (ref+alt)
        all_raw = np.concatenate([ref_raw, alt_raw])
        all_corr = np.concatenate([ref_corr, alt_corr])
        all_labels = np.concatenate([ref_labels, alt_labels])
        r_abs_raw, _ = pearsonr(all_labels, all_raw)
        r_abs_corr, _ = pearsonr(all_labels, all_corr)

        # Delta
        delta_raw = alt_raw - ref_raw
        delta_corr = alt_corr - ref_corr
        delta_labels = alt_labels - ref_labels
        r_delta_raw, _ = pearsonr(delta_labels, delta_raw)
        r_delta_corr, _ = pearsonr(delta_labels, delta_corr)

        results["snv_abs"] = {
            "pearson_raw": float(r_abs_raw),
            "pearson_corrected": float(r_abs_corr),
            "delta": float(r_abs_corr - r_abs_raw),
            "n": len(all_raw),
        }
        results["snv_delta"] = {
            "pearson_raw": float(r_delta_raw),
            "pearson_corrected": float(r_delta_corr),
            "delta": float(r_delta_corr - r_delta_raw),
            "n": len(delta_raw),
        }

    # ── Print results ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"POST-HOC CpG CORRECTION: excess_slope={EXCESS_CPG_SLOPE}, baseline={BASELINE_CPG}")
    print("=" * 80)

    print("\n--- Bias metrics (mean prediction) ---")
    for key in ["random_dna", "shuffled", "intergenic", "gosai_ctrl_neg"]:
        if key in results:
            r = results[key]
            print(
                f"  {key:20s}: raw={r.get('raw_mean', 0):+.3f} -> corrected={r.get('corrected_mean', 0):+.3f}"
            )
            if "real_mean" in r:
                print(f"  {'':20s}  real={r['real_mean']:+.3f}")

    print("\n--- Quality metrics (Pearson r) ---")
    for key in ["in_dist", "ood", "snv_abs", "snv_delta", "gosai_ctrl_neg"]:
        if key in results:
            r = results[key]
            if "pearson_raw" in r:
                d = r.get("delta", r.get("pearson_corrected", 0) - r.get("pearson_raw", 0))
                print(
                    f"  {key:20s}: raw={r['pearson_raw']:.4f} -> corrected={r['pearson_corrected']:.4f} ({d:+.4f})"
                )

    # Save
    out_dir = REPO / "outputs" / "cpg_correction"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cpg_correction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'cpg_correction_results.json'}")


if __name__ == "__main__":
    main()
