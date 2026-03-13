#!/usr/bin/env python
"""Generate K562 DREAM-RNN oracle pseudo-labels for test sets.

Loads the K562 DREAM-RNN oracle ensemble and generates oracle predictions
for all Experiment 1 test sets (in-distribution, SNV, OOD, random).

Usage::

    uv run --no-sync python experiments/generate_oracle_pseudolabels_k562_dream.py

Output::

    outputs/oracle_pseudolabels_k562_dream/
        test_in_dist_oracle_labels.npz
        test_snv_oracle_labels.npz
        test_ood_oracle_labels.npz
        test_random_10k_oracle_labels.npz
        summary.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]


def _safe_corr(a: np.ndarray, b: np.ndarray, fn: object) -> float:
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(fn(a, b)[0])


def _encode_k562(seq: str) -> np.ndarray:
    """Encode K562 200bp sequence to (5, 200) tensor."""
    from data.utils import one_hot_encode

    seq = seq.upper()
    if len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > 200:
        start = (len(seq) - 200) // 2
        seq = seq[start : start + 200]
    oh = one_hot_encode(seq, add_singleton_channel=False)
    rc = np.zeros((1, oh.shape[1]), dtype=np.float32)
    return np.concatenate([oh, rc], axis=0)


def _predict_ensemble(
    models: list[torch.nn.Module],
    sequences: list[str],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ensemble predictions. Returns (mean, std) arrays."""
    encoded = np.stack([_encode_k562(s) for s in sequences])

    all_preds = []
    for model in models:
        preds = []
        for i in range(0, len(encoded), batch_size):
            x = torch.from_numpy(encoded[i : i + batch_size]).float().to(device)
            with torch.no_grad():
                p = model.predict(x, use_reverse_complement=True)
                preds.append(p.cpu().numpy().reshape(-1))
        all_preds.append(np.concatenate(preds))

    arr = np.stack(all_preds, axis=0)
    return arr.mean(axis=0).astype(np.float32), arr.std(axis=0).astype(np.float32)


def main() -> None:
    import sys

    sys.path.insert(0, str(REPO))
    from models.dream_rnn import create_dream_rnn

    oracle_dir = Path(
        os.environ.get(
            "DREAM_ORACLE_DIR",
            str(REPO / "outputs" / "oracle_dream_rnn_k562_ensemble"),
        )
    )
    output_dir = Path(
        os.environ.get(
            "OUTPUT_DIR",
            str(REPO / "outputs" / "oracle_pseudolabels_k562_dream"),
        )
    )
    data_dir = Path(os.environ.get("K562_DATA_PATH", str(REPO / "data" / "k562")))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load oracle models
    models = []
    for run_dir in sorted(oracle_dir.glob("oracle_*")):
        ckpt = run_dir / "best_model.pt"
        if not ckpt.exists():
            continue
        m = create_dream_rnn(
            input_channels=5,
            sequence_length=200,
            task_mode="k562",
            hidden_dim=320,
            cnn_filters=160,
            dropout_cnn=0.1,
            dropout_lstm=0.1,
        )
        try:
            state = torch.load(ckpt, map_location="cpu")
        except RuntimeError as e:
            print(f"Skipping corrupt checkpoint {ckpt}: {e}")
            continue
        m.load_state_dict(state["model_state_dict"], strict=True)
        m.to(device).eval()
        models.append(m)
    print(f"Loaded {len(models)} oracle models from {oracle_dir}")

    if not models:
        raise FileNotFoundError(f"No oracle checkpoints found in {oracle_dir}")

    test_set_dir = data_dir / "test_sets"
    summary: dict[str, dict] = {}

    # In-distribution test set
    tsv = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if tsv.exists():
        df = pd.read_csv(tsv, sep="\t")
        seqs = df["sequence"].astype(str).tolist()
        true_labels = df["K562_log2FC"].to_numpy(dtype=np.float32)
        t0 = time.time()
        oracle_mean, oracle_std = _predict_ensemble(models, seqs, device)
        elapsed = time.time() - t0
        np.savez_compressed(
            output_dir / "test_in_dist_oracle_labels.npz",
            oracle_mean=oracle_mean,
            oracle_std=oracle_std,
            true_label=true_labels,
        )
        r = _safe_corr(oracle_mean, true_labels, pearsonr)
        summary["in_dist"] = {
            "n": len(seqs),
            "pearson_r": r,
            "time_s": elapsed,
        }
        print(f"  in_dist: {len(seqs):,} seqs, Pearson r={r:.4f} ({elapsed:.1f}s)")

    # SNV test set
    tsv = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if tsv.exists():
        df = pd.read_csv(tsv, sep="\t")
        ref_seqs = df["sequence_ref"].astype(str).tolist()
        alt_seqs = df["sequence_alt"].astype(str).tolist()
        true_delta = df["delta_log2FC"].to_numpy(dtype=np.float32)
        t0 = time.time()
        ref_mean, ref_std = _predict_ensemble(models, ref_seqs, device)
        alt_mean, alt_std = _predict_ensemble(models, alt_seqs, device)
        delta_mean = alt_mean - ref_mean
        elapsed = time.time() - t0
        np.savez_compressed(
            output_dir / "test_snv_oracle_labels.npz",
            ref_oracle_mean=ref_mean,
            alt_oracle_mean=alt_mean,
            delta_oracle_mean=delta_mean,
            alt_oracle_std=alt_std,
            true_delta=true_delta,
        )
        r = _safe_corr(delta_mean, true_delta, pearsonr)
        summary["snv"] = {
            "n_pairs": len(df),
            "delta_pearson_r": r,
            "time_s": elapsed,
        }
        print(f"  snv: {len(df):,} pairs, delta Pearson r={r:.4f} ({elapsed:.1f}s)")

    # OOD test set
    tsv = test_set_dir / "test_ood_designed_k562.tsv"
    if tsv.exists():
        df = pd.read_csv(tsv, sep="\t")
        seqs = df["sequence"].astype(str).tolist()
        true_labels = df["K562_log2FC"].to_numpy(dtype=np.float32)
        t0 = time.time()
        oracle_mean, oracle_std = _predict_ensemble(models, seqs, device)
        elapsed = time.time() - t0
        np.savez_compressed(
            output_dir / "test_ood_oracle_labels.npz",
            oracle_mean=oracle_mean,
            oracle_std=oracle_std,
            true_label=true_labels,
        )
        r = _safe_corr(oracle_mean, true_labels, pearsonr)
        summary["ood"] = {
            "n": len(seqs),
            "pearson_r": r,
            "time_s": elapsed,
        }
        print(f"  ood: {len(seqs):,} seqs, Pearson r={r:.4f} ({elapsed:.1f}s)")

    # Random 10K test set (generate same sequences as prepare_exp1_test_sets.py)
    rng = np.random.default_rng(42)
    nucleotides = np.array(list("ACGT"))
    indices = rng.integers(0, 4, size=(10_000, 200))
    random_seqs = ["".join(nucleotides[row]) for row in indices]
    t0 = time.time()
    oracle_mean, oracle_std = _predict_ensemble(models, random_seqs, device)
    elapsed = time.time() - t0
    np.savez_compressed(
        output_dir / "test_random_10k_oracle_labels.npz",
        oracle_mean=oracle_mean,
        oracle_std=oracle_std,
    )
    summary["random_10k"] = {
        "n": 10_000,
        "time_s": elapsed,
    }
    print(f"  random_10k: 10,000 seqs ({elapsed:.1f}s)")

    # Save summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved pseudolabels to {output_dir}")

    # -----------------------------------------------------------------------
    # Also create Exp 1 test set NPZs (same format as prepare_exp1_test_sets)
    # in data/k562/test_sets_dream/ so exp1_1_scaling.py can evaluate against
    # DREAM-RNN oracle labels.
    # -----------------------------------------------------------------------
    test_npz_dir = data_dir / "test_sets_dream"
    test_npz_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating Exp 1 test set NPZs in {test_npz_dir}")

    # In-distribution
    in_dist_pl = output_dir / "test_in_dist_oracle_labels.npz"
    in_dist_tsv = test_set_dir / "test_in_distribution_hashfrag.tsv"
    if in_dist_pl.exists() and in_dist_tsv.exists():
        pl = dict(np.load(in_dist_pl))
        df = pd.read_csv(in_dist_tsv, sep="\t")
        np.savez_compressed(
            test_npz_dir / "genomic_oracle.npz",
            sequences=df["sequence"].values,
            oracle_labels=pl["oracle_mean"],
            oracle_std=pl["oracle_std"],
            true_labels=pl["true_label"],
        )
        print(f"  genomic_oracle.npz: {len(df):,} sequences")

    # SNV
    snv_pl = output_dir / "test_snv_oracle_labels.npz"
    snv_tsv = test_set_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_pl.exists() and snv_tsv.exists():
        pl = dict(np.load(snv_pl))
        df = pd.read_csv(snv_tsv, sep="\t")
        np.savez_compressed(
            test_npz_dir / "snv_oracle.npz",
            ref_sequences=df["sequence_ref"].values,
            alt_sequences=df["sequence_alt"].values,
            ref_oracle_labels=pl["ref_oracle_mean"],
            alt_oracle_labels=pl["alt_oracle_mean"],
            delta_oracle_labels=pl["delta_oracle_mean"],
            alt_oracle_std=pl["alt_oracle_std"],
            true_delta=pl["true_delta"],
        )
        print(f"  snv_oracle.npz: {len(df):,} pairs")

    # OOD
    ood_pl = output_dir / "test_ood_oracle_labels.npz"
    ood_tsv = test_set_dir / "test_ood_designed_k562.tsv"
    if ood_pl.exists() and ood_tsv.exists():
        pl = dict(np.load(ood_pl))
        df = pd.read_csv(ood_tsv, sep="\t")
        np.savez_compressed(
            test_npz_dir / "ood_oracle.npz",
            sequences=df["sequence"].values,
            oracle_labels=pl["oracle_mean"],
            oracle_std=pl["oracle_std"],
            true_labels=pl["true_label"],
        )
        print(f"  ood_oracle.npz: {len(df):,} sequences")

    # Random 10K
    rand_pl = output_dir / "test_random_10k_oracle_labels.npz"
    if rand_pl.exists():
        pl = dict(np.load(rand_pl))
        np.savez_compressed(
            test_npz_dir / "random_10k_oracle.npz",
            sequences=np.array(random_seqs),
            oracle_labels=pl["oracle_mean"],
            oracle_std=pl["oracle_std"],
        )
        print(f"  random_10k_oracle.npz: 10,000 sequences")

    print(f"\nAll done. Test NPZs in {test_npz_dir}")


if __name__ == "__main__":
    main()
