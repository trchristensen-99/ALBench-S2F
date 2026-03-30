#!/usr/bin/env python3
"""Fix remaining bar plot anomalies by re-evaluating with correct labels/caches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.train_foundation_cached import MLPHead  # noqa: E402


def safe_corr(x, y, fn):
    m = np.isfinite(x) & np.isfinite(y)
    return float(fn(x[m], y[m])[0]) if m.sum() > 2 else 0.0


def make_metrics(pred, true):
    return {
        "pearson_r": safe_corr(pred, true, pearsonr),
        "spearman_r": safe_corr(pred, true, spearmanr),
        "mse": float(np.mean((pred[np.isfinite(true)] - true[np.isfinite(true)]) ** 2)),
        "n": int(np.isfinite(true).sum()),
    }


def fix_ntv3_s1_k562():
    """Re-evaluate NTv3 S1 K562 using cached embeddings against real labels."""
    print("\n=== FIX: NTv3 S1 K562 ===")
    cache = REPO / "outputs" / "ntv3_post_k562_cached" / "embedding_cache"

    from data.k562 import K562Dataset

    test_ds = K562Dataset(str(REPO / "data" / "k562"), split="test", label_column="K562_log2FC")
    test_labels = test_ds.labels.astype(np.float32)

    k562_snv = pd.read_csv(
        REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv", sep="\t"
    )
    alt_true = k562_snv["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    delta_true = k562_snv["delta_log2FC"].to_numpy(dtype=np.float32)

    ood_df = pd.read_csv(
        REPO / "data" / "k562" / "test_sets" / "test_ood_designed_k562.tsv", sep="\t"
    )
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    in_can = np.load(cache / "test_in_dist_canonical.npy").astype(np.float32)
    in_rc = np.load(cache / "test_in_dist_rc.npy").astype(np.float32)
    ref_can = np.load(cache / "test_snv_ref_canonical.npy").astype(np.float32)
    ref_rc = np.load(cache / "test_snv_ref_rc.npy").astype(np.float32)
    alt_can = np.load(cache / "test_snv_alt_canonical.npy").astype(np.float32)
    alt_rc = np.load(cache / "test_snv_alt_rc.npy").astype(np.float32)
    ood_can = np.load(cache / "test_ood_canonical.npy").astype(np.float32)
    ood_rc = np.load(cache / "test_ood_rc.npy").astype(np.float32)

    for seed_dir in sorted((REPO / "outputs" / "ntv3_post_k562_3seeds").iterdir()):
        ckpt_path = seed_dir / "best_model.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        head = MLPHead(1536, 512, dropout=0.1)
        head.load_state_dict(ckpt["model_state_dict"])
        head.eval()

        with torch.no_grad():
            in_p = (
                head(torch.tensor(in_can)).squeeze(-1).numpy()
                + head(torch.tensor(in_rc)).squeeze(-1).numpy()
            ) / 2
            ref_p = (
                head(torch.tensor(ref_can)).squeeze(-1).numpy()
                + head(torch.tensor(ref_rc)).squeeze(-1).numpy()
            ) / 2
            alt_p = (
                head(torch.tensor(alt_can)).squeeze(-1).numpy()
                + head(torch.tensor(alt_rc)).squeeze(-1).numpy()
            ) / 2
            ood_p = (
                head(torch.tensor(ood_can)).squeeze(-1).numpy()
                + head(torch.tensor(ood_rc)).squeeze(-1).numpy()
            ) / 2

        metrics = {
            "in_distribution": make_metrics(in_p, test_labels),
            "snv_abs": make_metrics(alt_p, alt_true),
            "snv_delta": make_metrics(alt_p - ref_p, delta_true),
            "ood": make_metrics(ood_p, ood_true),
        }

        rj = seed_dir / "result.json"
        r = json.loads(rj.read_text())
        r["test_metrics"] = metrics
        rj.write_text(json.dumps(r, indent=2, default=str))
        print(
            f"  {seed_dir.name}: ref={metrics['in_distribution']['pearson_r']:.4f} "
            f"snv={metrics['snv_abs']['pearson_r']:.4f} "
            f"delta={metrics['snv_delta']['pearson_r']:.4f} "
            f"ood={metrics['ood']['pearson_r']:.4f}"
        )


def fix_dream_rnn_ens3_k562():
    """Re-evaluate DREAM-RNN ens3 K562 with real labels."""
    print("\n=== FIX: DREAM-RNN ens3 K562 ===")
    from models.dream_rnn import DREAMRNN

    ckpt = torch.load(
        REPO
        / "outputs"
        / "dream_rnn_k562_with_preds"
        / "seed_42"
        / "seed_42"
        / "fraction_1.0000"
        / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = DREAMRNN(input_channels=5, seq_len=200, hidden_dim=128, cnn_filters=160)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def predict(seqs, bs=4096):
        preds = []
        for i in range(0, len(seqs), bs):
            batch = seqs[i : i + bs]
            encoded = []
            for seq in batch:
                if len(seq) < 200:
                    pad = 200 - len(seq)
                    seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
                elif len(seq) > 200:
                    start = (len(seq) - 200) // 2
                    seq = seq[start : start + 200]
                arr = np.zeros((5, 200), dtype=np.float32)
                for j, c in enumerate(seq.upper()):
                    if c in mapping:
                        arr[mapping[c], j] = 1.0
                encoded.append(arr)
            x = torch.tensor(np.stack(encoded))
            with torch.no_grad():
                preds.append(model(x).numpy().reshape(-1))
        return np.concatenate(preds)

    from data.k562 import K562Dataset

    test_ds = K562Dataset(str(REPO / "data" / "k562"), split="test", label_column="K562_log2FC")
    test_labels = test_ds.labels.astype(np.float32)

    k562_snv = pd.read_csv(
        REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv", sep="\t"
    )
    ood_df = pd.read_csv(
        REPO / "data" / "k562" / "test_sets" / "test_ood_designed_k562.tsv", sep="\t"
    )

    print("  Predicting on %d test sequences..." % len(test_ds))
    in_p = predict(list(test_ds.sequences))
    ref_p = predict(k562_snv["sequence_ref"].tolist())
    alt_p = predict(k562_snv["sequence_alt"].tolist())
    ood_p = predict(ood_df["sequence"].tolist())

    alt_true = k562_snv["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    delta_true = k562_snv["delta_log2FC"].to_numpy(dtype=np.float32)
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    metrics = {
        "in_distribution": make_metrics(in_p, test_labels),
        "snv_abs": make_metrics(alt_p, alt_true),
        "snv_delta": make_metrics(alt_p - ref_p, delta_true),
        "ood": make_metrics(ood_p, ood_true),
    }

    rj = (
        REPO
        / "outputs"
        / "dream_rnn_k562_with_preds"
        / "seed_42"
        / "seed_42"
        / "fraction_1.0000"
        / "result.json"
    )
    r = json.loads(rj.read_text())
    r["test_metrics"] = metrics
    rj.write_text(json.dumps(r, indent=2, default=str))
    print(
        f"  ref={metrics['in_distribution']['pearson_r']:.4f} "
        f"snv={metrics['snv_abs']['pearson_r']:.4f} "
        f"delta={metrics['snv_delta']['pearson_r']:.4f} "
        f"ood={metrics['ood']['pearson_r']:.4f} "
        f"(n={metrics['in_distribution']['n']})"
    )


if __name__ == "__main__":
    fix_ntv3_s1_k562()
    fix_dream_rnn_ens3_k562()
    print("\nDone! AG all-folds S1 HepG2/SKNSH SNV needs JAX-based eval (separate script).")
