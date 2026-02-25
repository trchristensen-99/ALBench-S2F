"""Utilities for yeast test-subset evaluation (random, SNV, genomic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _safe_corr(a: np.ndarray, b: np.ndarray, fn: Any) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    try:
        return float(fn(a, b)[0])
    except Exception:
        return 0.0


def _read_single_index_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "pos" not in df.columns:
        raise ValueError(f"Missing 'pos' column in {path}")
    return df["pos"].dropna().astype(int).to_numpy()


def _read_pair_index_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "ref_pos" not in df.columns or "alt_pos" not in df.columns:
        raise ValueError(f"Missing 'ref_pos'/'alt_pos' columns in {path}")
    pairs = np.stack(
        [df["ref_pos"].astype(int).to_numpy(), df["alt_pos"].astype(int).to_numpy()], axis=1
    )
    return pairs


def _load_public_single_indices(public_dir: Path) -> set[int]:
    files = [
        "high_exp_indices.json",
        "low_exp_indices.json",
        "yeast_exp_indices.json",
        "random_exp_indices.json",
        "challenging_exp_indices.json",
    ]
    indices: set[int] = set()
    for name in files:
        path = public_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for key in payload.keys():
            indices.add(int(key))
    return indices


def _load_public_pair_indices(public_dir: Path, filename: str) -> set[tuple[int, int]]:
    path = public_dir / filename
    pairs: set[tuple[int, int]] = set()
    if not path.exists():
        return pairs
    payload = json.loads(path.read_text())
    for key in payload.keys():
        left, right = key.split(",")
        pairs.add((int(left), int(right)))
    return pairs


def load_yeast_test_subsets(
    subset_dir: str | Path,
    public_dir: str | Path | None = None,
    use_private_only: bool = False,
) -> dict[str, np.ndarray]:
    """Load subset definitions for yeast/random/SNV test evaluation.

    Args:
        subset_dir: Directory containing subset CSV files from old repo.
        public_dir: Optional directory containing public leaderboard JSON index files.
        use_private_only: If True, remove public leaderboard indices/pairs.

    Returns:
        Dictionary with keys: ``random_idx``, ``genomic_idx``, ``snv_pairs``.
    """
    subset_path = Path(subset_dir)
    random_idx = _read_single_index_csv(subset_path / "all_random_seqs.csv")
    genomic_idx = _read_single_index_csv(subset_path / "yeast_seqs.csv")
    snv_pairs = _read_pair_index_csv(subset_path / "all_SNVs_seqs.csv")

    if use_private_only and public_dir is not None:
        pdir = Path(public_dir)
        public_single = _load_public_single_indices(pdir)
        public_snv_pairs = _load_public_pair_indices(pdir, "SNVs_exp_indices.json")

        if public_single:
            random_idx = np.array([i for i in random_idx if i not in public_single], dtype=int)
            genomic_idx = np.array([i for i in genomic_idx if i not in public_single], dtype=int)

        if public_snv_pairs:
            snv_pairs = np.array(
                [p for p in snv_pairs if (int(p[0]), int(p[1])) not in public_snv_pairs], dtype=int
            )

    return {
        "random_idx": random_idx,
        "genomic_idx": genomic_idx,
        "snv_pairs": snv_pairs,
    }


def evaluate_yeast_test_subsets(
    predictions: np.ndarray,
    labels: np.ndarray,
    subsets: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Evaluate predictions on yeast random/SNV/genomic subsets.

    Args:
        predictions: Model predictions aligned to full yeast test file rows.
        labels: Ground-truth labels aligned to full yeast test file rows.
        subsets: Output of ``load_yeast_test_subsets``.

    Returns:
        Nested metric dictionary keyed by subset name.
        `snv` is variant-effect delta Pearson. `snv_abs` is raw-expression Pearson
        across ref+alt alleles in the SNV subset.
    """
    out: dict[str, dict[str, float]] = {}

    random_idx = subsets["random_idx"]
    genomic_idx = subsets["genomic_idx"]
    snv_pairs = subsets["snv_pairs"]

    for name, idx in [("random", random_idx), ("genomic", genomic_idx)]:
        pred = predictions[idx]
        true = labels[idx]
        out[name] = {
            "pearson_r": _safe_corr(true, pred, pearsonr),
            "spearman_r": _safe_corr(true, pred, spearmanr),
            "n": int(len(idx)),
        }

    if len(snv_pairs) > 0:
        ref = snv_pairs[:, 0].astype(int)
        alt = snv_pairs[:, 1].astype(int)

        # Raw SNV expression metric across both alleles.
        pred_abs = np.concatenate([predictions[ref], predictions[alt]], axis=0)
        true_abs = np.concatenate([labels[ref], labels[alt]], axis=0)
        out["snv_abs"] = {
            "pearson_r": _safe_corr(true_abs, pred_abs, pearsonr),
            "spearman_r": _safe_corr(true_abs, pred_abs, spearmanr),
            "n": int(2 * len(snv_pairs)),
        }

        # Variant-effect metric (delta alt-ref), retained as primary SNV effect score.
        pred_delta = predictions[alt] - predictions[ref]
        true_delta = labels[alt] - labels[ref]
        out["snv"] = {
            "pearson_r": _safe_corr(true_delta, pred_delta, pearsonr),
            "spearman_r": _safe_corr(true_delta, pred_delta, spearmanr),
            "n": int(len(snv_pairs)),
        }
    else:
        out["snv_abs"] = {"pearson_r": 0.0, "spearman_r": 0.0, "n": 0}
        out["snv"] = {"pearson_r": 0.0, "spearman_r": 0.0, "n": 0}

    return out
