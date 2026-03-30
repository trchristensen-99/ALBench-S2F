"""Evaluation utilities for Experiment 1 oracle-labeled test sets.

Loads oracle-labeled NPZ files and evaluates student model predictions
against oracle ground-truth labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_corr(a: np.ndarray, b: np.ndarray, fn: Any) -> float:
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    try:
        return float(fn(a, b)[0])
    except Exception:
        return 0.0


def load_oracle_test_set(path: Path | str) -> dict[str, np.ndarray]:
    """Load an oracle-labeled NPZ test set."""
    return dict(np.load(str(path), allow_pickle=True))


def evaluate_predictions(
    predictions: np.ndarray,
    oracle_labels: np.ndarray,
) -> dict[str, float]:
    """Compute metrics comparing student predictions to oracle labels."""
    mask = np.isfinite(predictions) & np.isfinite(oracle_labels)
    p, o = predictions[mask], oracle_labels[mask]
    return {
        "pearson_r": _safe_corr(p, o, pearsonr),
        "spearman_r": _safe_corr(p, o, spearmanr),
        "mse": float(np.mean((p - o) ** 2)),
        "n": int(mask.sum()),
    }


def evaluate_snv(
    ref_predictions: np.ndarray,
    alt_predictions: np.ndarray,
    ref_oracle: np.ndarray,
    alt_oracle: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate SNV absolute and delta predictions."""
    # Absolute: both ref and alt expression
    pred_abs = np.concatenate([ref_predictions, alt_predictions])
    oracle_abs = np.concatenate([ref_oracle, alt_oracle])
    abs_metrics = evaluate_predictions(pred_abs, oracle_abs)

    # Delta: alt - ref
    pred_delta = alt_predictions - ref_predictions
    oracle_delta = alt_oracle - ref_oracle
    delta_metrics = evaluate_predictions(pred_delta, oracle_delta)

    return {"snv_abs": abs_metrics, "snv_delta": delta_metrics}


def evaluate_on_exp1_test_panel(
    student: Any,
    task: str,
    test_set_dir: Path | str,
    batch_size: int = 4096,
) -> dict[str, dict[str, float]]:
    """Evaluate a student model on the full Experiment 1 test panel.

    Args:
        student: Model with ``predict(list[str]) -> np.ndarray`` method.
        task: ``"k562"`` or ``"yeast"``.
        test_set_dir: Directory containing ``*_oracle.npz`` files.
        batch_size: Batch size for prediction (to avoid OOM).

    Returns:
        Dict mapping test set name to metrics dict.
    """
    test_dir = Path(test_set_dir)
    results: dict[str, dict[str, float]] = {}

    def _predict_batched(sequences: list[str] | np.ndarray) -> np.ndarray:
        seqs = [str(s) for s in sequences]
        preds_list = []
        for i in range(0, len(seqs), batch_size):
            preds_list.append(student.predict(seqs[i : i + batch_size]))
        return np.concatenate(preds_list)

    # In-distribution test set
    if task == "k562":
        id_name = "genomic"
    else:
        id_name = "random"

    id_path = test_dir / f"{id_name}_oracle.npz"
    if id_path.exists():
        data = load_oracle_test_set(id_path)
        preds = _predict_batched(data["sequences"])
        results["in_dist"] = evaluate_predictions(preds, data["oracle_labels"])
        if "true_labels" in data:
            results["in_dist_real"] = evaluate_predictions(preds, data["true_labels"])

    # OOD test set
    if task == "k562":
        ood_path = test_dir / "ood_oracle.npz"
    else:
        ood_path = test_dir / "genomic_oracle.npz"

    if ood_path.exists():
        data = load_oracle_test_set(ood_path)
        preds = _predict_batched(data["sequences"])
        results["ood"] = evaluate_predictions(preds, data["oracle_labels"])
        if "true_labels" in data:
            results["ood_real"] = evaluate_predictions(preds, data["true_labels"])

    # SNV test set
    snv_path = test_dir / "snv_oracle.npz"
    if snv_path.exists():
        data = load_oracle_test_set(snv_path)
        ref_preds = _predict_batched(data["ref_sequences"])
        alt_preds = _predict_batched(data["alt_sequences"])
        snv_results = evaluate_snv(
            ref_preds, alt_preds, data["ref_oracle_labels"], data["alt_oracle_labels"]
        )
        results.update(snv_results)
        # Also evaluate SNV delta against real labels if available
        if "true_delta" in data:
            pred_delta = alt_preds - ref_preds
            results["snv_delta_real"] = evaluate_predictions(pred_delta, data["true_delta"])

    # K562 random sequences (5th test set)
    if task == "k562":
        rand_path = test_dir / "random_10k_oracle.npz"
        if rand_path.exists():
            data = load_oracle_test_set(rand_path)
            if np.isfinite(data["oracle_labels"]).any():
                preds = _predict_batched(data["sequences"])
                results["random"] = evaluate_predictions(preds, data["oracle_labels"])

    return results
