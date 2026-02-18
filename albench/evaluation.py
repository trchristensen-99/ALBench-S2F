"""Evaluation and scaling-curve utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from albench.task import TaskConfig


def _safe_corr(pred: np.ndarray, target: np.ndarray, fn: Any) -> float:
    """Compute a robust correlation metric for small/constant vectors."""
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def evaluate_on_test_sets(student: Any, task: TaskConfig) -> dict[str, dict[str, float]]:
    """Evaluate student predictions on configured test sets.

    Args:
        student: Fitted student model implementing ``predict``.
        task: Task configuration containing test-set references.

    Returns:
        Mapping of test-set name to metric dictionary.
    """
    metrics: dict[str, dict[str, float]] = {}
    for name, payload in task.test_set.items():
        sequences = payload.get("sequences", [])
        labels = np.asarray(payload.get("labels", []), dtype=np.float32)
        if len(sequences) == 0:
            continue
        preds = np.asarray(student.predict(list(sequences)), dtype=np.float32)
        mse = float(np.mean((preds - labels) ** 2))
        metrics[name] = {
            "pearson_r": _safe_corr(preds, labels, pearsonr),
            "spearman_r": _safe_corr(preds, labels, spearmanr),
            "loss": mse,
        }
    return metrics


def compute_scaling_curve(results: list[Any]) -> pd.DataFrame:
    """Convert round results into a tabular scaling curve.

    Args:
        results: List of loop round results.

    Returns:
        DataFrame with one row per (round, test-set).
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        round_dict = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
        for test_name, test_metrics in round_dict.get("test_metrics", {}).items():
            row = {
                "round_idx": round_dict["round_idx"],
                "n_labeled": round_dict["n_labeled"],
                "test_set": test_name,
            }
            row.update(test_metrics)
            rows.append(row)
    return pd.DataFrame(rows)
