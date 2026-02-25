"""Evaluation utilities for albench.

These functions are self-contained — they depend only on ``numpy`` and
``scipy``, so they can be used stand-alone without any application-layer code.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_corr(pred: np.ndarray, target: np.ndarray, fn: Any) -> float:
    """Compute a robust correlation metric, returning 0.0 for edge cases."""
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def evaluate_on_test_sets(student: Any, task: Any) -> dict[str, dict[str, float]]:
    """Evaluate a student model on all configured test sets.

    Args:
        student: Fitted model implementing ``predict(list[str]) -> np.ndarray``.
        task: Task configuration with a ``test_set`` attribute mapping test-set
            names to ``{"sequences": [...], "labels": [...]}`` dicts.  This is
            intentionally typed as ``Any`` so ``albench`` remains independent of
            the concrete ``TaskConfig`` class used by the caller.

    Returns:
        Mapping of test-set name to metric dict
        ``{"pearson_r": float, "spearman_r": float, "loss": float}``.
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


def compute_scaling_curve(results: list[Any]) -> Any:
    """Convert a list of round results into a tabular scaling curve.

    Requires ``pandas`` (not installed by default in the minimal albench
    environment — install ``albench[analysis]`` for this function).

    Args:
        results: List of :class:`~albench.loop.RoundResult` objects.

    Returns:
        ``pandas.DataFrame`` with one row per (round, test-set).
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_scaling_curve requires pandas. Install it with: pip install pandas"
        ) from exc

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
