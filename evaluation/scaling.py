"""Scaling-curve utilities for the ALBench-S2F application layer.

``evaluate_on_test_sets`` is defined in ``albench.evaluation`` (the standalone
AL engine).  This module re-exports it for backward compatibility and adds
``compute_scaling_curve`` which is specific to the experimental report format.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

# Re-export from the core engine so callers can use either location.
from albench.evaluation import evaluate_on_test_sets  # noqa: F401


def compute_scaling_curve(results: list[Any]) -> pd.DataFrame:
    """Convert round results into a tabular scaling curve.

    Args:
        results: List of loop round results (e.g. :class:`~albench.loop.RoundResult`).

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
