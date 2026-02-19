"""Tests for Exp3 pool-size sensitivity helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.exp0_scaling import ExpRunArtifacts
from experiments.exp3_pool_size import _best_metric_row, _sensitivity_summary


def test_best_metric_row_extracts_best_result() -> None:
    """Best-metric helper should pick max metric row for one pool size."""
    curve = pd.DataFrame(
        [
            {"round_idx": 0, "n_labeled": 10, "test_set": "test", "pearson_r": 0.2},
            {"round_idx": 1, "n_labeled": 20, "test_set": "test", "pearson_r": 0.5},
        ]
    )
    artifacts = ExpRunArtifacts(results=[], curve=curve, output_dir=Path("/tmp"))
    row = _best_metric_row(pool_size=1000, artifacts=artifacts, selection_metric="pearson_r")
    assert row["pool_size"] == 1000
    assert row["best_round_idx"] == 1
    assert row["best_metric"] == 0.5


def test_sensitivity_summary_computes_range_and_slope() -> None:
    """Sensitivity summary should produce range and nonzero trend."""
    df = pd.DataFrame(
        [
            {"pool_size": 100, "best_metric": 0.1},
            {"pool_size": 1000, "best_metric": 0.2},
            {"pool_size": 10000, "best_metric": 0.4},
        ]
    )
    summary = _sensitivity_summary(df)
    assert int(summary.iloc[0]["n_pool_sizes"]) == 3
    assert float(summary.iloc[0]["metric_range"]) == pytest.approx(0.3)
    assert float(summary.iloc[0]["metric_per_log10_pool_slope"]) > 0.0
