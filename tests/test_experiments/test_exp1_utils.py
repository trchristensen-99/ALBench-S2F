"""Tests for Exp1 benchmark helper utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from experiments.exp0_scaling import ExpRunArtifacts
from experiments.exp1_benchmark import _best_metric_row, _load_component_cfg


def test_load_component_cfg_reads_yaml() -> None:
    """Component config loader should return a Hydra DictConfig."""
    cfg = _load_component_cfg("acquisition", "random")
    assert isinstance(cfg, DictConfig)
    assert str(cfg["_target_"]).endswith("RandomAcquisition")


def test_best_metric_row_extracts_combination_summary() -> None:
    """Best-metric helper should pick max row for selected metric."""
    curve = pd.DataFrame(
        [
            {"round_idx": 0, "n_labeled": 10, "test_set": "test", "pearson_r": 0.1},
            {"round_idx": 1, "n_labeled": 20, "test_set": "test", "pearson_r": 0.4},
        ]
    )
    artifacts = ExpRunArtifacts(results=[], curve=curve, output_dir=Path("/tmp"))
    row = _best_metric_row("random", "uncertainty", artifacts, "pearson_r")
    assert row["reservoir"] == "random"
    assert row["acquisition"] == "uncertainty"
    assert row["best_round_idx"] == 1
    assert row["best_metric"] == 0.4
