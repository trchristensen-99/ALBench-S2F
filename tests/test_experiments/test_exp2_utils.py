"""Tests for Exp2 protocol comparison helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from experiments.exp0_scaling import ExpRunArtifacts
from experiments.exp2_rounds import _clone_cfg, _protocol_row


def test_clone_cfg_creates_independent_copy() -> None:
    """Config clone should be mutable without touching source config."""
    cfg = OmegaConf.create({"experiment": {"n_rounds": 2}})
    cloned = _clone_cfg(cfg)
    cloned.experiment.n_rounds = 7
    assert int(cfg.experiment.n_rounds) == 2
    assert int(cloned.experiment.n_rounds) == 7


def test_protocol_row_extracts_best_metric() -> None:
    """Protocol row should report best round using selection metric."""
    curve = pd.DataFrame(
        [
            {"round_idx": 0, "n_labeled": 10, "test_set": "test", "pearson_r": 0.3},
            {"round_idx": 1, "n_labeled": 20, "test_set": "test", "pearson_r": 0.6},
        ]
    )
    artifacts = ExpRunArtifacts(results=[], curve=curve, output_dir=Path("/tmp"))
    row = _protocol_row(
        name="multi_round",
        n_rounds=3,
        batch_size=10,
        artifacts=artifacts,
        selection_metric="pearson_r",
    )
    assert row["best_round_idx"] == 1
    assert row["best_metric"] == 0.6
