"""Tests for lightweight experiment utility functions."""

from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from experiments.exp4_cost import _configured_cost_map, _reservoir_strategy_key
from experiments.exp5_best_student import _pick_best_row, _resolve_selection_metric


def test_reservoir_strategy_key() -> None:
    """Strategy key should normalize configured reservoir target class."""
    cfg = OmegaConf.create({"reservoir": {"_target_": "albench.reservoir.genomic.GenomicSampler"}})
    assert _reservoir_strategy_key(cfg) == "genomicsampler"


def test_configured_cost_map_override() -> None:
    """Custom synthesis costs should override defaults."""
    cfg = OmegaConf.create(
        {
            "experiment": {
                "synthesis_cost_per_sequence": {
                    "randomsampler": 1.2,
                    "genomicsampler": 0.3,
                }
            }
        }
    )
    cost_map = _configured_cost_map(cfg)
    assert cost_map["randomsampler"] == pytest.approx(1.2)
    assert cost_map["genomicsampler"] == pytest.approx(0.3)


def test_resolve_selection_metric_default_and_override() -> None:
    """Selection metric should default to pearson_r and allow override."""
    cfg_default = OmegaConf.create({"experiment": {}})
    cfg_override = OmegaConf.create({"experiment": {"selection_metric": "spearman_r"}})
    assert _resolve_selection_metric(cfg_default) == "pearson_r"
    assert _resolve_selection_metric(cfg_override) == "spearman_r"


def test_pick_best_row_by_metric() -> None:
    """Best-row picker should respect selected metric column."""
    curve = pd.DataFrame(
        [
            {"round_idx": 0, "pearson_r": 0.2, "spearman_r": 0.3},
            {"round_idx": 1, "pearson_r": 0.5, "spearman_r": 0.1},
        ]
    )
    best = _pick_best_row(curve, "spearman_r")
    assert int(best["round_idx"]) == 0
