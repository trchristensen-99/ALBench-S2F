"""Tests for the AL loop driver."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from albench.loop import RoundResult, RunConfig, run_al_loop
from albench.task import TaskConfig

# ---- Minimal test doubles -------------------------------------------------


class _FakeOracle:
    """Oracle that returns a hash-based label."""

    def predict(self, sequences: list[str]) -> np.ndarray:
        return np.array([float(hash(s) % 100) / 100.0 for s in sequences], dtype=np.float32)


class _FakeStudent:
    """Student that tracks fit calls."""

    def __init__(self) -> None:
        self.fit_count = 0
        self._fitted_on: list[str] = []

    def predict(self, sequences: list[str]) -> np.ndarray:
        return np.zeros(len(sequences), dtype=np.float32)

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        return np.random.default_rng(0).uniform(size=len(sequences)).astype(np.float32)

    def embed(self, sequences: list[str]) -> np.ndarray:
        return np.random.default_rng(0).normal(size=(len(sequences), 16)).astype(np.float32)

    def fit(self, sequences: list[str], labels: np.ndarray) -> None:
        self.fit_count += 1
        self._fitted_on = list(sequences)


class _FakeSampler:
    """Reservoir that returns sequential indices."""

    def sample(self, candidates: list[str], n_samples: int, metadata: Any = None) -> list[int]:
        return list(range(min(n_samples, len(candidates))))


class _FakeAcquirer:
    """Acquisition that returns the first k candidates."""

    def select(self, student: Any, candidates: list[str], n_select: int) -> list[int]:
        return list(range(min(n_select, len(candidates))))


# ---- Tests -----------------------------------------------------------------


@pytest.fixture()
def test_task(tmp_path: Path) -> TaskConfig:
    """Create a minimal TaskConfig for testing."""
    return TaskConfig(
        name="test",
        organism="test_organism",
        sequence_length=10,
        data_root=str(tmp_path),
        test_set={
            "simple": {
                "sequences": ["AAAAAAAAAA", "CCCCCCCCCC"],
                "labels": [0.5, 0.8],
            }
        },
    )


@pytest.fixture()
def run_config(tmp_path: Path) -> RunConfig:
    """Create a minimal RunConfig."""
    return RunConfig(
        n_rounds=2,
        batch_size=4,
        reservoir_schedule={"default": _FakeSampler()},
        acquisition_schedule={"default": _FakeAcquirer()},
        output_dir=str(tmp_path / "run_outputs"),
        n_reservoir_candidates=16,
    )


@patch("albench.loop._WANDB_AVAILABLE", False)
def test_run_al_loop_basic(test_task: TaskConfig, run_config: RunConfig) -> None:
    """Verify AL loop produces expected number of results and runs student.fit."""

    oracle = _FakeOracle()
    student = _FakeStudent()
    initial = ["ACGTACGTAC", "GCTAGCTAGC", "TATATATATAT", "GGGGGGGGGG"]

    results = run_al_loop(
        task=test_task,
        oracle=oracle,
        student=student,
        initial_labeled=initial,
        run_config=run_config,
        pool_sequences=[
            "AACCTGGTTC",
            "CCGGAATTCC",
            "TTAAGGCCTT",
            "GGCCAATTGG",
            "AATTCCGGAA",
            "CCAATTGGCC",
            "TTGGAACCTT",
            "GGTTCCAAGG",
        ],
    )

    # Should have results for each round
    assert len(results) == run_config.n_rounds
    # Student should be fitted once initially + once per round
    assert student.fit_count == 1 + run_config.n_rounds


@patch("albench.loop._WANDB_AVAILABLE", False)
def test_run_al_loop_round_metadata(test_task: TaskConfig, run_config: RunConfig) -> None:
    """Verify each RoundResult has correct metadata."""

    results = run_al_loop(
        task=test_task,
        oracle=_FakeOracle(),
        student=_FakeStudent(),
        initial_labeled=["ACGTACGTAC", "GCTAGCTAGC"],
        run_config=run_config,
        pool_sequences=["AACCTG" * 2, "CCGGAA" * 2, "TTAAGG" * 2, "GGCCAA" * 2] * 2,
    )

    for r in results:
        assert isinstance(r, RoundResult)
        assert r.n_labeled > 0
        assert isinstance(r.selected_sequences, list)
        assert isinstance(r.test_metrics, dict)
