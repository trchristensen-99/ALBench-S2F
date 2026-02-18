"""Tests for AL loop schedule dispatch and round bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from albench.loop import RunConfig, run_al_loop
from albench.model import SequenceModel
from albench.task import TaskConfig


class _Oracle(SequenceModel):
    def predict(self, sequences: list[str]) -> np.ndarray:
        return np.asarray([float(len(s)) for s in sequences], dtype=np.float32)


class _Student(SequenceModel):
    def __init__(self) -> None:
        self.fit_calls = 0

    def predict(self, sequences: list[str]) -> np.ndarray:
        return np.zeros(len(sequences), dtype=np.float32)

    def fit(self, sequences: list[str], labels: np.ndarray) -> None:
        self.fit_calls += 1


@dataclass
class _Sampler:
    def sample(self, candidates: list[str], n_samples: int, metadata=None):
        return list(range(min(n_samples, len(candidates))))


@dataclass
class _Acquirer:
    def select(self, student: SequenceModel, candidates: list[str], n_select: int):
        return np.arange(min(n_select, len(candidates)))


def test_run_al_loop_runs_rounds_and_updates_labels(tmp_path) -> None:
    task = TaskConfig(
        name="k562", organism="human", sequence_length=200, data_root=".", test_set={}
    )
    oracle = _Oracle()
    student = _Student()
    run_cfg = RunConfig(
        n_rounds=2,
        batch_size=2,
        reservoir_schedule={"default": _Sampler()},
        acquisition_schedule={"default": _Acquirer()},
        output_dir=str(tmp_path),
        n_reservoir_candidates=2,
    )
    results = run_al_loop(task, oracle, student, ["AAAA", "CCCC", "GGGG"], run_cfg)
    assert len(results) == 2
    assert student.fit_calls == 3
