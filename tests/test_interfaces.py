"""Interface-level tests for core ALBench abstractions."""

from __future__ import annotations

import numpy as np

from albench.acquisition.random_acq import RandomAcquisition
from albench.model import SequenceModel
from albench.reservoir.random_sampler import RandomSampler


class _DummyModel(SequenceModel):
    def predict(self, sequences: list[str]) -> np.ndarray:
        return np.asarray([float(len(s)) for s in sequences], dtype=np.float32)

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        return np.arange(len(sequences), dtype=np.float32)

    def embed(self, sequences: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(i), float(i) + 1.0] for i in range(len(sequences))], dtype=np.float32
        )

    def fit(self, sequences: list[str], labels: np.ndarray) -> None:
        return None


def test_sequence_model_interface_predict_shape() -> None:
    model = _DummyModel()
    preds = model.predict(["ACGT", "AAAA"])
    assert preds.shape == (2,)


def test_reservoir_sampler_returns_requested_count() -> None:
    sampler = RandomSampler(seed=1)
    idx = sampler.sample(["A", "C", "G", "T"], n_samples=2)
    assert len(idx) == 2


def test_acquisition_function_selects_requested_count() -> None:
    acq = RandomAcquisition(seed=1)
    selected = acq.select(_DummyModel(), ["A", "C", "G", "T"], n_select=3)
    assert selected.shape == (3,)
