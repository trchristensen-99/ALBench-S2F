"""Dataset-loader unit tests with patched file I/O."""

from __future__ import annotations

import numpy as np

from albench.data.k562 import K562Dataset
from albench.data.yeast import YeastDataset


def _fake_load_k562(self) -> None:
    self.sequences = np.asarray(["A" * 200, "C" * 200])
    self.labels = np.asarray([0.1, 0.2], dtype=np.float32)
    self.sequence_length = 200


def _fake_load_yeast(self) -> None:
    self.sequences = np.asarray(["A" * 150, "C" * 150])
    self.labels = np.asarray([0.1, 0.2], dtype=np.float32)
    self.is_singleton = np.asarray([0.0, 1.0], dtype=np.float32)
    self.sequence_length = 150


def test_k562_loader_patched(monkeypatch) -> None:
    monkeypatch.setattr(K562Dataset, "load_data", _fake_load_k562)
    ds = K562Dataset(data_path=".", split="train")
    assert len(ds) == 2
    assert ds.get_num_channels() == 5


def test_yeast_loader_patched(monkeypatch) -> None:
    monkeypatch.setattr(YeastDataset, "load_data", _fake_load_yeast)
    ds = YeastDataset(data_path=".", split="train")
    assert len(ds) == 2
    assert ds.get_num_channels() == 6
