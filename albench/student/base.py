"""Student interface base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from albench.model import SequenceModel


class Student(SequenceModel, ABC):
    """Abstract student model interface used by active learning loops."""

    @abstractmethod
    def fit(self, sequences: list[str], labels: np.ndarray) -> None:
        """Train or update student parameters."""
