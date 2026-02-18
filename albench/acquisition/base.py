"""Acquisition function interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from albench.model import SequenceModel


class AcquisitionFunction(ABC):
    """Base interface for active learning acquisition functions."""

    @abstractmethod
    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select indices from candidates."""
