"""Uncertainty acquisition function."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class UncertaintyAcquisition(AcquisitionFunction):
    """Select candidates with the largest uncertainty scores."""

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Rank by uncertainty and return top-k indices."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        scores = student.uncertainty(candidates)
        return np.argsort(scores)[-n_select:]
