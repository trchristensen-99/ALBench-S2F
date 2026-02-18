"""Random acquisition baseline."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class RandomAcquisition(AcquisitionFunction):
    """Select candidates uniformly at random."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize random acquisition.

        Args:
            seed: Random seed.
        """
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Return random candidate indices."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        return self._rng.choice(len(candidates), size=n_select, replace=False)
