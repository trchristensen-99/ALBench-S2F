"""Combined uncertainty + diversity acquisition."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class CombinedAcquisition(AcquisitionFunction):
    """Weighted sum of normalized uncertainty and diversity scores."""

    def __init__(self, alpha: float = 0.5) -> None:
        """Initialize combination weight.

        Args:
            alpha: Weight for uncertainty; ``1-alpha`` for diversity.
        """
        self.alpha = alpha

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select indices by combined ranking score."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        unc = student.uncertainty(candidates)
        emb = student.embed(candidates)
        centroid = emb.mean(axis=0, keepdims=True)
        div = np.linalg.norm(emb - centroid, axis=1)

        def normalize(x: np.ndarray) -> np.ndarray:
            denom = np.ptp(x)
            if denom == 0:
                return np.zeros_like(x)
            return (x - np.min(x)) / denom

        score = self.alpha * normalize(unc) + (1.0 - self.alpha) * normalize(div)
        return np.argsort(score)[-n_select:]
