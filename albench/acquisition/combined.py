"""Combined uncertainty + diversity + activity-prior acquisition."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class CombinedAcquisition(AcquisitionFunction):
    """Weighted sum of normalized uncertainty/diversity/activity scores."""

    def __init__(
        self,
        alpha: float | None = None,
        w_uncertainty: float | None = None,
        w_diversity: float | None = None,
        w_activity_prior: float = 0.0,
        target_activity: float | None = None,
    ) -> None:
        """Initialize combination weight.

        Args:
            alpha: Backward-compatible uncertainty weight. When provided and
                explicit weights are omitted, uses ``alpha`` and ``1-alpha`` for
                uncertainty/diversity, with activity prior weight from
                ``w_activity_prior``.
            w_uncertainty: Explicit weight on uncertainty score.
            w_diversity: Explicit weight on diversity score.
            w_activity_prior: Weight on activity prior score.
            target_activity: Optional target for activity prior. If omitted,
                higher predicted activity gets higher score.
        """
        if w_uncertainty is None or w_diversity is None:
            if alpha is None:
                w_uncertainty = 0.5
                w_diversity = 0.5
            else:
                w_uncertainty = float(alpha)
                w_diversity = 1.0 - float(alpha)

        total = float(w_uncertainty + w_diversity + w_activity_prior)
        if total <= 0.0:
            raise ValueError("At least one combined-acquisition weight must be > 0")
        self.w_uncertainty = float(w_uncertainty) / total
        self.w_diversity = float(w_diversity) / total
        self.w_activity_prior = float(w_activity_prior) / total
        self.target_activity = target_activity

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Min-max normalize scores with constant-safe fallback."""
        arr = np.asarray(x, dtype=np.float32)
        denom = float(np.ptp(arr))
        if denom == 0.0:
            return np.zeros_like(arr)
        return (arr - float(np.min(arr))) / denom

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select indices by combined ranking score."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        unc = self._normalize(np.asarray(student.uncertainty(candidates), dtype=np.float32))
        emb = np.asarray(student.embed(candidates), dtype=np.float32)
        centroid = emb.mean(axis=0, keepdims=True)
        div = self._normalize(np.linalg.norm(emb - centroid, axis=1))

        pred = np.asarray(student.predict(candidates), dtype=np.float32)
        if self.target_activity is None:
            act = self._normalize(pred)
        else:
            act = 1.0 - self._normalize(np.abs(pred - float(self.target_activity)))

        score = self.w_uncertainty * unc + self.w_diversity * div + self.w_activity_prior * act
        return np.argsort(score)[-n_select:]
