"""Prior-knowledge acquisition function."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class PriorKnowledgeAcquisition(AcquisitionFunction):
    """Combine sequence-prior scores for model-light acquisition.

    This strategy intentionally avoids model-feature uncertainty/embedding terms.
    It uses:
    - activity prior: favor high activity or a target activity
    - motif prior: favor presence of configured sequence motifs
    - GC prior: favor candidates near target GC fraction
    """

    def __init__(
        self,
        w_activity_prior: float = 0.5,
        w_motif_prior: float = 0.3,
        w_gc_prior: float = 0.2,
        target_activity: float | None = None,
        target_gc: float = 0.45,
        motifs: list[str] | None = None,
    ) -> None:
        """Initialize weighted prior-knowledge acquisition.

        Args:
            w_activity_prior: Weight on activity prior score.
            w_motif_prior: Weight on motif-presence score.
            w_gc_prior: Weight on GC-content proximity score.
            target_activity: Optional activity target. When omitted, high predicted
                activity is favored.
            target_gc: Desired GC fraction for candidates.
            motifs: Sequence motifs to up-weight. Case-insensitive.
        """
        total = w_activity_prior + w_motif_prior + w_gc_prior
        if total <= 0.0:
            raise ValueError("At least one weight must be > 0")
        self.w_activity_prior = w_activity_prior / total
        self.w_motif_prior = w_motif_prior / total
        self.w_gc_prior = w_gc_prior / total
        self.target_activity = target_activity
        self.target_gc = float(target_gc)
        self.motifs = [m.upper() for m in (motifs or ["TATA", "CACGTG", "GGGCGG"]) if m]

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Min-max normalize scores with constant-safe fallback."""
        arr = np.asarray(x, dtype=np.float32)
        denom = float(np.ptp(arr))
        if denom == 0.0:
            return np.zeros_like(arr)
        return (arr - float(np.min(arr))) / denom

    def _motif_scores(self, candidates: list[str]) -> np.ndarray:
        """Count motif hits per sequence (normalized later)."""
        if not self.motifs:
            return np.zeros(len(candidates), dtype=np.float32)
        scores = np.zeros(len(candidates), dtype=np.float32)
        for i, seq in enumerate(candidates):
            s = seq.upper()
            scores[i] = float(sum(s.count(motif) for motif in self.motifs))
        return scores

    @staticmethod
    def _gc_fraction(sequence: str) -> float:
        """Compute GC fraction for one sequence."""
        if not sequence:
            return 0.0
        s = sequence.upper()
        gc = s.count("G") + s.count("C")
        return float(gc) / float(len(s))

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select candidates by weighted composite score."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")

        predicted_activity = np.asarray(student.predict(candidates), dtype=np.float32)
        if self.target_activity is None:
            activity_prior = self._normalize(predicted_activity)
        else:
            activity_prior = 1.0 - self._normalize(
                np.abs(predicted_activity - float(self.target_activity))
            )

        motif_prior = self._normalize(self._motif_scores(candidates))
        gc_values = np.asarray([self._gc_fraction(seq) for seq in candidates], dtype=np.float32)
        gc_prior = 1.0 - self._normalize(np.abs(gc_values - self.target_gc))

        score = (
            +self.w_activity_prior * activity_prior
            + self.w_motif_prior * motif_prior
            + self.w_gc_prior * gc_prior
        )
        return np.argsort(score)[-n_select:]
