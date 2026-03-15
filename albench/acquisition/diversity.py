"""Diversity-driven acquisition based on student embeddings."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class DiversityAcquisition(AcquisitionFunction):
    """Greedy farthest-point (LCMD) selection in embedding space.

    Maintains minimum distances incrementally for O(N*k) total cost
    instead of the naive O(N*k^2) approach.

    Parameters
    ----------
    seed : int | None
        RNG seed used to pick the first point.  When ``None``, the
        first point is always index 0 for backward compatibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select a diverse subset of candidate indices."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")

        embeddings = student.embed(candidates)
        n = len(embeddings)

        # Pick first point
        if self.seed is not None:
            first = int(np.random.default_rng(self.seed).integers(n))
        else:
            first = 0

        selected: list[int] = [first]

        # min_dist[i] = squared distance from point i to nearest selected point
        min_dist = np.full(n, np.inf, dtype=np.float64)
        _update_min_sq_dist(min_dist, embeddings, embeddings[first])

        while len(selected) < n_select:
            min_dist[selected[-1]] = -np.inf  # exclude just-added
            best = int(np.argmax(min_dist))
            selected.append(best)
            _update_min_sq_dist(min_dist, embeddings, embeddings[best])

        return np.asarray(selected, dtype=np.int64)


def _update_min_sq_dist(
    min_dist: np.ndarray,
    points: np.ndarray,
    new_centre: np.ndarray,
) -> None:
    """Update *min_dist* in-place with squared L2 distances to *new_centre*."""
    sq_dist = np.sum((points - new_centre) ** 2, axis=1)
    np.minimum(min_dist, sq_dist, out=min_dist)
