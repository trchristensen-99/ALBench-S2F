"""BADGE: Batch Active learning by Diverse Gradient Embeddings."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class BADGEAcquisition(AcquisitionFunction):
    """Select candidates via k-means++ on uncertainty-scaled embeddings.

    Approximates the gradient embedding from Ash et al. (2020) by scaling
    each candidate's feature vector by its predictive uncertainty.  Running
    k-means++ initialisation on these scaled embeddings yields a batch that
    is jointly diverse *and* uncertain.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select indices via k-means++ on gradient embeddings."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")

        # Build approximate gradient embeddings.
        embeddings = student.embed(candidates)  # (N, D)
        uncertainty = student.uncertainty(candidates)  # (N,)
        grad_embs = embeddings * uncertainty[:, None]  # (N, D)

        # k-means++ initialisation to pick n_select diverse centres.
        rng = np.random.default_rng(self.seed)
        n = len(grad_embs)
        first = rng.integers(n)
        selected: list[int] = [int(first)]

        # min_dist[i] = squared distance from point i to nearest centre.
        min_dist = np.full(n, np.inf)
        _update_min_dist(min_dist, grad_embs, grad_embs[selected[-1]])

        while len(selected) < n_select:
            # Zero-out already-selected points so they aren't re-picked.
            probs = min_dist.copy()
            probs[selected] = 0.0
            total = probs.sum()
            if total == 0:
                # All remaining distances are zero; fall back to uniform.
                mask = np.ones(n, dtype=bool)
                mask[selected] = False
                remaining = np.where(mask)[0]
                idx = rng.choice(remaining)
            else:
                probs /= total
                idx = rng.choice(n, p=probs)
            selected.append(int(idx))
            _update_min_dist(min_dist, grad_embs, grad_embs[idx])

        return np.asarray(selected, dtype=np.int64)


def _update_min_dist(
    min_dist: np.ndarray,
    points: np.ndarray,
    new_centre: np.ndarray,
) -> None:
    """Update *min_dist* in-place with squared distances to *new_centre*."""
    sq_dist = np.sum((points - new_centre) ** 2, axis=1)
    np.minimum(min_dist, sq_dist, out=min_dist)
