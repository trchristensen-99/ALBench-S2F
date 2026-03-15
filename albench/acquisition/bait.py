"""BAIT: Bayesian Active learning by Imaginary Training."""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class BAITAcquisition(AcquisitionFunction):
    """Select candidates that maximally reduce expected posterior variance.

    Approximates the Fisher information gain criterion from Ash et al. (2021)
    by greedily selecting candidates whose uncertainty-weighted embeddings
    span the largest volume in feature space.  Uses QR factorization for
    numerical stability.

    Steps:
    1. Compute weighted features ``phi = emb * sqrt(uncertainty)``.
    2. Greedily pick the candidate with the largest ``||phi_i||^2``.
    3. Iteratively pick the candidate with the largest residual norm after
       projecting out the subspace already spanned by selected candidates
       (maintained via incremental QR factorization).
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select indices via greedy Fisher information maximization."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")

        # Build uncertainty-weighted feature vectors.
        embeddings = student.embed(candidates)  # (N, D)
        uncertainty = student.uncertainty(candidates)  # (N,)

        # Work in float64 for numerical stability.
        phi = np.asarray(embeddings, dtype=np.float64) * np.sqrt(
            np.asarray(uncertainty, dtype=np.float64)[:, None]
        )  # (N, D)

        n, d = phi.shape

        # Q stores the orthonormal basis of the selected subspace.
        # We build it column by column via modified Gram--Schmidt.
        Q = np.empty((d, 0), dtype=np.float64)  # (D, k)

        # Pre-compute squared norms (residual norms start as full norms).
        sq_norms = np.sum(phi**2, axis=1)  # (N,)

        selected: list[int] = []
        remaining_mask = np.ones(n, dtype=bool)

        for _ in range(n_select):
            # Score = residual squared norm for each unselected candidate.
            scores = sq_norms.copy()
            scores[~remaining_mask] = -np.inf

            idx = int(np.argmax(scores))
            selected.append(idx)
            remaining_mask[idx] = False

            # Update orthonormal basis Q with the new direction.
            v = phi[idx].copy()  # (D,)
            # Orthogonalize against existing Q columns.
            if Q.shape[1] > 0:
                coeffs = Q.T @ v  # (k,)
                v -= Q @ coeffs
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-12:
                v /= v_norm
                Q = np.column_stack([Q, v])  # (D, k+1)

                # Update residual squared norms for all remaining candidates.
                # Subtract the projection onto the new basis vector.
                projections = phi @ v  # (N,)
                sq_norms -= projections**2
                # Clamp to zero to avoid negative values from numerical noise.
                np.maximum(sq_norms, 0.0, out=sq_norms)

        return np.asarray(selected, dtype=np.int64)
