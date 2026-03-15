"""BatchBALD: Batch Bayesian Active Learning by Disagreement.

Greedy approximation to joint mutual information maximisation.
For regression, uses MC dropout samples to estimate predictive variance
and selects batches that are jointly diverse in prediction space.

Reference: Kirsch, van Amersfoort & Gal (2019).
"""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class BatchBALDAcquisition(AcquisitionFunction):
    """Select candidates that jointly maximise mutual information.

    Uses MC-dropout (or ensemble) prediction samples to build a batch
    where each new point adds maximal information beyond what was
    already selected.  Greedy selection in O(k * N * T) time, where
    *k* = ``n_select``, *N* = candidate count, *T* = number of MC
    samples.

    Parameters
    ----------
    n_mc_samples : int
        Number of MC forward passes (default 30).  If the student is
        an ensemble wrapper this is ignored — ensemble members are used
        directly.
    seed : int | None
        RNG seed for tie-breaking.
    """

    def __init__(self, n_mc_samples: int = 30, seed: int | None = None) -> None:
        self.n_mc_samples = n_mc_samples
        self.seed = seed

    # ------------------------------------------------------------------

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        """Select indices via greedy joint-entropy maximisation."""
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")

        rng = np.random.default_rng(self.seed)

        # Collect MC prediction samples: shape (T, N)
        samples = self._gather_samples(student, candidates)
        n_cand = samples.shape[1]

        # Marginal variance per candidate
        var = samples.var(axis=0)  # (N,)

        # Greedy selection: pick the candidate whose *conditional* variance
        # (given what we've already selected) is largest.
        #
        # For Gaussian-approximated BatchBALD the conditional variance of
        # candidate j given selected set S is:
        #   Var(y_j | S) = Var(y_j) - Cov(y_j, y_S) @ Cov(y_S, y_S)^{-1} @ Cov(y_S, y_j)
        #
        # We maintain a running Cholesky factor of Cov(y_S, y_S) and use
        # the Schur complement for fast updates.

        # Pre-compute full sample matrix (needed for covariances).
        # Use float64 throughout for numerical stability.
        means = samples.mean(axis=0, keepdims=True)  # (1, N)
        centred = (samples - means).astype(np.float64)  # (T, N)
        var = var.astype(np.float64)

        # Conditional variance starts as marginal variance
        cond_var = var.copy()

        selected: list[int] = []
        chosen = np.zeros(n_cand, dtype=bool)
        # L will store rows of the Cholesky-like factor for the selected set
        L_rows: list[np.ndarray] = []  # each (N,) — projected covariances

        for _ in range(n_select):
            # Mask already-selected
            scores = np.where(chosen, -1.0, cond_var)
            best = int(np.argmax(scores))
            if scores[best] <= 0 and len(selected) > 0:
                # Tie-breaking: pick randomly from remaining
                remaining = np.where(~chosen)[0]
                best = int(rng.choice(remaining))
            selected.append(best)
            chosen[best] = True

            # Update conditional variances via rank-1 Schur complement
            cov_with_best = centred[:, best] @ centred / centred.shape[0]  # (N,)

            # Subtract projections from earlier L rows
            proj = cov_with_best.copy()
            for row in L_rows:
                proj -= row[best] * row

            # New L row: proj / sqrt(cond_var_best)
            cv_best = max(cond_var[best], 1e-12)
            new_row = proj / np.sqrt(cv_best)
            # Clip to prevent unbounded growth
            np.clip(new_row, -1e6, 1e6, out=new_row)
            L_rows.append(new_row)

            # Update conditional variances
            cond_var -= new_row**2
            np.maximum(cond_var, 0.0, out=cond_var)

        return np.asarray(selected, dtype=np.int64)

    # ------------------------------------------------------------------

    def _gather_samples(self, student: SequenceModel, candidates: list[str]) -> np.ndarray:
        """Collect prediction samples, shape ``(T, N)``.

        Tries ensemble members first, then falls back to repeated
        ``student.uncertainty()``-style calls via ``student.predict()``.
        """
        # Try ensemble path (duck-type check)
        if hasattr(student, "models") and hasattr(student, "_predict_member"):
            preds = []
            for m in student.models:  # type: ignore[attr-defined]
                preds.append(student._predict_member(m, candidates))  # type: ignore[attr-defined]
            return np.stack(preds, axis=0)

        # Fallback: use predict() repeatedly (assumes dropout is active)
        # For deterministic models this will just repeat the same prediction,
        # in which case the method degrades to plain uncertainty ranking.
        preds = []
        for _ in range(self.n_mc_samples):
            preds.append(student.predict(candidates))
        return np.stack(preds, axis=0)
