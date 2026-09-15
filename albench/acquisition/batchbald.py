"""BatchBALD for regression: closed form, plus the rank guard it needs.

In regression BatchBALD is EASIER than in classification, not harder. For a Gaussian
posterior with homoscedastic noise the batch mutual information has a closed form:

    I(Y_B ; theta) = 0.5 * log det( I + Sigma_epi / sigma^2 )

which is submodular under greedy selection, costs O(N * B^2), and needs no
Monte-Carlo entropy estimation. This is D-optimal Bayesian experimental design.

THE BINDING CONSTRAINT IS POSTERIOR RANK, AND IT IS SEVERE. The epistemic covariance
is estimated from M ensemble members or MC passes, so its rank is at most M-1. Once
B selections have exhausted those directions the objective is flat and every
remaining pick is arbitrary -- with M=10 and a 384-sequence batch, roughly 375 of the
384 are chosen at random, and nothing in the output would reveal it.

Growing the ensemble is the wrong fix: you would need M > B, i.e. hundreds of models.
The right fix is a higher-rank posterior (last-layer linearised Laplace gives rank up
to the feature dimension). Here we cannot impose that on an arbitrary student, so we
DETECT the saturation and warn, which at least makes the degeneracy visible.

Reference: Kirsch, van Amersfoort & Gal (2019).
"""

from __future__ import annotations

import logging

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel

logger = logging.getLogger(__name__)


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

        # The posterior is estimated from T samples, so Sigma_epi has rank <= T-1 and
        # the objective can only distinguish that many directions. Past it, every
        # conditional variance is ~0 and picks are arbitrary. See the module docstring.
        max_informative = max(samples.shape[0] - 1, 1)
        if n_select > max_informative:
            logger.warning(
                "BatchBALD asked for %d selections from a rank-%d posterior (%d "
                "samples). Only the first ~%d are informative; the remaining %d will "
                "be effectively random. Increase samples above the batch size, or use "
                "a higher-rank posterior (last-layer Laplace).",
                n_select,
                max_informative,
                samples.shape[0],
                max_informative,
                n_select - max_informative,
            )
        self.n_informative_ = 0

        for _ in range(n_select):
            # Mask already-selected
            scores = np.where(chosen, -1.0, cond_var)
            best = int(np.argmax(scores))
            # Record how many picks were actually driven by the objective, so a caller
            # can tell an informative batch from a padded one after the fact.
            if scores[best] > 1e-10:
                self.n_informative_ = len(selected) + 1
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

        if self.n_informative_ < n_select:
            logger.warning(
                "BatchBALD: only %d/%d selections were driven by the objective; the "
                "rest were filled at random once the posterior saturated.",
                self.n_informative_,
                n_select,
            )
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
