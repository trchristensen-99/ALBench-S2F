"""Uncertainty acquisition variants from the 2026-09-16 meeting.

Three things were asked for, and they are genuinely different selection rules rather
than settings of one:

  AGGREGATE     total variance across cell types. What to use when the goal is "high
                in both" -- a sequence uncertain in either condition is worth
                measuring.

  DIFFERENTIAL  the RATIO of the two conditions' variances. Ratio rather than
                difference because variance is strictly positive, so a ratio is
                scale-free and symmetric in log space: var_A/var_B = 4 and
                var_B/var_A = 4 are equally interesting, which a difference does not
                give you.

  ACTIVITY-NORMALISED   uncertainty is strongly correlated with activity in
                regression -- larger predictions carry larger absolute errors -- so
                ranking by raw uncertainty largely re-ranks by activity. This fits
                the TREND of uncertainty against activity, takes a confidence band
                around it, and scores each candidate by how far ABOVE its own band it
                sits. The selection is then the Pareto-style frontier: sequences
                unusually uncertain FOR THEIR ACTIVITY LEVEL, not merely active.

AGGREGATE and DIFFERENTIAL need per-cell-type predictions. They raise if the student
cannot supply them rather than silently collapsing to single-task uncertainty, which
would make them duplicates of the plain uncertainty arm under different names.

NOTE, raised in the meeting: total and differential uncertainty CAN coincide -- high
uncertainty in one condition and low in the other produces both a large total and a
large ratio. They separate when both conditions are uncertain together (large total,
ratio near 1). Worth reporting their overlap rather than assuming independence.
"""

from __future__ import annotations

import logging

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel

logger = logging.getLogger(__name__)


def _per_condition_uncertainty(student: SequenceModel, candidates: list[str]) -> np.ndarray:
    """(n_candidates, n_conditions) uncertainty, or raise if unavailable."""
    fn = getattr(student, "uncertainty_per_condition", None)
    if fn is None:
        raise ValueError(
            "this acquisition needs per-cell-type uncertainty, but the student "
            "exposes only a single output. Train a multitask student (--multitask) "
            "or use the single-condition uncertainty arm. Falling back silently "
            "would make this a duplicate of that arm under a different name."
        )
    u = np.asarray(fn(candidates), dtype=float)
    if u.ndim != 2 or u.shape[1] < 2:
        raise ValueError(
            f"uncertainty_per_condition returned shape {u.shape}; need "
            f"(n_candidates, >=2 conditions) for an aggregate/differential rule"
        )
    return u


class AggregateUncertaintyAcquisition(AcquisitionFunction):
    """Top-k by TOTAL variance summed over conditions."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(self, student: SequenceModel, candidates: list[str], n_select: int) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        u = _per_condition_uncertainty(student, candidates)
        total = (u**2).sum(axis=1)          # variances add; uncertainties do not
        return np.argsort(-total)[:n_select].astype(np.int64)


class DifferentialUncertaintyAcquisition(AcquisitionFunction):
    """Top-k by |log variance ratio| between two conditions."""

    def __init__(self, seed: int | None = None, cond_a: int = 0, cond_b: int = 1) -> None:
        self.seed = seed
        self.cond_a = cond_a
        self.cond_b = cond_b

    def select(self, student: SequenceModel, candidates: list[str], n_select: int) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        u = _per_condition_uncertainty(student, candidates)
        va = np.maximum(u[:, self.cond_a] ** 2, 1e-12)
        vb = np.maximum(u[:, self.cond_b] ** 2, 1e-12)
        # |log ratio|: symmetric, scale-free, and finite because variance > 0.
        score = np.abs(np.log(va / vb))
        return np.argsort(-score)[:n_select].astype(np.int64)


class ActivityNormalisedUncertaintyAcquisition(AcquisitionFunction):
    """Top-k by uncertainty EXCESS over the trend at that activity level.

    Bins candidates by predicted activity, estimates the typical uncertainty and its
    spread within each bin, and scores by the standardised residual. A candidate only
    scores highly if it is uncertain relative to other sequences of similar predicted
    activity — which is what separates "genuinely ambiguous" from "merely active".
    """

    def __init__(
        self,
        seed: int | None = None,
        n_bins: int = 20,
        min_per_bin: int = 20,
    ) -> None:
        self.seed = seed
        self.n_bins = n_bins
        self.min_per_bin = min_per_bin

    def select(self, student: SequenceModel, candidates: list[str], n_select: int) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        act = np.asarray(student.predict(candidates), dtype=float).ravel()
        unc = np.asarray(student.uncertainty(candidates), dtype=float).ravel()

        # Equal-COUNT bins, not equal-width: activity is usually skewed, and
        # equal-width bins leave the tails with too few points to estimate a spread.
        n_bins = max(2, min(self.n_bins, len(candidates) // max(1, self.min_per_bin)))
        edges = np.quantile(act, np.linspace(0, 1, n_bins + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        idx_bin = np.clip(np.digitize(act, edges[1:-1], right=False), 0, n_bins - 1)

        resid = np.zeros_like(unc)
        for b in range(n_bins):
            m = idx_bin == b
            if m.sum() < 3:
                resid[m] = 0.0          # too few to judge; do not let noise rank
                continue
            centre = np.median(unc[m])
            # MAD as the spread: robust to the few very uncertain points that are
            # exactly what we are trying to detect, which would inflate an SD.
            mad = np.median(np.abs(unc[m] - centre))
            scale = 1.4826 * mad if mad > 0 else unc[m].std()
            resid[m] = (unc[m] - centre) / scale if scale > 0 else 0.0

        logger.info(
            "activity-normalised uncertainty: %d bins, residual range %.2f..%.2f",
            n_bins, float(resid.min()), float(resid.max()),
        )
        return np.argsort(-resid)[:n_select].astype(np.int64)
