"""Acquisition that spends its budget evenly across the predicted-activity range.

Every rule in the study so far ranks candidates and takes the top of that ranking, which
means the activity profile of the selected corpus is whatever the ranking happens to
produce. Measured on the 4-way pool: uncertainty reaches all ten deciles of the pool's
activity range, but expected-activity reaches nine and its 10th percentile sits at 1.68
against the pool's -0.29 -- it abandons the bottom of the range entirely. A corpus that
covers the range and a corpus that creams the top are different training sets, and a
ranking rule cannot be asked for one rather than the other.

This makes coverage the constraint instead of a side effect: split the candidates into
equal-count strata by PREDICTED activity and take the same number from each. What happens
inside a stratum is the second half of the rule, and both halves are worth having
separately -- `within="random"` is a pure coverage prior with no model uncertainty in it
at all, which is the version anyone can reproduce without running active learning, and
`within="epistemic"` adds uncertainty back within each stratum.

PREDICTED activity, never the oracle label. The candidate pool is already labelled and the
selection rule may not look at those labels; using them would make every arm a
reimplementation of "sort by the answer".
"""

from __future__ import annotations

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel


class ActivityCoverageAcquisition(AcquisitionFunction):
    """Equal allocation across predicted-activity strata.

    Parameters
    ----------
    n_strata:
        Equal-COUNT strata, not equal-width. Predicted activity is skewed, and equal-width
        strata would put most candidates in two or three bins and leave the tails with
        fewer rows than the per-stratum quota.
    within:
        ``"random"`` for a pure coverage prior, ``"epistemic"`` to take the most uncertain
        candidates inside each stratum.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_strata: int = 10,
        within: str = "epistemic",
    ) -> None:
        if within not in ("random", "epistemic"):
            raise ValueError(f"within must be 'random' or 'epistemic', got {within!r}")
        self.seed = seed
        self.n_strata = n_strata
        self.within = within
        self.realised_per_stratum_: list[int] | None = None
        self.n_short_strata_: int | None = None

    def select(self, student: SequenceModel, candidates: list[str], n_select: int) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        rng = np.random.default_rng(self.seed)
        act = np.asarray(student.predict(candidates), dtype=float).ravel()
        k = max(2, min(self.n_strata, len(candidates) // max(1, n_select // self.n_strata or 1)))
        k = max(2, min(k, self.n_strata))

        edges = np.quantile(act, np.linspace(0, 1, k + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        stratum = np.clip(np.digitize(act, edges[1:-1], right=False), 0, k - 1)

        score = None
        if self.within == "epistemic":
            score = np.asarray(student.uncertainty(candidates), dtype=float).ravel()

        # Fill in two passes. A stratum with fewer rows than its quota gives its shortfall
        # back, and the second pass redistributes it over the strata that still have
        # candidates -- otherwise the arm would quietly select fewer than n_select and the
        # size comparison against every other rule would be off.
        quota = np.full(k, n_select // k, dtype=int)
        quota[: n_select - quota.sum()] += 1
        picked: list[np.ndarray] = []
        taken = np.zeros(k, dtype=int)
        remaining: list[np.ndarray] = []
        for s in range(k):
            idx = np.flatnonzero(stratum == s)
            take = int(min(quota[s], len(idx)))
            chosen = self._pick(idx, take, score, rng)
            picked.append(chosen)
            taken[s] = take
            remaining.append(np.setdiff1d(idx, chosen, assume_unique=False))

        short = int(n_select - int(taken.sum()))
        self.n_short_strata_ = int((taken < quota).sum())
        if short > 0:
            pool = (
                np.concatenate([r for r in remaining if len(r)])
                if any(len(r) for r in remaining)
                else np.array([], dtype=int)
            )
            if len(pool) < short:
                raise ValueError(
                    f"only {len(pool) + int(taken.sum()):,} distinct candidates available for "
                    f"{n_select:,} requested"
                )
            picked.append(self._pick(pool, short, score, rng))

        out = np.concatenate([p for p in picked if len(p)])
        self.realised_per_stratum_ = [int(x) for x in taken]
        return out.astype(np.int64)

    def _pick(self, idx: np.ndarray, take: int, score, rng) -> np.ndarray:
        if take <= 0 or len(idx) == 0:
            return np.array([], dtype=np.int64)
        if score is None:
            return rng.choice(idx, size=take, replace=False)
        order = np.argsort(-score[idx], kind="stable")
        return idx[order[:take]]
