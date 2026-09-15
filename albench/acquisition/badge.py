"""BADGE: Batch Active learning by Diverse Gradient Embeddings, for regression.

A LITERAL PORT OF BADGE IS SILENTLY BROKEN HERE. BADGE embeds each candidate as the
last-layer loss gradient under a hypothetical label. For a Gaussian regression head
with the model's own prediction as that label,

    dL/dw = -((y - mu) / sigma^2) * z(x)   and with y = mu this is exactly 0

for every candidate, so k-means++ runs on a set of identical zero vectors and returns
an arbitrary subset. The method degenerates to random selection and nothing in the
code or the output says so. The obvious repair -- sampling y ~ N(mu, sigma^2) -- is
worse than it looks: the gradient norm becomes proportional to ||z||/sigma, which
DECREASES with uncertainty.

What we do instead: take the expected gradient outer product rather than one sampled
gradient. For a Gaussian head that is the Fisher information z z^T / sigma^2, so the
natural embedding is z(x)/sigma(x). We then weight by EPISTEMIC over ALEATORIC
uncertainty:

    e(x) = ( sigma_epi(x) / sigma_ale(x) ) * z(x)

The ratio matters for MPRA specifically. Activity is estimated from read counts, so
weakly-expressed sequences carry large aleatoric variance; an acquisition function
driven by total uncertainty will spend its whole batch on sequences whose labels we
cannot measure precisely however many we order. The ratio declines to pay for noise
that will not shrink.

INTERPRETING A RESULT FROM THIS REQUIRES THE CONTROLS in this module:
``epistemic_only`` (uncertainty without diversity) and ``kmeanspp_only`` (diversity
without uncertainty), plus random. Without them, a gain cannot be attributed to the
fusion rather than to either half alone.
"""

from __future__ import annotations

import logging

import numpy as np

from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel

logger = logging.getLogger(__name__)


def kmeanspp(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: pick k points, each far from those already chosen."""
    n = len(points)
    if k > n:
        raise ValueError(f"cannot select {k} from {n} candidates")
    first = int(rng.integers(n))
    selected = [first]
    min_dist = np.sum((points - points[first]) ** 2, axis=1)
    while len(selected) < k:
        probs = min_dist.copy()
        probs[selected] = 0.0
        total = probs.sum()
        if total <= 0:
            # Every remaining point coincides with a centre. Falling back to uniform
            # keeps the batch full, but it means the embedding has collapsed -- which
            # is exactly the degeneracy this module exists to avoid, so say so.
            logger.warning(
                "k-means++ distances all zero after %d/%d picks: the embedding has "
                "collapsed and the rest of the batch is effectively random",
                len(selected),
                k,
            )
            mask = np.ones(n, dtype=bool)
            mask[selected] = False
            remaining = np.where(mask)[0]
            take = min(k - len(selected), remaining.size)
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
            break
        idx = int(rng.choice(n, p=probs / total))
        selected.append(idx)
        np.minimum(min_dist, np.sum((points - points[idx]) ** 2, axis=1), out=min_dist)
    return np.asarray(selected[:k], dtype=np.int64)


class BADGEAcquisition(AcquisitionFunction):
    """Uncertainty-weighted embeddings seeded by k-means++.

    Args:
        seed: Random seed.
        mode: ``"ratio"`` uses sigma_epi/sigma_ale (the default and the intended
            method); ``"epistemic"`` uses sigma_epi alone, which is the
            uncertainty-without-diversity behaviour when combined with a top-k
            selector; ``"total"`` reproduces the naive total-uncertainty scaling for
            comparison.
        eps: Floor on the aleatoric term, so a near-zero denominator cannot make one
            candidate dominate the embedding.
    """

    def __init__(self, seed: int | None = None, mode: str = "ratio", eps: float = 1e-3) -> None:
        if mode not in ("ratio", "epistemic", "total"):
            raise ValueError(f"mode must be ratio/epistemic/total, got {mode!r}")
        self.seed = seed
        self.mode = mode
        self.eps = eps

    def _weights(self, student: SequenceModel, candidates: list[str]) -> np.ndarray:
        if self.mode == "total":
            return np.asarray(student.uncertainty(candidates), dtype=float)
        epi = np.asarray(student.epistemic_uncertainty(candidates), dtype=float)
        if self.mode == "epistemic":
            return epi
        if not getattr(student, "separates_uncertainty", False):
            logger.warning(
                "%s cannot separate epistemic from aleatoric uncertainty, so "
                "mode='ratio' reduces to total-uncertainty weighting. Use an ensemble "
                "or MC-dropout student, or pass mode='total' deliberately.",
                type(student).__name__,
            )
            return epi
        ale = np.asarray(student.aleatoric_uncertainty(candidates), dtype=float)
        return epi / np.maximum(ale, self.eps)

    def select(
        self,
        student: SequenceModel,
        candidates: list[str],
        n_select: int,
    ) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        z = np.asarray(student.embed(candidates), dtype=float)
        w = self._weights(student, candidates)
        if z.ndim != 2 or len(z) != len(candidates):
            raise ValueError(f"embed() must return (N, D); got {z.shape}")
        emb = z * w[:, None]
        spread = float(np.linalg.norm(emb, axis=1).std())
        if spread < 1e-12:
            raise ValueError(
                "BADGE embeddings have no spread -- every candidate maps to the same "
                "point, so selection would be random. This is the degeneracy described "
                "in the module docstring; check that uncertainty() is not constant."
            )
        return kmeanspp(emb, n_select, np.random.default_rng(self.seed))


class EpistemicOnlyAcquisition(AcquisitionFunction):
    """CONTROL: top-k by epistemic uncertainty. Uncertainty without diversity."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(self, student: SequenceModel, candidates: list[str], n_select: int) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        u = np.asarray(student.epistemic_uncertainty(candidates), dtype=float)
        return np.argsort(-u)[:n_select].astype(np.int64)


class KMeansPPOnlyAcquisition(AcquisitionFunction):
    """CONTROL: k-means++ on raw embeddings. Diversity without uncertainty."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(self, student: SequenceModel, candidates: list[str], n_select: int) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        z = np.asarray(student.embed(candidates), dtype=float)
        return kmeanspp(z, n_select, np.random.default_rng(self.seed))
