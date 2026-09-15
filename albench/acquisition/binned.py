"""Binned BADGE and BatchBALD: the textbook forms, made applicable to MPRA.

WHY BINNING IS NOT A HACK HERE. Both methods were derived for classification, and
both have a specific degeneracy when ported naively to a Gaussian regression head:

  BADGE      the last-layer gradient under the model's own prediction as pseudo-label
             is identically zero, so every candidate embeds to the same point
  BatchBALD  the epistemic covariance from M ensemble members has rank <= M-1, so a
             batch larger than that is padded with arbitrary picks

Discretising activity into K bins removes both problems rather than working around
them. With a categorical predictive distribution:

  BADGE      grad = (p - onehot(argmax p)) (x) z, which is non-zero whenever the
             model is not perfectly confident -- exactly the textbook embedding
  BatchBALD  the mutual information is the standard discrete expression, and its
             rank is governed by K and the batch, not by the ensemble size

This is also why the yeast task needs no adaptation: DREAM activities are already
18-bin soft labels, so the categorical form applies directly. For the human MPRA the
choice between binning and a continuous adaptation is an empirical one, which is why
both live in the codebase and can be compared as arms rather than argued about.

WHAT BINNING COSTS. Resolution within a bin is discarded, so the acquisition cannot
distinguish two candidates whose activity differs by less than a bin width. With
quantile edges the bins are equally populated, so that loss is spread evenly rather
than concentrated in the dense middle of the distribution. ``n_bins`` is the knob
trading resolution against how well the categorical assumption holds.
"""

from __future__ import annotations

import logging

import numpy as np

from albench.acquisition.badge import kmeanspp
from albench.acquisition.base import AcquisitionFunction
from albench.model import SequenceModel

logger = logging.getLogger(__name__)


def bin_edges(values: np.ndarray, n_bins: int, mode: str = "quantile") -> np.ndarray:
    """Interior bin edges for discretising a continuous activity.

    ``quantile`` gives equally-populated bins, which keeps every bin statistically
    usable; ``uniform`` gives equal-width bins, which leaves the tails nearly empty on
    a distribution as skewed as MPRA activity.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("cannot compute bin edges from an empty array")
    if mode == "quantile":
        e = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
    elif mode == "uniform":
        e = np.linspace(v.min(), v.max(), n_bins + 1)[1:-1]
    else:
        raise ValueError(f"mode must be quantile/uniform, got {mode!r}")
    # Duplicate edges arise when many values tie; collapsing them silently would give
    # fewer bins than requested, so make it visible.
    uniq = np.unique(e)
    if uniq.size < e.size:
        logger.warning(
            "bin_edges: %d of %d interior edges were duplicates (ties in the "
            "distribution); effective bins = %d, not %d",
            e.size - uniq.size,
            e.size,
            uniq.size + 1,
            n_bins,
        )
    return uniq


def samples_to_probs(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(T, N) posterior samples -> (N, K) categorical predictive distribution.

    Each of the T samples votes for one bin; the distribution over bins IS the
    posterior predictive, and its spread across samples is the epistemic uncertainty
    the acquisition functions consume.
    """
    t, n = samples.shape
    k = len(edges) + 1
    idx = np.digitize(samples, edges)  # (T, N) in [0, K-1]
    probs = np.zeros((n, k), dtype=float)
    for j in range(k):
        probs[:, j] = (idx == j).sum(axis=0)
    return probs / t


def gather_samples(student: SequenceModel, candidates: list[str], n_mc: int) -> np.ndarray:
    """Posterior samples, shape (T, N). Ensemble members first, else MC dropout."""
    models = getattr(student, "models", None)
    if models and hasattr(student, "_predict_member"):
        return np.stack([student._predict_member(m, candidates) for m in models], axis=0)
    preds = [np.asarray(student.predict(candidates), dtype=float) for _ in range(n_mc)]
    out = np.stack(preds, axis=0)
    if np.allclose(out, out[0]):
        raise ValueError(
            "Posterior samples are identical across passes, so epistemic uncertainty "
            "is zero and both binned methods reduce to random selection. The student "
            "needs active dropout or an ensemble."
        )
    return out


class BinnedBADGEAcquisition(AcquisitionFunction):
    """Textbook BADGE on a binned (categorical) head.

    The embedding is the true last-layer cross-entropy gradient,
    ``(p - onehot(argmax p)) (x) z``, flattened to D*K. Unlike the Gaussian case this
    is non-zero for any candidate the model is not perfectly certain about, so the
    k-means++ step has real geometry to work with.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_bins: int = 10,
        bin_mode: str = "quantile",
        n_mc_samples: int = 30,
    ) -> None:
        self.seed = seed
        self.n_bins = n_bins
        self.bin_mode = bin_mode
        self.n_mc_samples = n_mc_samples

    def select(self, student: SequenceModel, candidates: list[str], n_select: int) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        samples = gather_samples(student, candidates, self.n_mc_samples)
        edges = bin_edges(samples.ravel(), self.n_bins, self.bin_mode)
        p = samples_to_probs(samples, edges)  # (N, K)
        z = np.asarray(student.embed(candidates), dtype=float)  # (N, D)

        onehot = np.zeros_like(p)
        onehot[np.arange(len(p)), p.argmax(axis=1)] = 1.0
        resid = p - onehot  # (N, K)
        emb = (z[:, :, None] * resid[:, None, :]).reshape(len(z), -1)  # (N, D*K)

        norms = np.linalg.norm(emb, axis=1)
        if norms.std() < 1e-12:
            raise ValueError(
                "Binned BADGE embeddings have no spread: the model is equally "
                "confident everywhere, so selection would be random."
            )
        logger.info(
            "binned BADGE: %d bins, embedding dim %d, gradient norm mean %.4g "
            "(a Gaussian head would give exactly 0 here)",
            p.shape[1],
            emb.shape[1],
            float(norms.mean()),
        )
        return kmeanspp(emb, n_select, np.random.default_rng(self.seed))


class BinnedBatchBALDAcquisition(AcquisitionFunction):
    """Textbook BatchBALD on a binned head, greedy over the discrete joint entropy.

    Uses the standard estimator: for a candidate batch B,
    ``I(Y_B; theta) = H[E_theta p(Y_B)] - E_theta H[p(Y_B)]``. The joint is estimated
    by sampling bin assignments from each posterior member, which keeps the cost
    linear in the number of members rather than exponential in the batch size.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_bins: int = 10,
        bin_mode: str = "quantile",
        n_mc_samples: int = 30,
        n_joint_samples: int = 2000,
    ) -> None:
        self.seed = seed
        self.n_bins = n_bins
        self.bin_mode = bin_mode
        self.n_mc_samples = n_mc_samples
        self.n_joint_samples = n_joint_samples

    @staticmethod
    def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
        q = np.clip(p, 1e-12, 1.0)
        return -(q * np.log(q)).sum(axis=axis)

    def select(self, student: SequenceModel, candidates: list[str], n_select: int) -> np.ndarray:
        if n_select > len(candidates):
            raise ValueError("n_select cannot exceed candidate count")
        rng = np.random.default_rng(self.seed)
        samples = gather_samples(student, candidates, self.n_mc_samples)
        edges = bin_edges(samples.ravel(), self.n_bins, self.bin_mode)
        k = len(edges) + 1
        t, n = samples.shape
        idx = np.digitize(samples, edges)  # (T, N)
        per_member = np.zeros((t, n, k))
        per_member[np.arange(t)[:, None], np.arange(n)[None, :], idx] = 1.0

        # Marginal BALD score, the first selection and the base for the greedy step.
        mean_p = per_member.mean(axis=0)  # (N, K)
        bald = self._entropy(mean_p) - self._entropy(per_member).mean(axis=0)

        selected: list[int] = [int(np.argmax(bald))]
        chosen = np.zeros(n, dtype=bool)
        chosen[selected[0]] = True

        # Joint term: track, per posterior member, the configuration of the batch so
        # far. A candidate is valuable if it splits members that currently agree.
        while len(selected) < n_select:
            cur = idx[:, selected]  # (T, |S|)
            # hash each member's configuration so far
            _, conf = np.unique(cur, axis=0, return_inverse=True)
            n_conf = conf.max() + 1
            gains = np.full(n, -np.inf)
            for j in range(n):
                if chosen[j]:
                    continue
                joint = conf * k + idx[:, j]
                counts = np.bincount(joint, minlength=n_conf * k).astype(float)
                pj = counts / t
                # conditional entropy of the new variable given the configuration
                gains[j] = self._entropy(pj) - self._entropy(per_member[:, j, :]).mean()
            best = int(np.argmax(gains))
            if not np.isfinite(gains[best]) or gains[best] <= 1e-12:
                remaining = np.where(~chosen)[0]
                logger.warning(
                    "binned BatchBALD: joint information exhausted after %d/%d picks "
                    "(%d posterior members can only distinguish so many "
                    "configurations); remaining picks are random",
                    len(selected),
                    n_select,
                    t,
                )
                take = min(n_select - len(selected), remaining.size)
                selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
                break
            selected.append(best)
            chosen[best] = True
        return np.asarray(selected[:n_select], dtype=np.int64)
