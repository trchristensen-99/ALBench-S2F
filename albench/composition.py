"""Composition (GC / CpG) analysis and in-support vs OOD evaluation splits.

Why this exists: students trained on different reservoirs are scored against the
oracle across many eval sets, several of which (``gc_matched``, ``random_32k``,
``dinuc_shuffle``) reach GC fractions the genomic/biological manifold never
occupies.  The oracle *extrapolates* there, so a headline metric that mixes
in-distribution and extreme-composition sequences conflates "reproduces real
sequence-function biology" with "reproduces the oracle's out-of-support
extrapolation".  These helpers let every eval metric be reported separately for
sequences whose composition is inside the genomic support and for those outside
it, so the reservoir ranking can be read without the extrapolation tail.

Self-contained: depends only on numpy + scipy, like ``albench.evaluation``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Central-98% GC range of the Gosai genomic TEST windows (chr7/chr13 holdout),
# measured Oct 2026: mean 0.462, sd 0.107, p1=0.25, p99=0.72.  Sequences outside
# [LO, HI] are compositional extrapolation relative to the biological manifold.
GENOMIC_GC_LO = 0.25
GENOMIC_GC_HI = 0.72


def gc_content(seq: str) -> float:
    """Fraction of G/C among A/C/G/T (N and other chars are ignored)."""
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    at = s.count("A") + s.count("T")
    return gc / max(gc + at, 1)


def cpg_obs_exp(seq: str) -> float:
    """CpG observed/expected ratio = (#CpG / (N-1)) / (f_C * f_G).

    ~1.0 for composition-matched random DNA; genomic DNA is CpG-depleted
    (typically 0.2-0.4 genome-wide, and ~0.02 in the local dinucleotide sense
    relevant here).  Returns 0.0 when C or G is absent.
    """
    s = seq.upper()
    n = len(s)
    if n < 2:
        return 0.0
    c = s.count("C")
    g = s.count("G")
    if c == 0 or g == 0:
        return 0.0
    cpg = s.count("CG")
    expected = (c / n) * (g / n) * (n - 1)
    return cpg / max(expected, 1e-12)


def gc_content_batch(seqs: Iterable[str]) -> np.ndarray:
    """Vectorised-enough GC over an iterable of sequences (returns float array)."""
    return np.array([gc_content(s) for s in seqs], dtype=np.float64)


def support_range_from_sequences(seqs: Iterable[str], central: float = 0.98) -> tuple[float, float]:
    """Derive an in-support GC range as the ``central`` interval of a reference pool.

    Use this to define support from the reservoir/genomic pool actually in play,
    rather than the hard-coded genomic constants.  ``central=0.98`` returns the
    1st/99th percentiles.
    """
    gc = gc_content_batch(seqs)
    tail = (1.0 - central) / 2.0 * 100.0
    return float(np.percentile(gc, tail)), float(np.percentile(gc, 100.0 - tail))


def in_support_mask(
    gc: np.ndarray, lo: float = GENOMIC_GC_LO, hi: float = GENOMIC_GC_HI
) -> np.ndarray:
    """Boolean mask: GC within the inclusive ``[lo, hi]`` support band."""
    gc = np.asarray(gc, dtype=np.float64)
    return (gc >= lo) & (gc <= hi)


def _corr(pred: np.ndarray, target: np.ndarray, fn) -> float:
    """Correlation with the same edge-case guards as ``albench.evaluation``."""
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def split_support_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    gc: np.ndarray,
    lo: float = GENOMIC_GC_LO,
    hi: float = GENOMIC_GC_HI,
) -> dict[str, dict[str, float]]:
    """Pearson/Spearman/n for an eval set split into in-support vs OOD by GC.

    Args:
        pred: model predictions, shape (n,).
        target: ground-truth (oracle or experimental) labels, shape (n,).
        gc: per-sequence GC fraction, shape (n,).  Compute with
            :func:`gc_content_batch` when you have the sequences.
        lo, hi: inclusive GC support band; defaults to the genomic range.

    Returns:
        ``{"all": m, "in_support": m, "ood": m}`` where each ``m`` is
        ``{"pearson_r", "spearman_r", "n", "frac"}``.  ``frac`` is the share of
        the eval set in that partition (so OOD ``frac`` flags how much of a given
        eval set lives in the extrapolation tail).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gc = np.asarray(gc, dtype=np.float64)
    if not (pred.shape == target.shape == gc.shape):
        raise ValueError(
            f"pred/target/gc must share shape; got {pred.shape}, {target.shape}, {gc.shape}"
        )
    total = max(pred.size, 1)
    mask = in_support_mask(gc, lo, hi)

    def _m(sel: np.ndarray) -> dict[str, float]:
        return {
            "pearson_r": _corr(pred[sel], target[sel], pearsonr),
            "spearman_r": _corr(pred[sel], target[sel], spearmanr),
            "n": int(sel.sum()),
            "frac": float(sel.sum()) / total,
        }

    return {
        "all": _m(np.ones_like(mask)),
        "in_support": _m(mask),
        "ood": _m(~mask),
    }
