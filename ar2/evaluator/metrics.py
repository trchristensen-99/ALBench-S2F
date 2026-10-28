"""Metrics (spec 2.5, 9.2, 9.3). Everything is computed on the ORIGINAL label scale; errors inside activity or GC bins are reported as MSE, because r inside a
narrow bin is range-restricted."""

from __future__ import annotations

import numpy as np


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), np.float64)
    ranks[order] = np.arange(len(x))
    # average ranks over ties
    xs = x[order]
    start = 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or xs[i] != xs[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + i - 1) / 2.0
            start = i
    return ranks


def spearman(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return pearson(_rank(a), _rank(b))


def mse(pred, y) -> float:
    d = np.asarray(pred, np.float64).ravel() - np.asarray(y, np.float64).ravel()
    return float((d * d).mean())


def mse_by_activity_quintile(pred, y) -> list[float]:
    """MSE inside the five equal-count bins of the true activity."""
    pred, y = np.asarray(pred, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    order = np.argsort(y, kind="mergesort")
    return [mse(pred[i], y[i]) for i in np.array_split(order, 5)]


def mse_by_gc_tercile(pred, y, gc) -> list[float]:
    """MSE inside the three equal-count bins of sequence GC content."""
    pred, y, gc = (np.asarray(v, np.float64).ravel() for v in (pred, y, gc))
    order = np.argsort(gc, kind="mergesort")
    return [mse(pred[i], y[i]) for i in np.array_split(order, 3)]


def pred_sd_ratio(pred, y) -> float:
    """SD of the predictions over SD of the labels; well below 1 means compressed predictions."""
    sy = float(np.std(np.asarray(y, np.float64)))
    return float(np.std(np.asarray(pred, np.float64)) / sy) if sy > 0 else float("nan")


def still_improving(val_r_curve: list[float], per_epoch: float = 0.001, window: int = 5) -> bool:
    """True if validation r was still rising by more than ``per_epoch`` per epoch at the end of the run (least-squares slope over the last ``window``
    epochs, and the last epoch is within ``window * per_epoch`` of the best)."""
    c = np.asarray(val_r_curve[-window:], np.float64)
    c = c[np.isfinite(c)]
    if len(c) < 3:
        return False
    slope = np.polyfit(np.arange(len(c)), c, 1)[0]
    return bool(slope > per_epoch and np.max(val_r_curve) - c[-1] <= window * per_epoch)
