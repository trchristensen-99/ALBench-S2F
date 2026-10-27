"""Canonical positive-ElasticNetCV ensemble stack.

ONE source of truth for how member predictions are combined into an ensemble.
Both the SELECTION engines (greedy_deploy_select, best_per_strategy_combos,
strategy_combination_ablation) and the Phase-4 DEPLOY ensembler call this, so the
weighting rule can never drift between "how we chose the configs" and "how we run
them in production".

Invariant: weights are ALWAYS re-fit by ElasticNetCV(positive=True) on the cell's
OWN held-out val predictions — never frozen and carried across cells. The N* member
ARCHITECTURES are shared across every reservoir x acquisition x D combo, but each
combo refits its own blend, because the same configs can be worth different amounts
on different reservoir/acquisition landscapes. A combo whose val set wants only a few
members gets the rest zeroed (positive=True kills collinear/duplicate members).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNetCV

# l1_ratio grid for the richer (info-returning) path; matches the value previously
# hard-coded in strategy_combination_ablation.py.
DEFAULT_L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def fit_elasticnet_stack(
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    *,
    l1_ratio=None,
    n_alphas: int = 50,
    return_info: bool = False,
    return_estimator: bool = False,
):
    """Positive ElasticNetCV stack fit on this cell's own val predictions.

    val_X / test_X are (n_models, n_points); transposed to (n_points, n_models)
    for sklearn. Returns (val_pred, test_pred) by default. With return_info=True
    returns (test_pred, val_pred, info) — note the swapped order — where info holds
    the fitted blend (n_kept, alpha, l1_ratio, coef, intercept) for provenance.
    With return_estimator=True the fitted ElasticNetCV is appended to the returned
    tuple so the SAME blend can be applied to extra test sets via enet.predict(X.T).
    """
    kw = dict(positive=True, cv=5, max_iter=5000, n_jobs=1)
    if l1_ratio is not None:
        kw["l1_ratio"] = l1_ratio
    else:
        kw["n_alphas"] = n_alphas
    enet = ElasticNetCV(**kw)
    enet.fit(val_X.T, val_y)
    val_pred = enet.predict(val_X.T)
    test_pred = enet.predict(test_X.T)
    if return_info:
        info = {
            "n_models": int(val_X.shape[0]),
            "n_kept": int(np.sum(enet.coef_ > 0)),
            "alpha": float(enet.alpha_),
            "l1_ratio": float(enet.l1_ratio_),
            "coef": enet.coef_.tolist(),
            "intercept": float(enet.intercept_),
        }
        out = (test_pred, val_pred, info)
    else:
        out = (val_pred, test_pred)
    if return_estimator:
        return (*out, enet)
    return out


@dataclass
class ZStack:
    """A stack fitted on z-scored member predictions: ``prediction = sum_m coef_m * z_m + intercept`` with ``coef >= 0``."""

    mu: np.ndarray
    sd: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float = 0.0
    l1_ratio: float = 0.0

    def predict(self, member_preds: np.ndarray) -> np.ndarray:
        """``member_preds`` is (n_members, n_points)."""
        z = (np.asarray(member_preds, float) - self.mu[:, None]) / self.sd[:, None]
        return z.T @ self.coef + self.intercept

    def to_json(self) -> dict:
        return {
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
            "alpha": float(self.alpha),
            "l1_ratio": float(self.l1_ratio),
        }

    @classmethod
    def from_json(cls, d: dict) -> ZStack:
        return cls(
            np.array(d["mu"]),
            np.array(d["sd"]),
            np.array(d["coef"]),
            float(d["intercept"]),
            float(d.get("alpha", 0.0)),
            float(d.get("l1_ratio", 0.0)),
        )


def fit_zscored_stack(val_X: np.ndarray, val_y: np.ndarray, *, max_points: int = 8000) -> ZStack:
    """Positive ElasticNetCV on z-scored member predictions: the stack the study's numbers were produced with.

    Differs from :func:`fit_elasticnet_stack` only in scaling the columns first (so one member with an odd output scale cannot
    dominate or be zeroed) and in fitting on a fixed random subsample of at most ``max_points`` validation points, which keeps the fit
    to seconds at 29k points with no measurable change in the ensemble. ``val_X`` is (n_models, n_points).
    """
    n = val_X.shape[1]
    sub = np.random.default_rng(0).choice(n, min(n, max_points), replace=False)
    a = np.asarray(val_X, float)[:, sub].T
    mu, sd = a.mean(0), a.std(0) + 1e-12
    en = ElasticNetCV(
        cv=5, n_alphas=40, max_iter=5000, l1_ratio=[0.1, 0.5, 0.9], positive=True, n_jobs=1
    ).fit((a - mu) / sd, np.asarray(val_y, float)[sub])
    return ZStack(
        mu, sd, en.coef_.copy(), float(en.intercept_), float(en.alpha_), float(en.l1_ratio_)
    )
