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
):
    """Positive ElasticNetCV stack fit on this cell's own val predictions.

    val_X / test_X are (n_models, n_points); transposed to (n_points, n_models)
    for sklearn. Returns (val_pred, test_pred) by default. With return_info=True
    returns (test_pred, val_pred, info) — note the swapped order — where info holds
    the fitted blend (n_kept, alpha, l1_ratio, coef) for provenance.
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
        }
        return test_pred, val_pred, info
    return val_pred, test_pred
