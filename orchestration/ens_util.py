"""Scale-robust enet_all_pos. Free-architecture models output arbitrary scales; raw ElasticNet lets one large-scale column set the
alpha grid and can zero out everything. Columns are z-scored (positive scaling keeps the positivity constraint), then mapped back."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import ElasticNetCV

def enet_all_pos(V, yv, T, sub):
    """V:(nval,N) val preds; T: dict set->(nte,N) test preds. Returns (dict set->pred, n_kept). Non-finite columns are dropped."""
    ok = np.isfinite(V).all(0)
    for t in T.values(): ok &= np.isfinite(t).all(0)
    mu, sd = V[sub][:, ok].mean(0), V[sub][:, ok].std(0) + 1e-12
    Z = (V[sub][:, ok] - mu) / sd
    en = ElasticNetCV(cv=5, n_alphas=40, max_iter=5000, l1_ratio=[.1, .5, .9], positive=True).fit(Z, yv[sub])
    return {k: en.predict((t[:, ok] - mu) / sd) for k, t in T.items()}, int((en.coef_ > 0).sum()), int((~ok).sum())
