"""Reservoir x eval-set matrix for the fixed-N deploy ensemble.

Refits the ensemble weights on each cell's own validation set (ElasticNetCV,
positive) over the N models greedy selection chose, then applies those fixed
weights to every stored eval set. Weights are fit ONCE on val -- never per eval
set -- so the matrix answers "how does the deploy ensemble transfer?" rather
than "how well could we fit each eval set if we cheated".
"""

from __future__ import annotations

import glob
import json
import os
import re
import warnings

import numpy as np
from sklearn.linear_model import ElasticNetCV

warnings.filterwarnings("ignore")

N_ENS = int(os.environ.get("N_ENS", "5"))
FULL = int(os.environ.get("FULL_POOL", "60"))

SETS = ["genomic", "motif_planted", "gc_matched", "zoonomia",
        "random_32k", "dinuc_shuffle", "sub_low", "sub_med", "sub_high",
        "ins_low", "ins_med", "ins_high", "del_low", "del_med", "del_high",
        "translocation", "inversion", "ood"]


def mse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean((a[m] - b[m]) ** 2)) if m.sum() >= 3 else np.nan


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    sa, sb = a.std(), b.std()
    return float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else np.nan


def cell_row(cell_dir):
    gd = os.path.join(cell_dir, "ensemble", "greedy_deploy.json")
    lab = os.path.join(cell_dir, "labels.npz")
    if not (os.path.exists(gd) and os.path.exists(lab)):
        return None
    pool = len(glob.glob(os.path.join(cell_dir, "*_meta.json")))
    d = json.load(open(gd))
    ids = [c["added_id"] for c in (d.get("curve") or [])[:N_ENS]]
    if len(ids) < N_ENS:
        return None
    Y = np.load(lab, allow_pickle=True)
    yval = np.asarray(Y["val_labels"], float)

    vp, tp = [], {}
    for i in ids:
        f = os.path.join(cell_dir, f"{i}.npz")
        if not os.path.exists(f):
            return None
        z = np.load(f, allow_pickle=True)
        vp.append(np.asarray(z["val_pred"], float))
        for s in SETS + ["snv_ref", "snv_alt"]:
            k = f"test_pred_{s}"
            if k in z.files:
                tp.setdefault(s, []).append(np.asarray(z[k], float))
    V = np.vstack(vp).T
    m = np.isfinite(V).all(1) & np.isfinite(yval)
    en = ElasticNetCV(positive=True, cv=5, n_alphas=40, max_iter=5000).fit(V[m], yval[m])
    w, b = en.coef_, en.intercept_

    out = {"pool": pool}
    for s in SETS:
        # A cell can span the moment an eval set was ADDED to the battery, so some
        # of its models carry that prediction and some do not. Scoring a 5-weight
        # blend against 4 prediction vectors is a dimension error, and silently
        # dropping the missing member would score a different ensemble than the one
        # selected. Skip the set for this cell instead.
        if s not in tp or f"oracle_{s}" not in Y.files or len(tp[s]) != len(ids):
            continue
        P = np.vstack(tp[s]).T @ w + b
        y = np.asarray(Y[f"oracle_{s}"], float)
        out[s] = pearson(P, y)
        out[f"{s}_mse"] = mse(P, y)
    # SNV is a PAIRED effect: score the ref->alt delta, not the two arms separately
    if (len(tp.get("snv_ref", [])) == len(ids)
            and len(tp.get("snv_alt", [])) == len(ids)):
        pr = np.vstack(tp["snv_ref"]).T @ w + b
        pa = np.vstack(tp["snv_alt"]).T @ w + b
        dt = (np.asarray(Y["oracle_snv_alt"], float)
              - np.asarray(Y["oracle_snv_ref"], float))
        out["snv_delta"] = pearson(pa - pr, dt)
        out["snv_delta_mse"] = mse(pa - pr, dt)
    return out


def retrain_row(cell_dir):
    """Row from a RETRAINED cell. These carry the full 20-set battery including the
    reservoir-matched sets, which the original cells predate and cannot have."""
    lab = os.path.join(cell_dir, "labels.npz")
    members = sorted(glob.glob(os.path.join(cell_dir, "r*_meta.json")))
    if not os.path.exists(lab) or len(members) < N_ENS:
        return None
    ids = [os.path.basename(m).replace("_meta.json", "") for m in members][:N_ENS]
    Y = np.load(lab, allow_pickle=True)
    yval = np.asarray(Y["val_labels"], float)
    vp, tp = [], {}
    for i in ids:
        f = os.path.join(cell_dir, f"{i}.npz")
        if not os.path.exists(f):
            return None
        z = np.load(f, allow_pickle=True)
        if "val_pred" not in z.files:
            return None
        vp.append(np.asarray(z["val_pred"], float))
        for s_ in SETS + ["snv_ref", "snv_alt"]:
            k = f"test_pred_{s_}"
            if k in z.files:
                tp.setdefault(s_, []).append(np.asarray(z[k], float))
    V = np.vstack(vp).T
    m = np.isfinite(V).all(1) & np.isfinite(yval)
    en = ElasticNetCV(positive=True, cv=5, n_alphas=40, max_iter=5000).fit(V[m], yval[m])
    w, b = en.coef_, en.intercept_
    out = {"pool": len(members)}
    for s_ in SETS:
        if s_ not in tp or f"oracle_{s_}" not in Y.files or len(tp[s_]) != N_ENS:
            continue
        P_ = np.vstack(tp[s_]).T @ w + b
        y = np.asarray(Y[f"oracle_{s_}"], float)
        out[s_] = pearson(P_, y)
        out[f"{s_}_mse"] = mse(P_, y)
    if "snv_ref" in tp and "snv_alt" in tp and len(tp["snv_ref"]) == N_ENS:
        pr = np.vstack(tp["snv_ref"]).T @ w + b
        pa = np.vstack(tp["snv_alt"]).T @ w + b
        dt = (np.asarray(Y["oracle_snv_alt"], float) - np.asarray(Y["oracle_snv_ref"], float))
        out["snv_delta"] = pearson(pa - pr, dt)
        out["snv_delta_mse"] = mse(pa - pr, dt)
    return out


rows = {}
for tier, D in (("d100k", 100000), ("d30k", 30000), ("d10k", 10000)):
    for cd in sorted(glob.glob(f"outputs/stage4_v2_{tier}/*_d{D}")):
        arm = re.sub(r"_d\d+$", "", os.path.basename(cd))
        r = cell_row(cd)
        if r:
            rows[(D, arm)] = r
    for cd in sorted(glob.glob(f"outputs/retrain_v1/*_d{D}")):
        arm = re.sub(rf"_d{D}$", "", os.path.basename(cd))
        r = retrain_row(cd)
        if r:
            r["retrained"] = 1
            rows[(D, arm)] = r          # overwrite: superset of eval sets

base_cols = SETS + ["snv_delta"]
cols = base_cols + [f"{c}_mse" for c in base_cols]
os.makedirs("outputs/figures", exist_ok=True)
with open("outputs/figures/eval_matrix.csv", "w") as fh:
    fh.write("D,arm,pool,complete,retrained," + ",".join(cols) + "\n")
    for (D, arm), r in sorted(rows.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        fh.write(f"{D},{arm},{r['pool']},{int(bool(r['pool'] >= FULL or r.get('retrained')))},{r.get('retrained',0)},"
                 + ",".join("" if np.isnan(r.get(c, np.nan)) else f"{r.get(c):.4f}" for c in cols) + "\n")

for D in sorted({k[0] for k in rows}, reverse=True):
    sub = {a: r for (dd, a), r in rows.items() if dd == D and r["pool"] >= FULL}
    if not sub:
        continue
    print(f"\n=== D={D:,}  ensemble n={N_ENS}  (complete cells only) ===")
    hdr = ["genomic", "random_32k", "dinuc_shuffle", "sub_high", "ins_high",
           "del_high", "translocation", "inversion", "ood", "snv_delta"]
    print("  " + "arm".ljust(22) + "".join(h[:9].rjust(11) for h in hdr))
    for a, r in sorted(sub.items(), key=lambda kv: -(kv[1].get("genomic") or -9)):
        print("  " + a.ljust(22) + "".join(
            ("  n/a".rjust(11) if np.isnan(r.get(h, np.nan)) else f"{r.get(h):.4f}".rjust(11)) for h in hdr))
print(f"\nwrote outputs/figures/eval_matrix.csv  ({len(rows)} cells)")
