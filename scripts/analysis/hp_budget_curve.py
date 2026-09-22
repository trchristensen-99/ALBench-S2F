#!/usr/bin/env python
"""How many HP-search models does a cell actually need? Measure it, do not guess.

The deploy procedure is: search R rounds x 5 strategies, greedy-forward select on
validation, stop at N=5, ElasticNetCV the weights. The open parameter is R, and it
multiplies the whole study -- R=10 vs R=30 is 1,412 vs 4,235 GPU-h on the human grid.

This replays that exact procedure at increasing search budgets against bake-off cells
that are already trained, so the plateau is measured under the selection rule we will
actually use. Two details matter:

  SUBSAMPLE, DO NOT TAKE A PREFIX. Taking the first k models would inherit the search
  strategy's own ordering, which improves over rounds -- that measures "are later
  rounds better", not "is a budget of k enough". We draw k at random, repeatedly.

  SELECT ON VAL, SCORE ON TEST. Selecting and scoring on the same split would make
  every budget look good and the curve would plateau immediately by construction.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

N_ENSEMBLE = 5


def load_cell(cell_dir: Path):
    """Return (val_pred matrix, test_pred matrix, val_labels, test_labels)."""
    vals, tests = [], []
    for meta in sorted(glob.glob(str(cell_dir / "*" / "*_meta.json"))):
        npz = meta.replace("_meta.json", ".npz")
        if not os.path.exists(npz):
            continue
        try:
            with np.load(npz) as z:
                if "val_pred" not in z or "test_pred_genomic" not in z:
                    continue
                vals.append(z["val_pred"])
                tests.append(z["test_pred_genomic"])
        except Exception:
            continue
    if not vals:
        return None
    lab = cell_dir / "labels.npz"
    if not lab.exists():
        lab = next(iter(cell_dir.glob("*/labels.npz")), None)
    if lab is None:
        return None
    with np.load(lab, allow_pickle=True) as z:
        keys = list(z.keys())
        vy = z["val_labels"] if "val_labels" in keys else z[keys[0]]
        ty = z["test_labels_genomic"] if "test_labels_genomic" in keys else None
        if ty is None:
            ty = z["test_labels"] if "test_labels" in keys else None
    if ty is None:
        return None
    V = np.vstack(vals)
    T = np.vstack(tests)
    ok = [i for i in range(V.shape[0]) if V.shape[1] == len(vy) and T.shape[1] == len(ty)]
    if not ok:
        return None
    return V, T, np.asarray(vy), np.asarray(ty)


def greedy_then_stack(V, T, vy, ty, idx, n=N_ENSEMBLE):
    """Greedy-forward on val MSE to n members, then ElasticNetCV weights. Test Pearson."""
    chosen: list[int] = []
    for _ in range(min(n, len(idx))):
        best, best_mse = None, np.inf
        for c in idx:
            if c in chosen:
                continue
            m = V[chosen + [c]].mean(axis=0)
            mse = float(np.mean((m - vy) ** 2))
            if mse < best_mse:
                best, best_mse = c, mse
        if best is None:
            break
        chosen.append(best)
    Xv, Xt = V[chosen].T, T[chosen].T
    try:
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000, n_jobs=1).fit(Xv, vy)
        pred = en.predict(Xt)
    except Exception:
        pred = Xt.mean(axis=1)
    return float(pearsonr(pred, ty)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakeoff", default="outputs/hp_step1_bakeoff_e100")
    ap.add_argument("--budgets", default="10,25,50,75,100,150,200")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="outputs/analysis/hp_budget_curve.json")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    rng = np.random.default_rng(42)
    results: dict[str, dict] = {}

    for cell in sorted(Path(args.bakeoff).glob("k562_*_d*")):
        seed_dir = next(iter(cell.glob("seed*")), None)
        if seed_dir is None:
            continue
        loaded = load_cell(seed_dir)
        if loaded is None:
            print(f"  {cell.name}: no usable preds/labels, skipped", flush=True)
            continue
        V, T, vy, ty = loaded
        n_models = V.shape[0]
        row = {"n_models": n_models, "budgets": {}}
        print(f"  {cell.name}: {n_models} models", flush=True)
        for b in budgets:
            if b > n_models:
                continue
            scores = []
            for _ in range(args.repeats):
                idx = rng.choice(n_models, size=b, replace=False).tolist()
                scores.append(greedy_then_stack(V, T, vy, ty, idx))
            row["budgets"][str(b)] = {
                "mean": float(np.mean(scores)),
                "sd": float(np.std(scores)),
                "n_rep": len(scores),
            }
            print(f"      budget {b:4d}: {np.mean(scores):.4f} +- {np.std(scores):.4f}", flush=True)
        results[cell.name] = row

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
