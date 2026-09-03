"""Report the three v2 oracle comparisons: per-fold test metrics, ensembling curve, controls.

Everything here is measured on TEST folds. Each model's test fold was never in its train or val
split, so unlike v1 the number is not the one early stopping selected on.

  1. ENSEMBLE     per-fold test r/MSE, plus the mean across folds. Peter's point: with a rotating
                  split the honest summary is the average across folds, not any single fold.
  2. CURVE        on fold 0's test set, the ensembling gain at 1, 2, 4 and 8 models. Averaged over
                  random subsets at each size, so the curve is not an artefact of seed ordering.
  3. CONTROLS     single-factor changes against the matching main-config model: partial unfreeze,
                  and the reference roll_n shift.
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy.stats import pearsonr


def load(d):
    j = os.path.join(d, "test_metrics.json")
    if not os.path.exists(j):
        return None
    r = json.load(open(j))
    p = os.path.join(d, "test_predictions.npz")
    r["_pred"] = np.load(p, allow_pickle=True) if os.path.exists(p) else None
    return r


def m_of(y, p):
    return pearsonr(y, p)[0], float(np.mean((y - p) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/oracle_v2")
    a = ap.parse_args()

    print("1. ENSEMBLE - one model per fold, scored on its own held-out test fold")
    print(f"   {'fold':>4} {'n_test':>8} {'test r':>8} {'test MSE':>9} {'val r':>8} {'epoch':>6}")
    rows = []
    for f in range(10):
        r = load(f"{a.root}/fold_{f}")
        if not r or not r.get("test_metrics"):
            print(f"   {f:>4} {'pending':>8}")
            continue
        t = r["test_metrics"]
        rows.append((t["pearson"], t["mse"], r["best_val_pearson"]))
        print(f"   {f:>4} {t['n']:>8,} {t['pearson']:>8.4f} {t['mse']:>9.4f} "
              f"{r['best_val_pearson']:>8.4f} {r['best_epoch'] + 1:>6}")
    if rows:
        arr = np.array(rows)
        print(f"   {'MEAN':>4} {'':>8} {arr[:, 0].mean():>8.4f} {arr[:, 1].mean():>9.4f} "
              f"{arr[:, 2].mean():>8.4f}")
        print(f"   {'SD':>4} {'':>8} {arr[:, 0].std():>8.4f} {arr[:, 1].std():>9.4f} "
              f"{arr[:, 2].std():>8.4f}")
        print(f"   val - test = {arr[:, 2].mean() - arr[:, 0].mean():+.4f}  "
              f"(positive means the val fold is optimistic, which is the reason for a test fold)")

    print("\n2. WITHIN-FOLD ENSEMBLING CURVE - fold 0 test set, seeds 1-8")
    # fold_0 is fold 0 / seed 42 / unfreeze-all / crop - the same config as the prototype seeds, so
    # it counts as the 8th member and seed 8 was never trained.
    preds, y = [], None
    for d in [f"{a.root}/proto_fold0_seed{s}" for s in range(1, 8)] + [f"{a.root}/fold_0"]:
        r = load(d)
        if r and r.get("_pred") is not None:
            preds.append(np.asarray(r["_pred"]["y_pred"], float))
            y = np.asarray(r["_pred"]["y_true"], float)
    print(f"   {len(preds)} of 8 models available (7 prototype seeds + the fold-0 ensemble member)")
    if len(preds) >= 2:
        P = np.stack(preds)
        print(f"   {'k':>3} {'test r':>8} {'test MSE':>9} {'gain vs k=1':>12}")
        base = None
        for k in (1, 2, 4, 8):
            if k > len(P):
                continue
            combos = list(itertools.combinations(range(len(P)), k))
            if len(combos) > 40:
                rng = np.random.default_rng(0)
                combos = [tuple(rng.choice(len(P), k, replace=False)) for _ in range(40)]
            rs, ms = [], []
            for c in combos:
                rr, mm = m_of(y, P[list(c)].mean(axis=0))
                rs.append(rr)
                ms.append(mm)
            r_, m_ = float(np.mean(rs)), float(np.mean(ms))
            if base is None:
                base = r_
            print(f"   {k:>3} {r_:>8.4f} {m_:>9.4f} {r_ - base:>+12.4f}")

    print("\n3. CONTROLS - one factor changed, vs the main config on the same test fold")
    main_r = load(f"{a.root}/fold_0")
    if main_r and main_r.get("test_metrics"):
        mt = main_r["test_metrics"]
        print(f"   {'config':<34} {'test r':>8} {'test MSE':>9} {'delta r':>9}")
        print(f"   {'unfreeze all + crop (main)':<34} {mt['pearson']:>8.4f} "
              f"{mt['mse']:>9.4f} {'-':>9}")
        for d, lab in ((f"{a.root}/ctrl_uf45", "unfreeze 4,5 + crop"),
                       (f"{a.root}/ctrl_roll", "unfreeze all + roll_n (reference)")):
            c = load(d)
            if c and c.get("test_metrics"):
                ct = c["test_metrics"]
                print(f"   {lab:<34} {ct['pearson']:>8.4f} {ct['mse']:>9.4f} "
                      f"{ct['pearson'] - mt['pearson']:>+9.4f}")
            else:
                print(f"   {lab:<34} {'pending':>8}")


if __name__ == "__main__":
    main()
