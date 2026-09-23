#!/usr/bin/env python
"""Build the from-scratch scaling curves from Stage-4 cells.

Applies the deploy rule exactly as specified: greedy-forward selection on the cell's
OWN validation split, stop at N=5, then ElasticNetCV weights fitted on that same
validation split. Nothing is shared across cells, so a new reservoir strategy would
run the identical procedure -- that is what makes the arms comparable.

Reports the ensemble AND the best single model, because the gap between them is the
part that justifies ensembling at all.
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


def load_models(cell: Path):
    """Return per-model val/test prediction matrices for one cell."""
    V, T, ids = [], [], []
    for meta in sorted(glob.glob(str(cell / "**" / "*_meta.json"), recursive=True)):
        npz = meta.replace("_meta.json", ".npz")
        if not os.path.exists(npz):
            continue
        try:
            with np.load(npz) as z:
                if "val_pred" not in z:
                    continue
                tkey = next((k for k in ("test_pred_genomic", "test_pred") if k in z), None)
                if tkey is None:
                    continue
                V.append(z["val_pred"])
                T.append(z[tkey])
                ids.append(os.path.basename(meta))
        except Exception:
            continue
    if not V:
        return None
    n = min(len(v) for v in V)
    m = min(len(t) for t in T)
    return np.vstack([v[:n] for v in V]), np.vstack([t[:m] for t in T]), ids


def select_and_score(V, T, vy, ty, n=N_ENSEMBLE):
    chosen: list[int] = []
    for _ in range(min(n, V.shape[0])):
        best, best_mse = None, np.inf
        for c in range(V.shape[0]):
            if c in chosen:
                continue
            mse = float(np.mean((V[chosen + [c]].mean(axis=0) - vy) ** 2))
            if mse < best_mse:
                best, best_mse = c, mse
        if best is None:
            break
        chosen.append(best)
    try:
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000, n_jobs=1)
        en.fit(V[chosen].T, vy)
        pred = en.predict(T[chosen].T)
    except Exception:
        pred = T[chosen].mean(axis=0)
    ens = float(pearsonr(pred, ty)[0])
    singles = [float(pearsonr(T[i], ty)[0]) for i in range(T.shape[0])]
    best_single_by_val = int(np.argmin([np.mean((V[i] - vy) ** 2) for i in range(V.shape[0])]))
    return {
        "ensemble_r": ens,
        "best_single_r_by_val": singles[best_single_by_val],
        "oracle_best_single_r": max(singles),
        "n_models": int(V.shape[0]),
        "chosen": chosen,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/stage4_base0")
    ap.add_argument("--out", default="outputs/analysis/stage4_curves.json")
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for cell in sorted(Path(args.root).glob("*_d*")):
        loaded = load_models(cell)
        if loaded is None:
            print(f"  {cell.name}: no models yet", flush=True)
            continue
        V, T, _ = loaded
        lab = next(iter(cell.glob("**/labels.npz")), None)
        if lab is None:
            print(f"  {cell.name}: {V.shape[0]} models but no labels.npz", flush=True)
            continue
        with np.load(lab, allow_pickle=True) as z:
            keys = list(z.keys())
            vy = z["val_labels"] if "val_labels" in keys else None
            # The driver writes its held-out sets as oracle_<set>. Use oracle_genomic
            # to match test_pred_genomic; these align by construction, whereas the
            # separately re-labelled eval sets in outputs/eval_sets_v3 are a DIFFERENT
            # sequence set (40,718 vs 31,435) and would silently misalign if zipped.
            ty = next(
                (z[k] for k in ("oracle_genomic", "test_oracle", "test_labels") if k in keys),
                None,
            )
        if vy is None or ty is None:
            print(f"  {cell.name}: labels.npz lacks val/test keys ({keys})", flush=True)
            continue
        vy, ty = np.asarray(vy)[: V.shape[1]], np.asarray(ty)[: T.shape[1]]
        r = select_and_score(V, T, vy, ty)
        arm, _, d = cell.name.rpartition("_d")
        r.update(arm=arm, D=int(d))
        results[cell.name] = r
        print(
            f"  {arm:20s} D={int(d):>7,}  n={r['n_models']:3d}  "
            f"ensemble={r['ensemble_r']:.4f}  best_single={r['best_single_r_by_val']:.4f}  "
            f"gain={r['ensemble_r'] - r['best_single_r_by_val']:+.4f}",
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}  ({len(results)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
