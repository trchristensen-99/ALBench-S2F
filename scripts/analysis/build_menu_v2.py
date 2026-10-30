#!/usr/bin/env python
"""Menu v2: the student configurations found by AutoResearch under the study's input standard (20 real flank bases per side, shift +-10, 240-bp input).

Menu v1 was built from searches on the bare 200-bp insert (top configuration of every run plus the two best of the rest). v2 uses only searches that saw the
standard input, all on the canonical-size 30k genomic pool: the standard loop (ctx35), the two manager arms (mgr35), the steering arms a0 and a4 (st38) and the
long single-call runs (lng35, whatever they have finished). Per run the three models with the best validation r are candidates (identical sources are kept once),
then 20 are picked greedily by what they add to the stacked ensemble: at every step the candidate that gives the best 5-fold cross-validated Pearson r of a
non-negative least-squares stack of z-scored validation predictions (V0c, never a test set).

Entries have the format of menu v1 (id, cell, idea, orig_val, n_params, source, train_config, family), so ``menu_retrain.py --menu`` can train them at any size.

    python scripts/analysis/build_menu_v2.py --out outputs/menus_oct30/V2.json [--k 20] [--per-run 3]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np
from scipy.optimize import nnls

AR = "outputs/archsearch_reps_oct22"
RUNS = [
    f"{AR}/genomic_30k_ctx35/rep*",
    f"{AR}/genomic_30k_mgr35_c15/rep*",
    f"{AR}/genomic_30k_mgr35_e15/rep*",
    f"{AR}/st38_a0_genomic_30k/rep*",
    f"{AR}/st38_a4_genomic_30k/rep*",
    f"{AR}/genomic_30k_lng35/rep*",
]


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def candidates(per_run: int):
    cands, yv = [], None
    for pat in RUNS:
        for d in sorted(glob.glob(pat)):
            if not os.path.exists(f"{d}/labels.npz"):
                continue
            lab = np.load(f"{d}/labels.npz")
            y = lab["val_labels"].astype(np.float64)
            if yv is None:
                yv = y
            elif not np.allclose(yv, y):
                raise SystemExit(f"{d}: validation labels differ from the other runs")
            ms = []
            for mf in sorted(glob.glob(f"{d}/r*_arch_*_meta.json")):
                try:
                    m = json.load(open(mf))
                    v = np.load(mf.replace("_meta.json", ".npz"))["val_pred"].astype(np.float64)
                except Exception:
                    continue
                if not np.isfinite(v).all() or np.abs(v).max() > 1e3 or not m.get("source"):
                    continue
                ms.append((pearson(v, y), m, v))
            ms.sort(key=lambda x: -x[0])
            cell = "/".join(d.split("/")[-2:])
            for r, m, v in ms[:per_run]:
                cands.append(
                    {
                        "id": f"{cell.replace('/', '_')}_{m['model_id']}",
                        "cell": cell,
                        "idea": (m.get("idea") or "")[:200],
                        "orig_val": r,
                        "n_params": m.get("n_params"),
                        "source": m["source"],
                        "train_config": m.get("train_config", {}),
                        "family": "standard-input search (30k genomic)",
                        "_v": v,
                    }
                )
    seen, out = set(), []
    for c in sorted(cands, key=lambda c: -c["orig_val"]):
        h = hashlib.sha1(c["source"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(c)
    return out, yv


def cv_score(Z, y, folds) -> float:
    pred = np.zeros_like(y)
    for tr, te in folds:
        w, _ = nnls(Z[:, tr].T, y[tr] - y[tr].mean())
        pred[te] = w @ Z[:, te]
    return pearson(pred, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--per-run", type=int, default=3)
    a = ap.parse_args()
    cands, yv = candidates(a.per_run)
    if len(cands) < a.k:
        raise SystemExit(f"only {len(cands)} distinct candidates for K={a.k}")
    V = np.stack([c["_v"] for c in cands])
    Z = (V - V.mean(1, keepdims=True)) / V.std(1, keepdims=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(yv))
    parts = np.array_split(idx, 5)
    folds = [
        (np.concatenate([p for j, p in enumerate(parts) if j != i]), parts[i]) for i in range(5)
    ]
    chosen: list[int] = []
    while len(chosen) < a.k:
        best, best_s = None, -9.0
        for j in range(len(cands)):
            if j in chosen:
                continue
            s = cv_score(Z[chosen + [j]], yv, folds)
            if s > best_s:
                best, best_s = j, s
        chosen.append(best)
        print(
            f"step {len(chosen):2d}: + {cands[best]['id']} (val r {cands[best]['orig_val']:.4f}) -> cv stack r {best_s:.4f}",
            flush=True,
        )
    entries = []
    for j in chosen:
        e = {k: v for k, v in cands[j].items() if not k.startswith("_")}
        entries.append(e)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(entries, open(a.out, "w"), indent=1, default=float)
    n_run = len({e["cell"] for e in entries})
    print(
        f"wrote {a.out}: {len(entries)} configurations from {n_run} runs (of {len({c['cell'] for c in cands})} with candidates, {len(cands)} distinct candidates)"
    )


if __name__ == "__main__":
    main()
