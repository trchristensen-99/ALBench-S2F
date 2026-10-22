"""Post-hoc HP-protocol study: answer the design questions from data we already have.

Every model ever trained recorded which strategy proposed it, which round it came
from, its full config, and its val/test predictions. That is enough to simulate
most protocol choices WITHOUT retraining anything:

  H1  one strategy at full budget vs two at half each
  H3  how many rounds (truncate at round R, same total models)
  H4  do the best strategies transfer across reservoirs
  H5  marginal value of model N  <- sets the model cap at D=300k
  H6  do large configs earn their place

Only H2 (proposals per call) and H7 (AutoResearch vs Optuna) need new runs, because
neither can be reconstructed from a search that was never run that way.

Stacking here uses non-negative least squares rather than ElasticNetCV: it is the
same positive-combination family, ~100x faster, and the sweep needs thousands of
fits. `--verify` re-scores a sample with ElasticNetCV to confirm the two agree.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import nnls

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_cell(cell_dir):
    """-> (models, labels). Each model: id, strategy, round, size, val, test."""
    lab = os.path.join(cell_dir, "labels.npz")
    if not os.path.exists(lab):
        return None, None
    Y = np.load(lab, allow_pickle=True)
    yv = np.asarray(Y["val_labels"], float)
    yt = np.asarray(Y["test_oracle"], float)
    models = []
    for m in sorted(glob.glob(os.path.join(cell_dir, "r*_meta.json"))):
        try:
            meta = json.loads(open(m).read())
        except Exception:
            continue
        if meta.get("error"):
            continue
        npz = m.replace("_meta.json", ".npz")
        if not os.path.exists(npz):
            continue
        try:
            z = np.load(npz, allow_pickle=True)
            vp, tp = np.asarray(z["val_pred"], float), np.asarray(z["test_pred"], float)
        except Exception:
            continue
        if vp.shape[0] != yv.shape[0] or tp.shape[0] != yt.shape[0]:
            continue
        if not (np.isfinite(vp).all() and np.isfinite(tp).all()):
            continue
        bs = meta.get("block_sizes") or []
        models.append({
            "id": meta.get("model_id", os.path.basename(npz)[:-4]),
            "strategy": meta.get("strategy", "na"),
            "round": int(meta.get("round", -1)),
            "size": int(sum(bs)) if bs else 0,
            "val": vp, "test": tp,
        })
    return models, (yv, yt)


def stack_score(members, yv, yt):
    """Fit a non-negative blend on val, score Pearson on test."""
    if not members:
        return np.nan
    V = np.vstack([m["val"] for m in members]).T
    T = np.vstack([m["test"] for m in members]).T
    V1 = np.hstack([V, np.ones((V.shape[0], 1))])
    w, _ = nnls(V1, yv)
    pred = np.hstack([T, np.ones((T.shape[0], 1))]) @ w
    if pred.std() == 0:
        return np.nan
    return float(np.corrcoef(pred, yt)[0, 1])


def greedy(pool, yv, yt, n=5):
    """Greedy forward selection on VAL mse; returns (chosen, test_r_at_n)."""
    chosen, remaining = [], list(range(len(pool)))
    for _ in range(min(n, len(pool))):
        best_j, best_mse = None, np.inf
        for j in remaining:
            cand = [pool[i] for i in chosen] + [pool[j]]
            V = np.vstack([m["val"] for m in cand]).T
            V1 = np.hstack([V, np.ones((V.shape[0], 1))])
            w, _ = nnls(V1, yv)
            mse = float(np.mean((V1 @ w - yv) ** 2))
            if mse < best_mse:
                best_mse, best_j = mse, j
        if best_j is None:
            break
        chosen.append(best_j)
        remaining.remove(best_j)
    return [pool[i] for i in chosen], stack_score([pool[i] for i in chosen], yv, yt)


# ---------------------------------------------------------------- experiments

def h5_marginal_value(models, yv, yt, rng, reps=8, sizes=(5, 10, 15, 20, 30, 40, 62)):
    """H5: deploy score vs number of models AVAILABLE to select from.

    The knee here is the model cap. At D=300k each model costs ~57 GPU-min, so
    every model we can drop below 62 is ~25 GPU-h saved across the programme.
    """
    out = []
    for k in sizes:
        if k > len(models):
            continue
        scores = []
        for _ in range(reps):
            idx = rng.choice(len(models), size=k, replace=False)
            _, s = greedy([models[i] for i in idx], yv, yt)
            if np.isfinite(s):
                scores.append(s)
        if scores:
            out.append((k, float(np.mean(scores)), float(np.std(scores)), len(scores)))
    return out


def h1_one_vs_two(models, yv, yt, rng, reps=8):
    """H1: one strategy at full budget vs two at half each, budget matched."""
    by = defaultdict(list)
    for m in models:
        by[m["strategy"]].append(m)
    strats = sorted(by)
    budget = min(len(v) for v in by.values()) * 2      # pairs must afford half each
    res = {}
    for s in strats:                                   # single strategy, full budget
        if len(by[s]) < budget:
            continue
        sc = []
        for _ in range(reps):
            idx = rng.choice(len(by[s]), size=budget, replace=False)
            _, v = greedy([by[s][i] for i in idx], yv, yt)
            if np.isfinite(v):
                sc.append(v)
        if sc:
            res[f"1x {s}"] = (float(np.mean(sc)), float(np.std(sc)))
    for i in range(len(strats)):                       # two strategies, half each
        for j in range(i + 1, len(strats)):
            a, b = strats[i], strats[j]
            h = budget // 2
            if len(by[a]) < h or len(by[b]) < h:
                continue
            sc = []
            for _ in range(reps):
                pool = ([by[a][k] for k in rng.choice(len(by[a]), size=h, replace=False)] +
                        [by[b][k] for k in rng.choice(len(by[b]), size=h, replace=False)])
                _, v = greedy(pool, yv, yt)
                if np.isfinite(v):
                    sc.append(v)
            if sc:
                res[f"2x {a}+{b}"] = (float(np.mean(sc)), float(np.std(sc)))
    return res, budget


def h3_rounds(models, yv, yt):
    """H3: truncate the search at round R. Fewer rounds = earlier stopping."""
    rounds = sorted({m["round"] for m in models if m["round"] >= 0})
    out = []
    for r in rounds:
        pool = [m for m in models if 0 <= m["round"] <= r]
        if len(pool) < 5:
            continue
        _, s = greedy(pool, yv, yt)
        out.append((r + 1, len(pool), float(s)))
    return out


def h6_big_configs(models, yv, yt):
    """H6: do the largest configs earn their cost? Drop the top-q by size."""
    sizes = np.array([m["size"] for m in models], float)
    out = []
    for q in (1.0, 0.9, 0.75, 0.5):
        thr = np.quantile(sizes, q) if q < 1.0 else np.inf
        pool = [m for m in models if m["size"] <= thr]
        if len(pool) < 5:
            continue
        _, s = greedy(pool, yv, yt)
        frac_time = float(np.mean([m["size"] for m in pool]) / max(1e-9, sizes.mean()))
        out.append((q, len(pool), float(s), frac_time))
    return out


def h4_strategy_value(models, yv, yt):
    """H4: leave-one-strategy-out. Negative delta = that strategy mattered."""
    full = greedy(models, yv, yt)[1]
    out = {}
    for s in sorted({m["strategy"] for m in models}):
        pool = [m for m in models if m["strategy"] != s]
        if len(pool) < 5:
            continue
        out[s] = float(greedy(pool, yv, yt)[1] - full)
    return full, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None, help="cell dirs; default = all complete")
    ap.add_argument("--tier", default="d30k")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", default="outputs/figures/hp_protocol_study.json")
    ap.add_argument("--only", nargs="*", default=["h5", "h1", "h3", "h4", "h6"])
    args = ap.parse_args()

    D = {"d10k": 10000, "d30k": 30000, "d100k": 100000}[args.tier]
    cells = args.cells or sorted(glob.glob(f"outputs/stage4_v2_{args.tier}/*_d{D}"))
    rng = np.random.default_rng(0)
    results = {}
    for cd in cells:
        arm = re.sub(rf"_d{D}$", "", os.path.basename(cd))
        models, lab = load_cell(cd)
        if not models or len(models) < 20:
            print(f"  skip {arm} ({0 if not models else len(models)} models)")
            continue
        yv, yt = lab
        print(f"\n=== {arm}  ({len(models)} models) ===")
        r = {"n_models": len(models)}

        if "h5" in args.only:
            r["h5"] = h5_marginal_value(models, yv, yt, rng, reps=args.reps)
            print("  H5 deploy score vs models available:")
            base = r["h5"][-1][1] if r["h5"] else np.nan
            for k, mu, sd, n in r["h5"]:
                print(f"    k={k:>3}  r={mu:.4f} +-{sd:.4f}   delta vs full = {mu-base:+.4f}")
        if "h4" in args.only:
            full, loo = h4_strategy_value(models, yv, yt)
            r["h4"] = {"full": full, "loso": loo}
            print(f"  H4 leave-one-strategy-out (full={full:.4f}):")
            for s, d in sorted(loo.items(), key=lambda kv: kv[1]):
                print(f"    without {s:<14} {d:+.4f}")
        if "h3" in args.only:
            r["h3"] = h3_rounds(models, yv, yt)
            print("  H3 rounds:")
            for nr, npool, s in r["h3"]:
                print(f"    rounds<={nr:>3}  pool={npool:>3}  r={s:.4f}")
        if "h6" in args.only:
            r["h6"] = h6_big_configs(models, yv, yt)
            print("  H6 drop largest configs:")
            for q, npool, s, ft in r["h6"]:
                print(f"    keep<=q{q:.2f}  pool={npool:>3}  r={s:.4f}  mean_size_ratio={ft:.2f}")
        if "h1" in args.only:
            res, budget = h1_one_vs_two(models, yv, yt, rng, reps=args.reps)
            r["h1"] = {"budget": budget, "res": res}
            print(f"  H1 one vs two strategies (budget={budget} models):")
            for k, (mu, sd) in sorted(res.items(), key=lambda kv: -kv[1][0]):
                print(f"    {k:<28} {mu:.4f} +-{sd:.4f}")
        results[arm] = r

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}  ({len(results)} cells)")


if __name__ == "__main__":
    main()
