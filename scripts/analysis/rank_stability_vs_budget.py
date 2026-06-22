"""Does capping the bake-off at N reorder the strategy ranking vs the full 200?
Use the deep genomic anchor (3 seeds). For each cap N, rank strategies by
mean(best-val-by-N) over seeds; report Spearman vs the N=200 ranking and whether
the top-6 set changes. Also print each strategy's truncation COST = best200-bestN
(the magnitude the trunc_N band hides). Plus epoch stats to explain the slowdown."""

import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = "outputs/hp_step1_bakeoff_e100/k562_genomic_d30000"
SEEDS = ["seed42_0", "seed43_1", "seed44_2"]
CAPS = [50, 75, 100, 150, 200]
TOPK = 6


def best_by_cap(cd, cap):
    vals = []
    for m in sorted(glob.glob(os.path.join(cd, "r*_meta.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        vp = d.get("val_pearson")
        if vp is None:
            continue
        vals.append((int(d.get("round", -1)), float(vp)))
    vals.sort()
    v = [x[1] for x in vals[:cap]]
    return max(v) if v else np.nan


def strat_means(cap):
    out = {}
    for s in SEEDS:
        for cd in sorted(glob.glob(os.path.join(ROOT, s, "*"))):
            if not os.path.isdir(cd):
                continue
            b = best_by_cap(cd, cap)
            if np.isfinite(b):
                out.setdefault(os.path.basename(cd), []).append(b)
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def rank(d, keys):
    order = sorted(keys, key=lambda k: -d[k])
    return {k: i for i, k in enumerate(order)}, order


def main():
    full = strat_means(200)
    keys = sorted(full)
    full_rank, full_order = rank(full, keys)
    full_top = set(full_order[:TOPK])

    print("=== rank stability vs full-200 (genomic anchor, 3 seeds) ===")
    print(f"  {'cap':>4s}  {'spearman':>8s}  {'top6_kept':>9s}  notes")
    per_cap = {}
    for cap in CAPS:
        m = strat_means(cap)
        per_cap[cap] = m
        r = [full[k] for k in keys]
        c = [m[k] for k in keys]
        rho = spearmanr(r, c).statistic
        _, order = rank(m, keys)
        kept = len(set(order[:TOPK]) & full_top)
        moved = (
            ""
            if cap == 200
            else "; ".join(
                f"{k}:{full_rank[k]}->{rank(m, keys)[0][k]}"
                for k in keys
                if abs(full_rank[k] - rank(m, keys)[0][k]) >= 2
            )
        )
        print(f"  {cap:>4d}  {rho:8.4f}  {kept:>6d}/{TOPK}   {moved[:80]}")

    print("\n=== per-strategy truncation cost  best200 - best75  (mean over seeds) ===")
    m75 = per_cap[75]
    rows = sorted(keys, key=lambda k: -(full[k] - m75[k]))
    print(f"  {'strategy':22s} {'best200':>8s} {'best75':>7s} {'cost':>7s} {'rank200':>7s}")
    for k in rows:
        print(f"  {k:22s} {full[k]:8.4f} {m75[k]:7.4f} {full[k] - m75[k]:7.4f} {full_rank[k]:7d}")

    # epoch stats to explain the slowdown vs old 15-epoch runs
    eps, es = [], 0
    n = 0
    for s in SEEDS:
        for m in glob.glob(os.path.join(ROOT, s, "*", "r*_meta.json")):
            try:
                d = json.load(open(m))
            except Exception:
                continue
            if "epochs_trained" in d:
                eps.append(int(d["epochs_trained"]))
                es += int(bool(d.get("early_stopped")))
                n += 1
    eps = np.array(eps)
    print(f"\n=== epochs_trained (folder=e100; legacy HP search ran e15) ===")
    print(
        f"  n={n}  median={np.median(eps):.0f}  mean={eps.mean():.1f}  p90={np.percentile(eps, 90):.0f}  max={eps.max()}  early_stopped={100 * es / max(1, n):.0f}%"
    )


if __name__ == "__main__":
    main()
