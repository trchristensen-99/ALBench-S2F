# ruff: noqa
#!/usr/bin/env python
"""Two quick checks from the Oct-28 discussion.

(1) Does the best architecture depend on the dataset size? Menu v1 retrained at 10k, 30k and 100k on the genomic, random and motif CT-enriched pools (v1_{D}k_{pool}_r1, 20 configurations, one seed):
    Spearman correlation across configurations of validation r and of genomic-test r between sizes, and the overlap of the best 5 / best 10 (retrained validation r).
(2) Near-duplicate rate inside a pool: 6-mer count vectors (4096 dimensions, unit length), cosine similarity to the nearest other sequence. The threshold is calibrated on pairs made by mutating
    genomic sequences at 5%, 10% and 20% per base (cosine of such a pair against the median cosine of unrelated pairs), so "similar" has a meaning in sequence identity. Pools: Zoonomia orthologs,
    genomic (reference), uniform random (null), 10% mutagenesis (positive control: every parent may appear twice).

    python scripts/analysis/analyze_rankD_dedup.py [--out outputs/justification_oct22] [--n 30000]      (cpuq)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_oct32 as a32  # noqa: E402

K = 6
NEST = "outputs/grid_pools/k562_nested"
POW = 4 ** np.arange(K - 1, -1, -1)


def an_rank(out):
    res = {}
    for pool in ("genomic", "random", "motif_ct_enriched"):
        per = {}
        for D in (10, 30, 100):
            c = a32.load_all(f"v1_{D}k_{pool}_r1")
            if not c:
                continue
            v = {}
            for r in c["rows"]:
                v[r["meta"].get("menu_entry")] = (
                    a32.pearson(r["val"], c["val_y"]),
                    a32.pearson(r["test"]["genomic"], c["y"]["genomic"]),
                )
            per[D] = v
        cell = {"n": {D: len(v) for D, v in per.items()}, "pairs": {}}
        for d1, d2 in itertools.combinations(sorted(per), 2):
            common = sorted(set(per[d1]) & set(per[d2]))
            if len(common) < 8:
                continue
            a = np.array([per[d1][e] for e in common])
            b = np.array([per[d2][e] for e in common])
            top = lambda x, k: set(np.argsort(-x)[:k])
            cell["pairs"][f"{d1}k_vs_{d2}k"] = {
                "n": len(common),
                "spearman_val": float(spearmanr(a[:, 0], b[:, 0])[0]),
                "spearman_test": float(spearmanr(a[:, 1], b[:, 1])[0]),
                "top5_overlap": len(top(a[:, 0], 5) & top(b[:, 0], 5)),
                "top10_overlap": len(top(a[:, 0], 10) & top(b[:, 0], 10)),
            }
        res[pool] = cell
        print(pool, json.dumps(cell["pairs"]), flush=True)
    out["rankD"] = res


def kmer_matrix(seqs):
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(seqs), 4**K), np.float32)
    for i, s in enumerate(seqs):
        a = np.array([code.get(ch, -1) for ch in str(s).upper()], np.int64)
        if len(a) < K:
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, K)
        ok = (w >= 0).all(1)
        np.add.at(X[i], (w[ok] * POW).sum(1), 1.0)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def nearest(X, chunk=2000):
    best = np.zeros(len(X), np.float32)
    for i in range(0, len(X), chunk):
        S = X[i : i + chunk] @ X.T
        S[np.arange(S.shape[0]), np.arange(i, i + S.shape[0])] = -1
        best[i : i + chunk] = S.max(1)
    return best


def mutate(seqs, rate, rng):
    out = []
    for s in seqs:
        a = np.array(list(s))
        hit = rng.random(len(a)) < rate
        for j in np.where(hit)[0]:
            a[j] = rng.choice([x for x in "ACGT" if x != a[j]])
        out.append("".join(a))
    return out


def an_dedup(out, n):
    rng = np.random.default_rng(0)
    ref = [
        str(s) for s in np.load(f"{NEST}/genomic_d30000.npz", allow_pickle=True)["sequences"][:3000]
    ]
    X0 = kmer_matrix(ref)
    unrelated = (X0[:1500] * X0[1500:3000]).sum(1)
    calib = {
        "unrelated_median": float(np.median(unrelated)),
        "unrelated_p99": float(np.percentile(unrelated, 99)),
    }
    for rate in (0.05, 0.10, 0.20, 0.30):
        Xm = kmer_matrix(mutate(ref[:1500], rate, rng))
        cos = (X0[:1500] * Xm).sum(1)
        calib[f"mut{rate}"] = {"median": float(np.median(cos)), "p5": float(np.percentile(cos, 5))}
    print("calibration", json.dumps(calib), flush=True)
    thr = {f"identity~{int((1 - r) * 100)}%": calib[f"mut{r}"]["p5"] for r in (0.05, 0.10, 0.20)}
    res = {"calibration": calib, "thresholds": thr, "pools": {}}
    for name in ("zoonomia_orthologs", "genomic", "random", "mutagenesis_mr10", "evoaug_paper2023"):
        f = f"{NEST}/{name}_d30000.npz"
        if not os.path.exists(f):
            continue
        seqs = [str(s) for s in np.load(f, allow_pickle=True)["sequences"][:n]]
        nn = nearest(kmer_matrix(seqs))
        res["pools"][name] = {
            "n": len(seqs),
            "nn_cos_quantiles": {q: float(np.percentile(nn, q)) for q in (5, 25, 50, 75, 95, 99)},
            **{f"share_above_{k}": float((nn >= v).mean()) for k, v in thr.items()},
        }
        print(name, json.dumps(res["pools"][name]), flush=True)
    out["dedup"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument("--n", type=int, default=30000)
    a = ap.parse_args()
    out = {}
    for fn, args in ((an_rank, ()), (an_dedup, (a.n,))):
        try:
            fn(out, *args)
        except Exception:
            import traceback

            traceback.print_exc()
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/data_rankD_dedup.json", "w"), indent=1, default=float)
    print("wrote", f"{a.out}/data_rankD_dedup.json")


if __name__ == "__main__":
    main()
