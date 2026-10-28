# ruff: noqa
#!/usr/bin/env python
"""Activity, GC and 4-mer distributions of every reservoir at 30k, and how many Zoonomia orthologue windows survive each uncalled-base (N) threshold.

Peter (Sep 30 notes): "look at activity and k-mer distributions per strategy" and "Zoonomia: remove Ns and add flanks, plot sequence counts at N thresholds". The nested
canonical pools are used (seed 42, oracle_v3 labels), so the numbers describe the corpora the scaling cells trained on, not the legacy k562_cumulative pools.

    python scripts/analysis/analyze_reservoir_distributions.py pools    --out outputs/justification_oct22/data_reservoirs.json     (cpuq, a few minutes)
    python scripts/analysis/analyze_reservoir_distributions.py zoonomia --out outputs/justification_oct22/data_zoonomia_n.json     (cpuq, reads the 71 GB alignment)

Run from the repository root. Everything is vectorised: sequences become a (N, L) uint8 matrix once, GC and k-mer counts are array operations on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())

POOLS = "outputs/grid_pools/k562_nested"
TS = "data/k562/test_sets_ag_s2_chrsplit"
STRATS = [
    "genomic",
    "zoonomia_orthologs",
    "evoaug_paper2023",
    "mutagenesis_mr10",
    "motif_ct_enriched",
    "motif_shared_core",
    "random",
    "random_gc_sweep",
]
K = 4
LUT = np.full(256, 4, np.uint8)
for i, ch in enumerate("ACGT"):
    LUT[ord(ch)] = i
    LUT[ord(ch.lower())] = i


def to_codes(seqs) -> np.ndarray:
    """(N, Lmax) uint8: 0-3 for ACGT, 4 for N or padding."""
    seqs = [str(s) for s in seqs]
    lens = np.fromiter((len(s) for s in seqs), np.int64, len(seqs))
    if (lens == lens[0]).all():
        raw = np.frombuffer("".join(seqs).encode(), np.uint8).reshape(len(seqs), int(lens[0]))
        return LUT[raw]
    out = np.full((len(seqs), int(lens.max())), 4, np.uint8)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = LUT[np.frombuffer(s.encode(), np.uint8)]
    return out


def kmer_counts(codes: np.ndarray, k: int = K) -> np.ndarray:
    n, length = codes.shape
    c = codes.astype(np.int32)
    idx = np.zeros((n, length - k + 1), np.int32)
    ok = np.ones((n, length - k + 1), bool)
    for j in range(k):
        w = c[:, j : length - k + 1 + j]
        ok &= w < 4
        idx = idx * 4 + np.minimum(w, 3)
    return np.bincount(idx[ok], minlength=4**k).astype(np.float64)


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def js_bits(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        s = a > 0
        return float((a[s] * np.log2(a[s] / b[s])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def load_pool(name: str):
    z = np.load(f"{POOLS}/{name}_d30000.npz", allow_pickle=True)
    return list(z["sequences"]), np.asarray(z["oracle_labels"], float)


def load_refs():
    from albench.core.paths import resolve

    z = np.load(resolve("val_v0c"), allow_pickle=True)
    v = (list(z["sequences"]), np.asarray(z["oracle_labels"], float))
    z = np.load(f"{TS}/genomic_oracle.npz", allow_pickle=True)
    t = (list(z["sequences"]), np.asarray(z["oracle_mean"], float))
    return {"val_v0c": v, "genomic_test": t}


def describe(seqs, y, ref):
    codes = to_codes(seqs)
    valid = codes < 4
    nvalid = valid.sum(1)
    gc = ((codes == 1) | (codes == 2)).sum(1) / np.maximum(nvalid, 1)
    cnt = kmer_counts(codes)
    p = cnt / cnt.sum()
    qs = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    return {
        "n": len(seqs),
        "n_unique_frac": len(set(map(str, seqs))) / len(seqs),
        "length_min": int(np.fromiter((len(s) for s in seqs), int, len(seqs)).min()),
        "length_max": int(np.fromiter((len(s) for s in seqs), int, len(seqs)).max()),
        "frac_seqs_with_N": float((nvalid < codes.shape[1]).mean()),
        "label": {
            "mean": float(y.mean()),
            "sd": float(y.std()),
            "q": {str(q): float(np.percentile(y, q)) for q in qs},
            "frac_below_ref_p10": float((y < ref["p10"]).mean()),
            "frac_above_ref_p90": float((y > ref["p90"]).mean()),
            "spearman_vs_gc": spearman(y, gc),
        },
        "gc": {
            "mean": float(gc.mean()),
            "sd": float(gc.std()),
            "q": {str(q): float(np.percentile(gc, q)) for q in qs},
        },
        "kmer": {"entropy_bits": entropy_bits(p), "p": p.tolist()},
        "_y": y,
        "_gc": gc,
    }


def an_pools(out):
    refs = load_refs()
    pools = {n: load_pool(n) for n in STRATS}
    gseqs, gy = pools["genomic"]
    ref = {"p10": float(np.percentile(gy, 10)), "p90": float(np.percentile(gy, 90))}
    res = {n: describe(s, y, ref) for n, (s, y) in {**pools, **refs}.items()}
    pg = np.asarray(res["genomic"]["kmer"]["p"])
    pt = np.asarray(res["genomic_test"]["kmer"]["p"])
    allv = np.concatenate([r["_y"] for r in res.values()])
    ylo, yhi = np.percentile(allv, [0.2, 99.8])
    yedges = np.linspace(ylo, yhi, 61)
    gedges = np.linspace(0.0, 1.0, 51)
    for n, r in res.items():
        p = np.asarray(r["kmer"]["p"])
        r["kmer"]["js_vs_genomic_pool"] = js_bits(p, pg)
        r["kmer"]["js_vs_genomic_test"] = js_bits(p, pt)
        y = np.clip(r.pop("_y"), ylo, yhi)
        g = r.pop("_gc")
        r["label"]["hist"] = (np.histogram(y, yedges)[0] / len(y)).tolist()
        r["gc"]["hist"] = (np.histogram(g, gedges)[0] / len(g)).tolist()
        # mean activity by GC band: where the activity of a strategy sits
        yy = np.asarray((pools.get(n) or refs[n])[1])
        bands = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]
        r["label"]["by_gc_band"] = [
            {
                "band": [lo, hi],
                "n": int(((g >= lo) & (g < hi)).sum()),
                "mean": float(yy[(g >= lo) & (g < hi)].mean())
                if ((g >= lo) & (g < hi)).any()
                else None,
            }
            for lo, hi in bands
        ]
    out_d = {
        "label_edges": yedges.tolist(),
        "gc_edges": gedges.tolist(),
        "ref": ref,
        "order": STRATS,
        "refs": list(refs),
        "strategies": res,
        "k": K,
    }
    json.dump(out_d, open(out, "w"), indent=1)
    for n, r in res.items():
        print(
            f"{n:22s} n={r['n']:6d} uniq={r['n_unique_frac']:.3f} label mean {r['label']['mean']:.3f} sd {r['label']['sd']:.3f} "
            f"<gP10 {r['label']['frac_below_ref_p10']:.2f} >gP90 {r['label']['frac_above_ref_p90']:.2f} GC {r['gc']['mean']:.3f}+-{r['gc']['sd']:.3f} "
            f"H {r['kmer']['entropy_bits']:.3f} JS(g) {r['kmer']['js_vs_genomic_pool']:.4f} JS(test) {r['kmer']['js_vs_genomic_test']:.4f} rho(GC) {r['label']['spearman_vs_gc']:+.2f}"
        )
    print("wrote", out)


def an_zoonomia(out, n_windows, budget_s, seed):
    """What the sampler's own window draw yields at each N threshold. Windows are drawn as ZoonomiaOrthologSampler.generate does: a chromosome uniformly from the
    non-excluded ones, a start uniformly along it; every non-human species row is one (window, species) pair."""
    import h5py

    from albench.core.paths import resolve
    from albench.reservoir.zoonomia_orthologs import ZoonomiaOrthologSampler

    L = 200
    THR = [0, 1, 2, 5, 10, 20]
    excl = set(ZoonomiaOrthologSampler().exclude_chroms)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    with h5py.File(str(resolve("zoonomia_alignment")), "r") as f:
        chroms = [
            (c, int(f[c]["seq"].shape[1])) for c in f if c.startswith("chr") and c not in excl
        ]
        n_sp = int(f[chroms[0][0]]["seq"].shape[0])
        species = [s.decode() if isinstance(s, bytes) else str(s) for s in f["species"][:]]
        hist = np.zeros(L + 1, np.int64)
        sp_ok = np.zeros((len(THR), n_sp), np.int64)
        per_window = []
        human_n = []
        runs = {
            "pairs_1_to_20N": 0,
            "single_run": 0,
            "ends_only_20bp": 0,
            "single_run_and_ends_only": 0,
        }
        done = 0
        for _ in range(n_windows):
            if time.time() - t0 > budget_s:
                break
            c, clen = chroms[int(rng.integers(len(chroms)))]
            st = int(rng.integers(0, clen - L))
            blk = np.asarray(f[c]["seq"][:, st : st + L])
            m = blk == 0
            nN = m.sum(1)
            human_n.append(int(nN[0]))
            nn, mm = nN[1:], m[1:]
            hist += np.bincount(nn, minlength=L + 1)
            ok = np.stack([nn <= t for t in THR])
            sp_ok += np.pad(ok, ((0, 0), (1, 0)))
            per_window.append(ok.sum(1))
            sel = (nn >= 1) & (nn <= 20)
            if sel.any():
                mk = mm[sel]
                nruns = mk[:, 0].astype(int) + (mk[:, 1:] & ~mk[:, :-1]).sum(1)
                ends = ~mk[:, 20 : L - 20].any(1)
                runs["pairs_1_to_20N"] += int(sel.sum())
                runs["single_run"] += int((nruns == 1).sum())
                runs["ends_only_20bp"] += int(ends.sum())
                runs["single_run_and_ends_only"] += int(((nruns == 1) & ends).sum())
            done += 1
        pw = np.asarray(per_window)
    pairs = int(hist.sum())
    cum = np.cumsum(hist)
    res = {
        "windows_done": done,
        "windows_requested": n_windows,
        "elapsed_s": time.time() - t0,
        "seq_len": L,
        "n_species_rows_excluding_human": n_sp - 1,
        "chroms_used": [c for c, _ in chroms],
        "excluded_chroms": sorted(excl),
        "pairs": pairs,
        "hist_n_uncalled": hist.tolist(),
        "thresholds": THR,
        "pairs_kept": [int(cum[t]) for t in THR],
        "frac_kept": [float(cum[t] / pairs) for t in THR],
        "sampler": {
            "min_called_frac": 0.95,
            "max_N_allowed_by_called_frac": int(np.floor(L * 0.05)),
            "then_drops_any_N": True,
            "frac_pass_called_frac": float(cum[int(np.floor(L * 0.05))] / pairs),
            "frac_pass_final_no_N": float(cum[0] / pairs),
        },
        "species_pass_rate": {
            str(t): (sp_ok[i, 1:] / max(done, 1)).tolist() for i, t in enumerate(THR)
        },
        "species": species[1:],
        "species_per_window_mean": {str(t): float(pw[:, i].mean()) for i, t in enumerate(THR)},
        "species_per_window_median": {
            str(t): float(np.median(pw[:, i])) for i, t in enumerate(THR)
        },
        "windows_with_zero_species": {
            str(t): float((pw[:, i] == 0).mean()) for i, t in enumerate(THR)
        },
        "human_uncalled_frac_windows_gt0": float((np.asarray(human_n) > 0).mean()),
        "n_nonoverlapping_windows": int(sum(l // L for _, l in chroms)),
        "run_structure": runs,
    }
    # the 30k pool the scaling cells used
    seqs, _ = load_pool("zoonomia_orthologs")
    res["pool30k_with_N"] = int(sum("N" in s for s in map(str, seqs)))
    json.dump(res, open(out, "w"), indent=1)
    print(f"windows {done} pairs {pairs:,} in {res['elapsed_s']:.0f}s")
    for t, k, fr in zip(THR, res["pairs_kept"], res["frac_kept"]):
        print(f"  <= {t:2d} N: {k:>10,} pairs ({100 * fr:.2f}%)")
    print("  sampler:", res["sampler"], "| 30k pool sequences containing N:", res["pool30k_with_N"])
    print("  run structure (pairs with 1-20 N):", runs)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pools", "zoonomia"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-windows", type=int, default=6000)
    ap.add_argument("--budget-s", type=float, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.what == "pools":
        an_pools(a.out)
    else:
        an_zoonomia(a.out, a.n_windows, a.budget_s, a.seed)


if __name__ == "__main__":
    main()
