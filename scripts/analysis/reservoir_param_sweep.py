"""In-silico sweep over reservoir strategies AND their parameters.

The design treats perturbation rates, motif counts, backgrounds and evolution depth as IN-SILICO
DECIDABLE: rather than spending experimental arms enumerating them, screen them against the oracle
and order ONE tuned configuration per family. This is the screen. It is also the pruning instrument
Carl asked for -- generate everything, pool it, and see what the acquisition functions actually pick.

Scored WITHOUT training anything, so it is cheap:
  activity     oracle-predicted mean/sd/range, and coverage of the activity dynamic range
  diversity    mean pairwise k-mer cosine distance (lower = more redundant pool)
  novelty      max k-mer similarity to a genomic reference sample (high = near-duplicate of training)
  duplicates   exact-duplicate fraction, which wasted ~26% of oracle compute in an earlier pass

IMPORTANT SCORING CAVEAT: judge these on ABSOLUTE activity only. The oracle's out-of-fold
variant-effect correlation is 0.289 against an assay ceiling of 0.579, so any parameter chosen on a
delta metric would be tuned largely to noise.

MOTIF CAVEAT: motif-family sweeps are bounded by the hardcoded 9-consensus vocabulary in
albench/reservoir/motif_planted_v2.py (one entry, 'CTCFCC', is not even valid DNA). Motif-parameter
results will not transfer until a real PWM database is wired in.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

# strategy -> {param: [values]}. Names/defaults verified against the sampler constructors.
GRIDS = {
    "partial_mutagenesis": {
        "mutation_rate_distribution": ["fixed", "uniform"],
        "mutation_rate": [0.01, 0.03, 0.05, 0.10, 0.20],
    },
    "evoaug_structural": {"p_deletion": [0.1, 0.3, 0.5]},
    "motif_planted_v2": {"min_motifs": [1, 3, 5], "motif_set": ["auto"]},
    "motif_density": {"n_motifs": [1, 3, 5, 8]},
    "motif_grammar": {"min_motifs": [1, 3]},
    "phylogenetic_zoonomia": {"mut_rate": [0.005, 0.02, 0.05, 0.10]},
    "in_silico_evolution_generative": {"n_evolution_rounds": [1, 3, 5, 10]},
    "activity_stratified": {"n_bins": [5, 10, 20]},
    "gc_matched": {"n_gc_bins": [10, 50]},
    "recombination": {"crossover_mode": ["uniform", "single_point"]},
    "random": {},
    "dinuc_shuffle": {},
}


def kmer_matrix(seqs, k=6, max_n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(seqs) > max_n:
        seqs = [seqs[i] for i in rng.choice(len(seqs), max_n, replace=False)]
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    M = np.zeros((len(seqs), 4**k), dtype=np.float32)
    for r, s in enumerate(seqs):
        s = str(s).upper()
        h = 0
        valid = 0
        for c in s:
            v = idx.get(c)
            if v is None:
                h, valid = 0, 0
                continue
            h = (h * 4 + v) % (4**k)
            valid += 1
            if valid >= k:
                M[r, h] += 1
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-8)


def diversity_stats(seqs, ref=None):
    M = kmer_matrix(seqs)
    sim = M @ M.T
    np.fill_diagonal(sim, np.nan)
    out = {
        "mean_pairwise_dist": float(1.0 - np.nanmean(sim)),
        "dup_frac": float(1.0 - len(set(map(str, seqs))) / max(1, len(seqs))),
    }
    if ref is not None and len(ref):
        R = kmer_matrix(list(ref))
        out["max_sim_to_genomic"] = float(np.mean((M @ R.T).max(axis=1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seqs", type=int, default=2000)
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/param_sweep/sweep.json")
    ap.add_argument("--score", action="store_true", help="also score with the oracle (GPU)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from albench.reservoir import get_sampler  # registry lookup

    names = args.strategies or list(GRIDS)
    ref = None
    try:
        z = np.load("outputs/chr_split_cache/chr_train_ref_only.npz", allow_pickle=True)
        ref = [str(s) for s in z["sequences"][:2000]]
    except Exception as e:
        print(f"[warn] no genomic reference for novelty ({e})")

    oracle = None
    if args.score:
        from albench.oracle import load_oracle

        oracle = load_oracle()

    rows = []
    for name in names:
        grid = GRIDS.get(name, {})
        keys = list(grid)
        combos = list(itertools.product(*(grid[k] for k in keys))) or [()]
        for combo in combos:
            kw = dict(zip(keys, combo))
            t0 = time.time()
            try:
                sampler = get_sampler(name, seed=args.seed, **kw)
                seqs, _meta = sampler.generate(args.n_seqs)
            except Exception as e:
                print(f"  {name:<30} {kw} -> FAILED: {type(e).__name__}: {e}")
                rows.append({"strategy": name, "params": kw, "error": f"{type(e).__name__}: {e}"})
                continue
            rec = {
                "strategy": name,
                "params": kw,
                "n": len(seqs),
                "gen_sec": round(time.time() - t0, 1),
            }
            rec.update(diversity_stats(seqs, ref))
            if oracle is not None:
                p = np.asarray(oracle.predict(seqs), dtype=float)
                rec.update(
                    pred_mean=float(p.mean()),
                    pred_sd=float(p.std()),
                    pred_p5=float(np.percentile(p, 5)),
                    pred_p95=float(np.percentile(p, 95)),
                )
            rows.append(rec)
            extra = f" act={rec.get('pred_mean', float('nan')):.2f}" if oracle else ""
            print(
                f"  {name:<30} {str(kw):<52} div={rec['mean_pairwise_dist']:.3f} "
                f"dup={rec['dup_frac']:.3f}{extra}"
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    ok = [r for r in rows if "error" not in r]
    print(f"\n[sweep] {len(ok)}/{len(rows)} configs succeeded -> {args.out}")
    if ok:
        best = max(ok, key=lambda r: r["mean_pairwise_dist"])
        print(
            f"[sweep] most diverse: {best['strategy']} {best['params']} "
            f"({best['mean_pairwise_dist']:.3f})"
        )


if __name__ == "__main__":
    main()
