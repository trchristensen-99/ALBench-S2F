"""Localization control: are ISM hot windows ENRICHED for Sp1/GC-box k-mers over
RANDOM windows from the same sequences? If hot >> random, the model specifically
lights up GC-box motifs; if hot ~= random, it reads GC diffusely.

Pure numpy; reuses the saved ISM profiles and regenerates the exact same random
sequences (rand_at_gc draws are the first draws of default_rng(0), so reproducible).
"""
from __future__ import annotations

import json
import sys

import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
from data.k562 import K562Dataset  # noqa: E402

L = 200
N = 500
BASES = np.array(list("ACGT"))
SP1_CORES = ["GGGCGG", "CCGCCC", "GGGGCGGGG", "GGGCGGGG", "GCGGGGCGG"]


def rand_at_gc(gc, n, rng):
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return ["".join(BASES[r]) for r in rng.choice(4, size=(n, L), p=p)]


def windows_at(seqs, centers, w=9):
    half = w // 2
    out = []
    for s, cs in zip(seqs, centers):
        for c in cs:
            a, b = max(0, c - half), min(L, c + half + 1)
            out.append(s[a:b])
    return out


def top_centers(prof, n_peaks=2, w=9):
    half = w // 2  # noqa: F841
    res = []
    for p in prof:
        taken = np.zeros(L, dtype=bool)
        cs = []
        for _ in range(n_peaks):
            c = int(np.argmax(np.where(taken, -np.inf, p)))
            if p[c] <= 0:
                break
            cs.append(c)
            taken[max(0, c - w) : min(L, c + w + 1)] = True
        res.append(cs)
    return res


def frac_with_core(wins, cores=SP1_CORES):
    if not wins:
        return 0.0
    return sum(any(cc in w for cc in cores) for w in wins) / len(wins)


def ggc_count_per_win(wins):
    if not wins:
        return 0.0
    return np.mean([w.count("GGGCGG") + w.count("CCGCCC") for w in wins])


def cpg_per_bp(wins):
    tot = sum(w.count("CG") for w in wins)
    bp = sum(max(0, len(w) - 1) for w in wins)
    return tot / max(1, bp)


def main():
    d = np.load(f"{REPO}/outputs/gc_bias/ism_highgc_vs_genomic.npz", allow_pickle=True)
    # reproduce exact sequences (same draw order as main script)
    rng = np.random.default_rng(0)
    hi = rand_at_gc(0.80, N, rng)
    mid = rand_at_gc(0.50, N, rng)
    ds = K562Dataset(data_path=f"{REPO}/data/k562", split="test")
    gi = rng.choice(len(ds.sequences), size=N, replace=False)
    gen = [ds.sequences[i] for i in gi]
    groups = {"highgc": hi, "midgc": mid, "genomic": gen}

    rng2 = np.random.default_rng(7)
    report = {}
    for name, seqs in groups.items():
        prof = d[f"{name}_prof"]
        hot_c = top_centers(prof, n_peaks=2)
        rnd_c = [rng2.integers(4, L - 4, size=len(c)).tolist() for c in hot_c]
        hot = windows_at(seqs, hot_c)
        rnd = windows_at(seqs, rnd_c)
        report[name] = {
            "sp1_core_hot": round(frac_with_core(hot), 4),
            "sp1_core_rand": round(frac_with_core(rnd), 4),
            "sp1_enrichment": round(frac_with_core(hot) / max(1e-6, frac_with_core(rnd)), 2),
            "ggc_ccc_per_win_hot": round(float(ggc_count_per_win(hot)), 4),
            "ggc_ccc_per_win_rand": round(float(ggc_count_per_win(rnd)), 4),
            "cpg_hot": round(cpg_per_bp(hot), 4),
            "cpg_rand": round(cpg_per_bp(rnd), 4),
        }
        r = report[name]
        print(
            f"{name:8s} Sp1-core hot={r['sp1_core_hot']:.3f} rand={r['sp1_core_rand']:.3f} "
            f"enrich={r['sp1_enrichment']:.2f}x | GGGCGG|CCGCCC/win hot={r['ggc_ccc_per_win_hot']:.3f} "
            f"rand={r['ggc_ccc_per_win_rand']:.3f} | CpG/bp hot={r['cpg_hot']:.3f} rand={r['cpg_rand']:.3f}",
            flush=True,
        )

    with open(f"{REPO}/outputs/gc_bias/ism_localization_control.json", "w") as f:
        json.dump(report, f, indent=2)
    print("wrote ism_localization_control.json", flush=True)


if __name__ == "__main__":
    main()
