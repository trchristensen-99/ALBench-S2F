"""Why does random DNA score higher than Gosai genomic under our K562 oracle?

Generates random 200bp at fixed GC from 10% to 90%, labels with the SAME oracle_v3
that labels every pool in the benchmark, and bins the genomic pool by each window's
own GC. Output feeds a ridgeline + median-vs-GC figure matching the collaborator's
Malinois reference, but for OUR oracle (the one that actually drives the scaling
curves). Isolates how much of the random>genomic gap is pure GC composition.
"""
from __future__ import annotations
import sys
import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
L = 200
N_PER_BIN = 10_000
GC_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BASES = np.array(list("ACGT"))


def random_at_gc(gc: float, n: int, rng) -> list[str]:
    # per-base: G or C with prob gc (split evenly), A or T with prob 1-gc.
    # vectorised: draw all bases at once, expected GC == gc exactly in the limit.
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A,C,G,T
    idx = rng.choice(4, size=(n, L), p=p)
    return ["".join(BASES[row]) for row in idx]


def gc_of(seqs):
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])


def main():
    from scripts.build_pool import _pin_ag_s2_oracle
    from scripts.generate_labeled_pools import _label_sequences
    from experiments.exp1_1_scaling import _load_oracle

    _pin_ag_s2_oracle("oracle_v3")
    oracle = _load_oracle("k562", oracle_type="ag_s2")
    rng = np.random.default_rng(42)

    out = {"gc_levels": np.array(GC_LEVELS)}
    for gc in GC_LEVELS:
        seqs = random_at_gc(gc, N_PER_BIN, rng)
        lab = np.asarray(_label_sequences(oracle, seqs), dtype=np.float32).ravel()
        out[f"random_gc{int(gc*100)}"] = lab
        print(f"  random GC={gc:.1f}: n={len(lab)} actual_gc={gc_of(seqs).mean():.3f} "
              f"pred mean={lab.mean():.3f} median={np.median(lab):.3f}", flush=True)

    z = np.load(f"{REPO}/outputs/grid_pools/k562_cumulative/genomic_d100000.npz",
                allow_pickle=True)
    gseq = [str(x) for x in z["sequences"]]
    out["genomic_gc"] = gc_of(gseq)
    out["genomic_pred"] = np.asarray(z["oracle_labels"], dtype=np.float32).ravel()
    print(f"  genomic: n={len(gseq)} mean_gc={out['genomic_gc'].mean():.3f} "
          f"pred mean={out['genomic_pred'].mean():.3f}", flush=True)

    np.savez_compressed(f"{REPO}/outputs/gc_bias/gc_bias_oracle_v3_k562.npz", **out)
    print("  wrote outputs/gc_bias/gc_bias_oracle_v3_k562.npz", flush=True)


if __name__ == "__main__":
    main()
