"""How much sequence survives at each N (uncalled) threshold?

Relaxing min_called_frac admits more species per window and so enlarges the pool,
but each admitted sequence carries more uncalled bases. This quantifies that
trade so the threshold is a measured choice rather than a guess, and shows how
much real sequence the delete-and-extend construction has to manufacture back
from the flanks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from albench.plotting import apply_style

N_WINDOWS = 4000
L = 200


def main():
    import h5py
    from albench.paths import resolve
    from albench.reservoir.zoonomia_orthologs import CODE_TO_BASE

    rng = np.random.default_rng(0)
    fracs = []           # called fraction, per (window, species)
    with h5py.File(str(resolve("zoonomia_alignment")), "r") as f:
        chroms = [c for c in f if c.startswith("chr") and c not in ("chr7", "chr13")]
        for _ in range(N_WINDOWS):
            c = chroms[rng.integers(len(chroms))]
            n = f[c]["seq"].shape[1]
            st = int(rng.integers(L, n - 2 * L))
            blk = f[c]["seq"][:, st : st + L]
            fracs.append((blk != 0).mean(axis=1))
    F = np.concatenate(fracs)
    pctN = 100 * (1 - F)

    print(f"  sampled {N_WINDOWS:,} windows x {len(fracs[0])} species = {F.size:,} pairs")
    print(f"\n  {'min_called_frac':>16} {'kept':>12} {'% of pairs':>11} {'mean %N kept':>13}")
    for thr in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00):
        keep = F >= thr
        mn = pctN[keep].mean() if keep.any() else float("nan")
        print(f"  {thr:>16.2f} {keep.sum():>12,} {100*keep.mean():>10.1f}% {mn:>12.2f}%")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.hist(pctN, bins=np.linspace(0, 100, 101), color="#4c72b0")
    ax.set_yscale("log")
    ax.set_xlabel("% uncalled (N) in a 200bp window")
    ax.set_ylabel("count of (window, species) pairs")
    ax.set_title("Density of uncalled fraction")
    ax = axes[1]
    thrs = np.linspace(0, 1, 101)
    kept = [(F >= t).sum() for t in thrs]
    ax.plot(100 * thrs, kept, lw=2, color="#55a868")
    for t in (0.60, 0.80, 0.95):
        ax.axvline(100 * t, ls="--", lw=1, color="#888")
        ax.annotate(f"{t:.2f}\n{(F>=t).sum():,}", (100 * t, (F >= t).sum()),
                    textcoords="offset points", xytext=(5, 8), fontsize=9)
    ax.set_xlabel("min_called_frac threshold (%)")
    ax.set_ylabel("sequences retained")
    ax.set_title("Pool size vs threshold")
    fig.suptitle("Zoonomia: what each N threshold costs", y=1.02)
    fig.tight_layout()
    out = REPO / "outputs/figures/zoonomia_n_thresholds.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
