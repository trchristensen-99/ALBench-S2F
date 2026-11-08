"""Check the GPU k-means++ used by `acq_ladder.py` against the library's exact numpy one.

The two cannot agree pick-for-pick: both sample a point with probability proportional to its squared distance
from the nearest chosen centre, but one draws from numpy's generator and the other from torch's, so they
follow different paths through the same distribution. Pick-for-pick agreement is therefore the wrong test.
What has to hold is that the GPU version is the same ALGORITHM -- that its batches are as spread out as the
library's, and far more spread out than a random batch of the same size, which is the property BADGE exists
to provide and the one a silent bug would destroy.

Reported per method: the k-means objective (mean squared distance from each candidate to its nearest selected
point; lower is better-covered), the mean nearest-neighbour distance WITHIN the batch (higher is more
diverse), and how long it took.

    python scripts/verify_kmeanspp.py [--n 20000] [--k 2000] [--d 256]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/program_oct22"))


def stats(X: np.ndarray, sel: np.ndarray) -> tuple[float, float]:
    P = X[sel]
    d2 = ((X[:, None, :] - P[None, :, :]) ** 2).sum(-1) if len(P) * len(X) < 4e7 else None
    if d2 is None:  # chunk, so a big case does not try to build an n x k matrix at once
        near = np.empty(len(X))
        for i in range(0, len(X), 2000):
            near[i : i + 2000] = ((X[i : i + 2000, None, :] - P[None]) ** 2).sum(-1).min(1)
    else:
        near = d2.min(1)
    inner = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(inner, np.inf)
    return float(near.mean()), float(np.sqrt(inner.min(1)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    # clustered, like a real embedding: uniform noise would make every selector look the same
    centres = rng.normal(size=(40, a.d)) * 3
    X = (centres[rng.integers(40, size=a.n)] + rng.normal(size=(a.n, a.d))).astype(np.float32)
    X *= rng.gamma(2.0, 0.5, size=(a.n, 1)).astype(np.float32)  # the uncertainty weighting

    from acq_ladder import kmeanspp_torch

    from albench.acquisition.badge import kmeanspp

    rows = []
    t0 = time.time()
    sel = rng.choice(a.n, size=a.k, replace=False)
    rows.append(("random", sel, time.time() - t0))

    t0 = time.time()
    sel = kmeanspp(X.astype(float), a.k, np.random.default_rng(a.seed))
    rows.append(("library (numpy, exact)", sel, time.time() - t0))

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    sel = kmeanspp_torch(X, a.k, a.seed, device=dev)
    rows.append((f"acq_ladder ({dev}, torch)", sel, time.time() - t0))

    print(f"n={a.n:,}  k={a.k:,}  d={a.d}\n")
    print(
        f"{'method':28s} {'coverage (lower=better)':>24s} {'in-batch NN dist':>18s} {'seconds':>9s}"
    )
    ref = None
    for name, sel, secs in rows:
        cov, nn = stats(X, np.asarray(sel))
        if name.startswith("library"):
            ref = (cov, nn)
        print(f"{name:28s} {cov:24.4f} {nn:18.4f} {secs:9.1f}")
    cov, nn = stats(X, np.asarray(rows[-1][1]))
    assert ref is not None
    print(
        f"\ngpu vs library: coverage {100 * (cov / ref[0] - 1):+.1f}%, "
        f"in-batch spacing {100 * (nn / ref[1] - 1):+.1f}%"
    )
    print(f"overlap of the two selected sets: {len(set(rows[1][1]) & set(rows[2][1])) / a.k:.1%}")


if __name__ == "__main__":
    main()
