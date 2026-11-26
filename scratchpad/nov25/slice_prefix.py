"""Write the first N rows of a pool as its own file, keeping every stamp.

`bench33_legnet.py --D N --pool P` does not truncate P: it asserts ``len(P) == N`` and
dies otherwise (``AssertionError: (320000, 300000)``). Every pool in `k562_nested` is
already sized to its name, so the convention is invisible until you build a pool of your
own -- a 320,000-row combination corpus cannot be trained at 30k, 100k or 300k without
three separate files, and a from-scratch control cannot reuse the nearest larger pool.

Prefixes, not samples: the pools are nested, so the first N rows ARE the size-N corpus and
slicing keeps that property. Taking a random N would break it and make the size axis vary
the composition as well as the amount.

    python scratchpad/nov25/slice_prefix.py --in pool.npz --n 30000 --out pool_d30000.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        try:
            with np.load(a.out, allow_pickle=True) as z:
                if len(z["sequences"]) == a.n:
                    print(f"SKIP already present: {a.out}")
                    return 0
        except Exception:
            pass

    z = np.load(a.src, allow_pickle=True)
    n_src = len(z["sequences"])
    if n_src < a.n:
        raise SystemExit(f"{a.src} holds {n_src:,} rows, fewer than the {a.n:,} requested")
    # Every per-row column is sliced; every scalar stamp is carried through untouched, so
    # the slice still says which oracle labelled it and which strategy produced it.
    out = {}
    for k in z.files:
        v = z[k]
        out[k] = v[: a.n] if getattr(v, "ndim", 0) == 1 and len(v) == n_src else v
    out["sliced_from"] = np.array(os.path.basename(a.src))
    uniq = len(set(map(str, out["sequences"])))
    if uniq < a.n:
        raise SystemExit(
            f"the {a.n:,}-row prefix holds only {uniq:,} distinct sequences; "
            "training on it would silently weight the repeats"
        )
    tmp = f"{a.out}.tmp.{os.getpid()}"
    np.savez_compressed(tmp, **out)
    os.replace(tmp if os.path.exists(tmp) else tmp + ".npz", a.out)
    print(f"wrote {a.out}  ({a.n:,} rows of {n_src:,}, all distinct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
