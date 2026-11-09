"""Cut extra rungs out of the nested pools, so every curve can be sampled at the same sizes.

The nested pools exist at 10k / 30k / 100k / 300k / 1M, but the acquisition experiments land on 55k, 80k
and 105k (al52, al54) and on 400k, 500k and 600k (al53), so a from-scratch curve read off the existing
ladder has to be interpolated to be compared with them -- which is exactly the kind of comparison that
quietly turns a measurement into a drawing. Because the pools are nested, the honest version is free: a
600k prefix of the 1M pool IS 600,000 sequences of that strategy drawn the same way, so the rung can be
cut rather than generated.

Every stamp on the source is carried over and `sliced_from` records which file the rows came from, so a
pool's provenance stays readable from the pool itself.

    python scripts/slice_nested_prefix.py --strategy motif_shared_core --sizes 55000,80000,105000 [--dry]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NEST = REPO / "outputs/grid_pools/k562_nested"


def source_for(strategy: str, d: int) -> Path:
    """The smallest existing pool that still contains d rows: the least truncation, and the same draw."""
    have = sorted(
        (int(p.stem.split("_d")[-1]), p)
        for p in NEST.glob(f"{strategy}_d*.npz")
        if p.stem.split("_d")[-1].isdigit()
    )
    for n, p in have:
        if n >= d:
            return p
    raise SystemExit(
        f"no {strategy} pool holds {d:,} rows (largest is {have[-1][0]:,})"
        if have
        else f"no {strategy} pools"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--sizes", required=True, help="comma-separated")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    for d in (int(x) for x in a.sizes.split(",")):
        dest = NEST / f"{a.strategy}_d{d}.npz"
        if dest.exists():
            print(f"  {dest.name}: exists, left alone")
            continue
        src = source_for(a.strategy, d)
        z = np.load(src, allow_pickle=True)
        n = len(z["sequences"])
        out = {}
        for k in z.files:
            v = z[k]
            # the per-sequence arrays get cut; the scalar stamps (oracle_id, strategy, seed) are carried
            out[k] = v[:d] if (getattr(v, "ndim", 0) >= 1 and v.shape[0] == n) else v
        out["sliced_from"] = src.name
        uniq = len(set(str(s) for s in out["sequences"]))
        print(
            f"  {dest.name}: {len(out['sequences']):,} rows from {src.name} ({n:,}), "
            f"{uniq:,} distinct, label {out['oracle_labels'].mean():+.3f}"
            + ("  [DRY]" if a.dry else "")
        )
        assert len(out["sequences"]) == d, "prefix is the wrong length"
        if a.dry:
            continue
        tmp = dest.with_suffix(".tmpnew.npz")
        np.savez(tmp, **out)
        tmp.replace(dest)


if __name__ == "__main__":
    main()
