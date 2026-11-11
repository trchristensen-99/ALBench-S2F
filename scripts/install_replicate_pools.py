"""Install freshly built 300k replicate draws as nested per-size pools.

`slice_replicate_pools.py` does this for replicates 1-3, but it is written around the three
synthesised reservoirs it knew about and around raw pools in `k562_reps_raw/`. The later
replicates are built by `register_nov11e.py` into `k562_reps_raw300k_new/{strategy}__g{seed}/`,
one fresh 300k draw per (strategy, replicate), and this turns each into the four prefix files the
scaling and acquisition code reads.

It is deliberately simpler than `extend_replicate_pools_300k.py`: that script exists because
replicates 1-3 already had 10k/30k/100k pools on disk that a fresh 300k draw would not have been a
superset of, so it had to append rather than replace. A replicate with nothing on disk has no such
constraint, and slicing prefixes of one draw makes nesting true by construction. The script refuses
to run where that assumption does not hold, rather than silently overwriting a pool that smaller
runs already used.

    python scripts/install_replicate_pools.py --reps 4 5 6 [--dry]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GP = REPO / "outputs/grid_pools"
RAW = GP / "k562_reps_raw300k_new"
OUT = GP / "k562_reps"
SIZES = (10_000, 30_000, 100_000, 300_000)
SEED0 = {1: 43, 2: 44, 3: 45, 4: 46, 5: 47, 6: 48}


def load(d: Path) -> tuple[list[str], np.ndarray, str]:
    files = [f for f in glob.glob(str(d / "*.npz")) if not f.endswith("_MANIFEST.npz")]
    assert len(files) == 1, f"{d}: expected one .npz, found {len(files)}"
    z = np.load(files[0], allow_pickle=True)
    return (
        [str(s) for s in z["sequences"]],
        z["oracle_labels"].astype(np.float32),
        str(z["oracle_id"]),
    )


def save(path: Path, seqs, labels, oracle_id, strategy, seed) -> None:
    tmp = str(path).replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(
        tmp,
        sequences=np.asarray(seqs, dtype=object),
        oracle_labels=np.asarray(labels, np.float32),
        oracle_id=oracle_id,
        strategy=strategy,
        seed=np.int64(seed),
    )
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, nargs="+", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    report: dict = {}
    for rep in a.reps:
        g = SEED0[rep]
        for d in sorted(RAW.glob(f"*__g{g}")):
            strat = d.name.split("__g")[0]
            if not (d / "COMPLETE").exists():
                print(f"  {strat} r{rep}: not finished, skipped")
                continue

            # a prefix slice is only safe where nothing smaller is already on disk; if a size
            # exists, some run has used it and rewriting it would change that run's data
            clash = [n for n in SIZES if (OUT / f"{strat}_r{rep}_d{n}.npz").exists()]
            if clash:
                print(f"  {strat} r{rep}: {clash} already exist, refusing to overwrite")
                continue

            seqs, labels, oid = load(d)
            if len(seqs) < SIZES[-1]:
                print(f"  {strat} r{rep}: only {len(seqs):,} sequences, below {SIZES[-1]:,}")
            uniq = len(set(seqs))
            for n in SIZES:
                if len(seqs) < n:
                    continue
                p = OUT / f"{strat}_r{rep}_d{n}.npz"
                if not a.dry:
                    save(p, seqs[:n], labels[:n], oid, strat, g)
                print(f"  wrote {p.name}  ({n:,} of {len(seqs):,}, {uniq:,} distinct)")
            report[f"{strat}_r{rep}"] = {
                "generation_seed": g,
                "oracle_id": oid,
                "n": len(seqs),
                "n_distinct": uniq,
                "duplicate_fraction": round(1 - uniq / max(len(seqs), 1), 4),
                "sizes": [n for n in SIZES if len(seqs) >= n],
            }

    if report and not a.dry:
        mp = OUT / f"MANIFEST_reps_{'_'.join(str(r) for r in a.reps)}.json"
        json.dump(report, open(mp, "w"), indent=1)
        print(f"\nwrote {mp}")
    print(f"{len(report)} (strategy, replicate) pairs installed")


if __name__ == "__main__":
    main()
