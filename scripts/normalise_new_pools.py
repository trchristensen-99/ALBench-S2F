#!/usr/bin/env python
"""Put the Nov-4/5 genome-sourced pools into the nested naming the rest of the study uses, so bank cells can address them by size.

The accessibility and reference-genome pools were written at whatever size their source allowed (K562 accessible tiles 60,689; the three ENCODE partitions 53,839 / 28,330 / 116,817; hg38 random 100,000),
but every trainer takes `--D` and asserts the pool holds exactly D sequences. Each pool is already in a fixed order, so a smaller size is its prefix, which also keeps a size comparison nested.

    python scripts/normalise_new_pools.py [--dry]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
NEST = f"{ROOT}/outputs/grid_pools/k562_nested"
UNLAB = f"{ROOT}/outputs/grid_pools/k562_unlabelled"
SIZES = (10000, 30000, 100000)
SOURCES = {
    "accessible_tiles_k562": f"{NEST}/accessible_tiles_k562_d100000.npz",
    "accessible_k562_only": f"{UNLAB}/labelled_accessible_k562_only.npz",
    "accessible_shared": f"{UNLAB}/labelled_accessible_shared_open_both.npz",
    "accessible_hepg2_only": f"{UNLAB}/labelled_accessible_hepg2_only.npz",
    "hg38_random": f"{NEST}/hg38_random_d100000.npz",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for name, src in SOURCES.items():
        if not os.path.exists(src):
            print(f"{name:26s} SOURCE MISSING {src}")
            continue
        z = np.load(src, allow_pickle=True)
        have = len(z["sequences"])
        made = []
        for n in SIZES:
            if have < n:
                continue
            dst = f"{NEST}/{name}_d{n}.npz"
            if os.path.exists(dst):
                made.append(f"d{n // 1000}k(exists)")
                continue
            if not a.dry:
                out = {
                    k: (z[k][:n] if getattr(z[k], "shape", ())[:1] == (have,) else z[k])
                    for k in z.files
                }
                np.savez_compressed(dst, **out)
            made.append(f"d{n // 1000}k")
        print(
            f"{name:26s} {have:7,} sequences -> {', '.join(made) if made else 'TOO SMALL for 10k'}"
        )


if __name__ == "__main__":
    main()
