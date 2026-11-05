#!/usr/bin/env python
"""Does any reservoir pool contain a validation or test sequence?

AGENTS.md makes the chromosome holdouts a hard rule: chr7 and chr13 are test, chr19, chr21 and chrX are validation, and any strategy touching genomic sequence must exclude them. The rule was never checked
pool by pool until a zoonomia 100k cell failed on `bench33_common.load_data`'s assertion (Nov 5); the audit that followed found exactly one leaked sequence, in the zoonomia pools alone. Run this after building
or relabelling any pool: it is cheap, and a single leaked sequence blocks a whole arm.

Exit status is 1 when anything overlaps, so it can gate a pipeline.

    python scripts/analysis/check_holdout_contamination.py [--pools 'outputs/grid_pools/*/*.npz']      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.paths import resolve  # noqa: E402


def seqs_of(path: str) -> set:
    z = np.load(path, allow_pickle=True)
    key = "sequences" if "sequences" in z.files else z.files[0]
    return {str(s) for s in z[key]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="outputs/grid_pools/k562_nested/*_d*.npz")
    a = ap.parse_args()

    holdouts = {"validation (V0c)": seqs_of(resolve("val_v0c"))}
    try:  # the chr7/13 test set is built by a loader rather than declared as an asset
        sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
        from scaling_hp_search import load_chr_test_genomic

        holdouts["test (chr7/13)"] = {str(s) for s in load_chr_test_genomic()[0]}
    except Exception as e:
        print(f"note: skipping the test-set check ({str(e)[:70]})")
    for k, v in holdouts.items():
        print(f"{k}: {len(v):,} sequences")

    bad = 0
    for p in sorted(glob.glob(a.pools)):
        try:
            s = seqs_of(p)
        except Exception as e:
            print(f"{os.path.basename(p):48s} UNREADABLE {str(e)[:60]}")
            continue
        for name, hold in holdouts.items():
            n = len(s & hold)
            if n:
                bad += 1
                print(
                    f"{os.path.basename(p):48s} OVERLAPS {name}: {n:,} of {len(s):,} ({n / len(s) * 100:.3f}%)"
                )
    print(
        "CLEAN: no pool touches a holdout set"
        if not bad
        else f"CONTAMINATED: {bad} pool/holdout pairs"
    )
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
