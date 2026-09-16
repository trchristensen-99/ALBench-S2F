"""Gate: refuse to proceed if any pool is duplicated, short, or mis-shaped.

Runs between generation and labelling, and again before training. Labelling is the
expensive step and a bad pool poisons every curve point built from it, so the cheap
check goes first. Exits non-zero so a scheduler dependency chain stops here rather
than spending GPU-hours on data that will have to be thrown away.
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glob", default="outputs/curves/pools/*.npz")
    ap.add_argument("--require-labels", action="store_true")
    ap.add_argument("--max-dup-frac", type=float, default=0.0)
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(args.glob)) if "__labeled" in f or not args.require_labels]
    if args.require_labels:
        files = [f for f in sorted(glob.glob(args.glob)) if f.endswith("__labeled.npz")]
    if not files:
        print(f"no pools matched {args.glob}", file=sys.stderr)
        return 1

    bad = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        s = [str(x) for x in z["sequences"]]
        n, u = len(s), len(set(s))
        dup_frac = (n - u) / n if n else 1.0
        lens = {len(x) for x in s[:500]}
        issues = []
        if dup_frac > args.max_dup_frac:
            issues.append(f"{n - u:,} duplicates ({dup_frac:.2%})")
        if len(lens) != 1:
            issues.append(f"mixed lengths {sorted(lens)}")
        if args.require_labels:
            if "oracle_labels" not in z.files:
                issues.append("no oracle_labels")
            else:
                y = np.asarray(z["oracle_labels"], dtype=float)
                if len(y) != n:
                    issues.append(f"{len(y):,} labels for {n:,} sequences")
                elif not np.isfinite(y).all():
                    issues.append(f"{int((~np.isfinite(y)).sum()):,} non-finite labels")
            if "oracle_id" not in z.files:
                issues.append("no oracle_id provenance stamp")
        status = "; ".join(issues) if issues else "ok"
        print(f"  {'FAIL' if issues else 'PASS'}  {f.split('/')[-1]:<52} n={n:>7,} uniq={u:>7,}  {status}")
        if issues:
            bad.append(f)

    print(f"\n{len(files) - len(bad)}/{len(files)} pools pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
