"""Validate every planned curve command BEFORE the scheduler runs it.

The chain's gates check the pools. This checks the 284 commands built from them: that
each names a reservoir whose linked pool will exist, a baseline file that exists, and a
training size the pool can actually supply. A single malformed command costs one array
task; a systematically wrong one costs the night.

Deliberately offline -- it opens npz headers and inspects paths, loads no model and
touches no GPU, so it can run on a login node in seconds.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
POOLS = REPO / "outputs" / "curves" / "pools"


def linked_name(pool_file: Path) -> str:
    """The reservoir name the linker will expose this pool under."""
    return re.sub(r"__n\d+__seed\d+$", "", pool_file.name.replace("__labeled.npz", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", default="outputs/curves/jobs.txt")
    args = ap.parse_args()

    jobs_file = REPO / args.jobs
    if not jobs_file.exists():
        print(f"no plan at {jobs_file}; run build_curve_plan.py --stage train", file=sys.stderr)
        return 1
    cmds = [c for c in jobs_file.read_text().splitlines() if c.strip()]

    # What the linker WILL expose, derived from the labelled pools on disk.
    labelled = sorted(POOLS.glob("*__labeled.npz"))
    available = {linked_name(f): f for f in labelled}
    capacity = {}
    for name, f in available.items():
        with np.load(f, allow_pickle=True) as z:
            capacity[name] = len(z["sequences"])

    print(f"labelled pools -> linked names ({len(available)}):")
    for n in sorted(available):
        print(f"  {n:<24} capacity {capacity[n]:>8,}")

    errs: list[str] = []
    seen_res: set[str] = set()
    for i, c in enumerate(cmds, 1):
        res = re.search(r"--reservoir (\S+)", c)
        size = re.search(r"--training-sizes (\d+)", c)
        bp = re.search(r"--base-pool (\S+)", c)
        bn = re.search(r"--base-n (\d+)", c)
        out = re.search(r"--output-dir (\S+)", c)
        if not (res and size and out):
            errs.append(f"line {i}: malformed command")
            continue
        r, n = res.group(1), int(size.group(1))
        seen_res.add(r)
        if r not in available:
            errs.append(f"line {i}: reservoir {r!r} has no labelled pool yet")
        elif n > capacity[r]:
            errs.append(f"line {i}: {r} asked for {n:,} but pool holds {capacity[r]:,}")
        if bp:
            p = REPO / bp.group(1)
            if not p.exists():
                errs.append(f"line {i}: base pool missing: {bp.group(1)}")
            elif bn:
                with np.load(p, allow_pickle=True) as z:
                    have = len(z["sequences"])
                if int(bn.group(1)) > have:
                    errs.append(f"line {i}: base-n {int(bn.group(1)):,} > baseline {have:,}")

    missing = sorted(seen_res - set(available))
    if missing:
        print(f"\nreservoirs still awaiting labels (fine if labelling is queued): {missing}")

    print(f"\nchecked {len(cmds)} planned commands")
    if errs:
        print(f"{len(errs)} PROBLEM(S):")
        for e in errs[:25]:
            print(f"  {e}")
        # Unlabelled-yet reservoirs are expected mid-chain; only other errors are fatal.
        fatal = [e for e in errs if "has no labelled pool yet" not in e]
        return 1 if fatal else 0
    print("all commands reference existing pools and feasible sizes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
