"""Put the three scaling curves on one axis: from scratch, from a 30k pilot, from a 300k pilot.

The question is whether *selecting* the next N sequences beats simply *drawing* them, and it can only
be read off curves measured at the same sizes with the same student. All three are LegNet at the per-size
tuned recipe, five seeds, stacked with `enet_all_pos`, in the 240-bp standard context.

Four curves, and the naming rule each one is keyed on (stated, because a tag that silently resolved to
another experiment's directory is what voided al54's first run):

  scratch        motif_shared_core, the canonical seed-42 draw plus the independent `_r{n}` draws.
                 The replicates are what give this curve an error bar rather than seed noise only.
  from30k_gen    al52: a 30k *genomic* pilot, then selection. Start `genomic_s{rep}`, rounds under the
                 bare arm tag at 55k/80k/105k.
  from30k_msc    al54: the same, from a 30k *motif_shared_core* pilot -- the control that separates
                 "good selector" from "bad pilot". Arm tags carry the `a54_` prefix.
  from300k_msc   al53: a 300k motif_shared_core pilot, rounds at 400k/500k/600k.

al52 and al53 use the *same* bare arm tags and are told apart only by size. That is load-bearing, so
SIZES below is asserted to be disjoint across the curves that share a tag namespace: if a later arm is
ever registered at an overlapping size, this reader stops instead of averaging two experiments together.

    python scripts/analysis/read_curves.py [--set designed]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_acq52 import SETS, read_run  # noqa: E402

MR = REPO / "outputs/menu_retrain_v0"
PRETTY = {"genomic": "genomic", "ood": "designed", "random_32k": "random"}

# curve -> (sizes it owns in k, iterative arm tag, pilot reservoir tag)
SIZES = {
    "from30k_gen": ((55, 80, 105), "it3_epi", "genomic"),
    "from30k_msc": ((55, 80, 105), "a54_it3_epi", "motif_shared_core"),
    "from300k_msc": ((400, 500, 600), "it3_epi", "motif_shared_core"),
}
NAME = re.compile(r"^bench38_ln_(.+?)_(\d+)k_c20_(.+)$")


def classify(res: str, dk: int) -> tuple[str, str] | None:
    """Which curve and which replicate a finished directory belongs to, or None to ignore it."""
    if m := re.fullmatch(r"motif_shared_core(?:_r(\d+))?", res):
        return "scratch", f"r{m.group(1)}" if m.group(1) else "canonical"
    # a pilot is round 0 of whichever curve starts at this size
    if m := re.fullmatch(r"(motif_shared_core|genomic)_s(\d+)", res):
        for curve, (sizes, _arm, pilot) in SIZES.items():
            if m.group(1) == pilot and dk == (30 if "30k" in curve else 300):
                return curve, f"s{m.group(2)}"
        return None
    for curve, (sizes, arm, _pilot) in SIZES.items():
        if (m := re.fullmatch(rf"{arm}_s(\d+)", res)) and dk in sizes:
            return curve, f"s{m.group(1)}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, choices=list(PRETTY.values()))
    a = ap.parse_args()

    # the bare-tag collision between al52 and al53 is only safe while their sizes stay disjoint
    bare = [c for c, (_s, arm, _p) in SIZES.items() if not arm.startswith("a54_")]
    for i, c1 in enumerate(bare):
        for c2 in bare[i + 1 :]:
            assert not set(SIZES[c1][0]) & set(SIZES[c2][0]), (
                f"{c1} and {c2} share a tag AND a size"
            )

    cells: dict[tuple[str, int], dict[str, tuple]] = defaultdict(dict)
    for d in sorted(MR.glob("bench38_ln_*_c20_*")):
        m = NAME.match(d.name)
        if not m or not list(d.glob("*_meta.json")):
            continue
        hit = classify(m.group(3), int(m.group(2)))
        if not hit:
            continue
        got = read_run(d)
        if got:
            cells[(hit[0], int(m.group(2)))][hit[1]] = got

    want = [k for k, v in PRETTY.items() if a.set in (None, v)]
    order = ["scratch", "from30k_gen", "from30k_msc", "from300k_msc"]
    for s in want:
        print(f"\n=== stacked test r, {PRETTY[s]} ===")
        print(f"{'size':>7s}  " + "  ".join(f"{c:>20s}" for c in order))
        for dk in sorted({d for _c, d in cells}):
            row = [f"{dk * 1000:>7,}"]
            for c in order:
                got = cells.get((c, dk))
                if not got:
                    row.append(f"{'-':>20s}")
                    continue
                v = [stack[s] for _single, stack, _n in got.values()]
                spread = (max(v) - min(v)) / 2 if len(v) > 1 else 0.0
                mark = "" if len(v) > 1 else "!"  # one draw: no band
                row.append(f"{np.mean(v):>13.4f}+-{spread:.4f}{mark:1s}".rjust(20))
            print("  ".join(row))
    print("\n`!` = a single draw, so the number carries seed noise only and no band.")
    print("Draws per cell:")
    for c in order:
        got = {dk: sorted(cells[(c, dk)]) for _c, dk in cells if _c == c}
        if got:
            print(f"  {c:14s} " + "  ".join(f"{k}k:{','.join(v)}" for k, v in sorted(got.items())))


if __name__ == "__main__":
    main()
