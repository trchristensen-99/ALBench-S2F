"""Read the mixed-reservoir ladder: which sources, in what ratio, at every size.

mx51 asked which reservoirs to mix and mx53 asked in what ratio, but they write into the same
`bench38_ln_{cfg}_{D}k_c{ctx}_{pool}_r{rep}` tree as the single-reservoir ladder, so the parents, the
50/50 anchor and the two brackets can all be read side by side -- which is the only way the ratio
question has an answer. Every number is the stacked ensemble over that cell's seeds, stacked by the
same `enet_all_pos` the rest of the study uses, then averaged over the independent replicate draws
with their spread, because a gap smaller than that spread is not a gap.

    python scripts/analysis/read_mixtures.py [--ctx 20] [--pools mix_ms_mc,motif_shared_core]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_acq52 import SETS, read_run  # noqa: E402

MR = REPO / "outputs/menu_retrain_v0"
# the mixtures first, then the parents they are built from: the ordering is the argument
ORDER = (
    "motif_shared_core",
    "mix_ms2_mc1",
    "mix_ms_mc",
    "mix_ms1_mc2",
    "motif_ct_enriched",
    "mix_ms_gen",
    "mix_ms_mc_gen",
    "genomic",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=20)
    ap.add_argument("--pools", help="comma-separated; default is the mixture set and its parents")
    a = ap.parse_args()
    want = tuple(a.pools.split(",")) if a.pools else ORDER

    # the replicate draws carry `_r{rep}`; the canonical seed-42 pools do not, and the 1M rung exists
    # only as canonical -- the replicate draws stop at 300k -- so a reader that insists on the suffix
    # silently drops the top of the ladder. The recipe is part of a cell's identity: the 1M rung was
    # run at two of them, and keying without it makes one quietly overwrite the other.
    pat = re.compile(rf"^bench38_ln_(.+?)_(\d+)k_c{a.ctx}_(.+?)(?:_r(\d+))?$")
    cells: dict[tuple[str, int, str], dict[int, tuple]] = {}
    for d in sorted(MR.glob(f"bench38_ln_*_c{a.ctx}_*")):
        m = pat.match(d.name)
        if not m:
            continue
        # build_mixture_pools names a canonical mixture `{name}_c_d{D}`, and the training script takes
        # the out_dir from the pool stem, so the canonical draw of a mixture lands under `..._mix_x_c`
        # while a single reservoir's canonical draw keeps its bare name. Fold the tag back in.
        pool = m.group(3)
        rep = int(m.group(4) or 0)
        if pool.startswith("mix_") and pool.endswith("_c"):
            pool = pool[:-2]
        if pool not in want:
            continue
        got = read_run(d)
        if got:
            cells.setdefault((pool, int(m.group(2)), m.group(1)), {})[rep] = got

    sizes = sorted({D for _, D, _ in cells})
    for D in sizes:
        print(f"\n=== D = {D},000 ===")
        print(
            f"{'corpus':20s} {'recipe':>10s} {'reps':>4s} "
            + " ".join(f"{s.replace('ood', 'designed') + ' stack':>22s}" for s in SETS)
        )
        best = {}
        rows = []
        for pool in want:
            for (p, dd, cfg), reps in sorted(cells.items()):
                if p != pool or dd != D:
                    continue
                vals = {s: np.array([stk[s] for _, stk, _ in reps.values()]) for s in SETS}
                rows.append((f"{pool}|{cfg}", reps, vals))
                for s in SETS:
                    best[s] = max(best.get(s, -1.0), float(vals[s].mean()))
        for name, reps, vals in rows:
            pool, cfg = name.split("|")
            cols = []
            for s in SETS:
                mu, sd = float(vals[s].mean()), float(vals[s].std())
                # a mark only where the lead is bigger than the spread it is measured against
                lead = mu >= best[s] - 1e-9
                cols.append(f"{mu:.4f}+-{sd:.4f}{' *' if lead else '  '}")
            print(f"{pool:20s} {cfg:>10s} {len(reps):>4d} " + " ".join(f"{c:>22s}" for c in cols))
    if not cells:
        print("no cells matched; is --ctx right?")


if __name__ == "__main__":
    main()
