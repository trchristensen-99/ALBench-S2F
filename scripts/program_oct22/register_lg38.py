"""Oct-30: the collaborator's recipe family (a LegNet with weight averaging and batch-norm recalculation, optimised by him at D = 30k) carried to the other two sizes, as the comparison point for our ensembles at 10k and 100k.

  lg38_c20_<D>k         the LegNet-style reference with the study's input standard, 3 seeds (as lg35c_c20_30k)
  lg38_c20_emabn_<D>k   the same with weight averaging (0.9995) and batch-norm recalculation (as lg35c_c20_emabn_30k; the closest we have to his best models)

D = 10k and 100k on the canonical nested genomic pools, V0c validation, patience 16. The 30k cells already exist (lg35c_*).

    python scripts/program_oct22/register_lg38.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_fx35b as fb  # noqa: E402
import register_oct30 as r30  # noqa: E402
import register_pk35 as pk  # noqa: E402

NEST = r30.NEST
DRY = "--dry" in sys.argv
STD = " --flank_ctx 20 --shift_max 10"
CAP = {10000: 15, 100000: 60}


def all_cells():
    out = []
    for D in (10000, 100000):
        pool = f"{NEST}/genomic_d{D}.npz"
        out += fb.cells(
            f"lg38_c20_{D // 1000}k", pk.LG_MENU, pool, D, {}, STD, shards=3, cap=CAP[D], seeds=3
        )
        out += fb.cells(
            f"lg38_c20_emabn_{D // 1000}k",
            pk.LG_MENU,
            pool,
            D,
            pk.EMA,
            STD,
            shards=3,
            cap=CAP[D],
            seeds=3,
        )
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells at the end | registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_lg38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
