"""Nov-1: the published LegNet recipe turned out to be under-trained (lr38_ sweeps: 150 to 800 epochs and weight decay 0.2 to 0.4 gave up to +0.066 genomic r at 10k, +0.024 at 30k, +0.006 at 100k). The menu members were
trained for 60 to 80 epochs with patience-16 early stopping, so they may be under-trained too, which would change every menu-versus-LegNet comparison. This runs menu v2 (20 configurations, one seed, canonical genomic
pools) with every epoch budget tripled and early stopping effectively off, against the same menu with its own recipe (v2m38_genomic_<D>k), paired by entry.

  ep38_v2x3_genomic_<D>k   D = 10k, 30k   (--epoch_scale 3, AL_PATIENCE 9999; the per-model wall cap is raised so that the longer schedules are not truncated)

    python scripts/program_oct22/register_ep38.py [--dry]
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

R, NEST = r30.R, r30.NEST
DRY = "--dry" in sys.argv
V2 = f"{R}/outputs/menus_oct30/V2.json"
STD = " --flank_ctx 20 --shift_max 10 --epoch_scale 3"
ENV = {"AL_PATIENCE": "9999"}
SIZES = {10000: (4, 90), 30000: (5, 180)}  # shards, per-model cap in H100-equivalent minutes


def all_cells():
    out = []
    for D, (shards, cap) in SIZES.items():
        pool = f"{NEST}/genomic_d{D}.npz"
        out += fb.cells(f"ep38_v2x3_genomic_{D // 1000}k", V2, pool, D, ENV, STD, shards, cap)
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
            print(new[0]["cmd"][:420] if new else "nothing new")
            return
        shutil.copy(path, path + ".bak_ep38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
