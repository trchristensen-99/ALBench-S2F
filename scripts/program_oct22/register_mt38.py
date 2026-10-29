"""Oct-29 (evening): does an auxiliary HepG2 target help the K562 student? (cells mt38_)

Replicate pool 1 of the genomic reservoir at 30k (K562 and HepG2 oracle_v3-family labels of the same 30,000 sequences, correlation 0.85), menu v1 (the 15 configurations that take wider input), 20 real flank bases
and a +-10 shift, V0c validation, patience 16. The auxiliary head is attached to the input of each architecture's final readout and used only in training; every prediction is the K562 output.

  mt38_k562only_r1_30k   control (same pool and settings, no auxiliary target)
  mt38_aux05_r1_30k      + HepG2 target, weight 0.5
  mt38_aux10_r1_30k      + HepG2 target, weight 1.0

    python scripts/program_oct22/register_mt38.py [--dry]
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

R, V1 = r30.R, r30.V1
DRY = "--dry" in sys.argv
POOL = f"{R}/outputs/grid_pools/k562_reps/genomic_r1_d30000.npz"
AUX = f"{R}/outputs/grid_pools/k562_hepg2/hepg2/genomic_r1_d30000.npz"
BASE = " --flank_ctx 20 --shift_max 10"
ARMS = {
    "mt38_k562only_r1_30k": BASE,
    "mt38_aux05_r1_30k": BASE + f" --aux_labels {AUX} --aux_weight 0.5",
    "mt38_aux10_r1_30k": BASE + f" --aux_labels {AUX} --aux_weight 1.0",
}


def all_cells():
    out = []
    for name, extra in ARMS.items():
        out += fb.cells(name, V1, POOL, 30000, {}, extra)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        i = max(i for i, c in enumerate(reg) if c["name"].startswith("fx36_c20_tta8_30k"))
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells after index", i, "| registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_mt38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
