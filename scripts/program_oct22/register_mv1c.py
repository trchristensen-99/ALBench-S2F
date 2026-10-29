"""Oct-29 (evening): menu v1c = menu v1 with the five architectures that were hard-wired to 200 positions (indices 0, 10, 12, 13, 15) rewritten to accept any input length divisible by 8. At 200 bp each is the
same network as before (identical state dict, outputs equal to 1e-4 with the original's weights). Trained here with 20 real flank bases and a +-10 shift on the canonical genomic 30k pool (V0c validation, patience 16):
all 20 configurations, paired with menu v1 without context (hp34_base_30k) and, for the 15 old ones, with fx35_c20s10_30k.

  mv1c_c20_30k   all 20 configurations

    python scripts/program_oct22/register_mv1c.py [--dry]
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
MENU = f"{R}/outputs/menus_oct29/V1c.json"


def all_cells():
    return fb.cells(
        "mv1c_c20_30k",
        MENU,
        f"{NEST}/genomic_d30000.npz",
        30000,
        {},
        " --flank_ctx 20 --shift_max 10",
    )


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
        shutil.copy(path, path + ".bak_mv1c")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
