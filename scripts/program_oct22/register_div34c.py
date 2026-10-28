"""Oct-28 night: the menu architectures under the MPRA-LegNet training recipe (AdamW, peak lr 0.01, weight decay 0.1, one-cycle; batch 256 and the menu's epoch budgets), without and with the real-flank shift.

Why: the best single model from all AutoResearch runs at 30k sits below one tuned MPRA-LegNet, and the benchmark LegNets used a recipe (AdamW, wd 0.1, lr 0.01) that the proposer's recipe range (wd up to 0.05, lr up to about 4e-3) never reached.
  lg34_lgrecipe_30k       menu v1 under the LegNet recipe (paired with hp34_base_30k)
  lg34_lgrecipe_fl_30k    the same with AL_FLANK_SHIFT=1 (paired with fl34_flank_30k)

  python scripts/program_oct22/register_div34c.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_div34 as d34  # noqa: E402

DRY = "--dry" in sys.argv
LG = " --optimizer adamw --lr_abs 0.01 --wd_abs 0.1"


def main():
    path = "scratchpad/dispatch_registry.json"
    new_all = d34.cells("lg34_lgrecipe_30k", d34.V1, 30000, LG) + d34.cells(
        "lg34_lgrecipe_fl_30k", d34.V1, 30000, LG, {"AL_FLANK_SHIFT": "1"}
    )
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in new_all if c["name"] not in have]
        i = max(i for i, c in enumerate(reg) if c["name"].startswith("ds34_m100_100k_s"))
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len(reg2) == len(reg) + len(new) and len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells after index", i, "| registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_div34c")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
