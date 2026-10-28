"""Oct-28 evening, idle-capacity follow-up to register_div34.py: a SECOND training seed of two arms, to measure the seed noise of a paired per-configuration comparison.

hp34s1_base_30k   menu v1 at 30k, base recipe, --seed_offset 1  (paired with hp34_base_30k: the difference per configuration between two seeds is the noise floor for every recipe arm)
fl34s1_flank_30k  the flank-aware shift arm with --seed_offset 1 (paired with fl34_flank_30k and hp34s1_base_30k: two seeds of the flank effect)

python scripts/program_oct22/register_div34b.py [--dry]
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


def main():
    path = "scratchpad/dispatch_registry.json"
    new_all = d34.cells("hp34s1_base_30k", d34.V1, 30000, " --seed_offset 1") + d34.cells(
        "fl34s1_flank_30k", d34.V1, 30000, " --seed_offset 1", {"AL_FLANK_SHIFT": "1"}
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
        shutil.copy(path, path + ".bak_div34b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
