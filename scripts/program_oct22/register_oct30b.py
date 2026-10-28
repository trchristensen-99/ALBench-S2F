"""Four more arms of the 30k-genomic top-up experiment (from the frontier model's notes on acquisition families): the oracle-error ceiling, 2D activity x GC
stratified baseline, Neyman-allocation composition design and stochastic SD sampling. Pools come from acq_arms2.py. Cells go right after the al31_ block; the
CPU forward gate is off and V1 menu entry 6 never goes to a MIG slice, as for the other al31_ cells.

  python scripts/program_oct22/register_oct30b.py [--dry]
"""

import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scripts/program_oct22")
import register_oct30 as r  # noqa: E402

os.chdir(r.R)
ARMS = ("oracle", "strat2d", "neyman", "powersd")
reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
new = []
for a in ARMS:
    for c in r.retrain(
        f"al31_mix_{a}",
        f"{r.ACQ}/sel_{a}_d100000.npz",
        100000,
        f"outputs/menu_retrain_v0/al31_mix_{a}",
        6,
    ):
        c["env"]["MENU_MAX_FWD_MS"] = "1000000000"
        if c["name"] not in have:
            new.append(c)
i = max(k for k, c in enumerate(reg) if c["name"].startswith("al31_"))
reg2 = reg[: i + 1] + new + reg[i + 1 :]
print("new cells", len(new), "registry", len(reg), "->", len(reg2))
if "--dry" not in sys.argv:
    shutil.copy("scratchpad/dispatch_registry.json", "scratchpad/dispatch_registry.json.bak_oct30b")
    json.dump(reg2, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
    os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
    print("registered")
