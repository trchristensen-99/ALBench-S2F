"""Oct-30: AutoResearch searches at D = 100k with the study's input standard, the 100k counterpart of the collaborator's optimisation at 30k and of the st38 runs at 10k and 30k.

  st38_a0_genomic_100k/rep{1,2}   the standard loop (known-good guidance, ensemble-aware, coordinator, proposer-chosen recipes)
  st38_a4_genomic_100k/rep{1,2}   the gentle-steering arm (minimal guidance, reference LegNet, quota manager, harness recipes)

Same loop, model (Sonnet 5.5), 10 rounds x 2 models and validation (V0c) as st38; independent replicate pools r1, r2 (genomic_r{1,2}_d100000). The per-model time cap is raised from 25 to 45 base
minutes because a 100k model trains about 3.3x longer than a 30k one and the V100 rule scales the cap by 5.

    python scripts/program_oct22/register_st100.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_st38 as st  # noqa: E402

DRY = "--dry" in sys.argv


def all_cells():
    out = []
    for r in (1, 2):
        for arm in ("a0", "a4"):
            c = st.cell(arm, "genomic", 100000, r)
            assert "--base_cap_min 25" in c["cmd"]
            c["cmd"] = c["cmd"].replace("--base_cap_min 25", "--base_cap_min 45")
            out.append(c)
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
        shutil.copy(path, path + ".bak_st100")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
