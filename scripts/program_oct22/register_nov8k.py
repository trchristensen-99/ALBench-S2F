"""Nov-8: give al54 its iterative arm, so the 30k-start curve has interior rungs.

al54 was registered as a confound check -- is al52's +0.102 the selector, or an escape from a genomic
pilot? -- and for that it only needed one-shot arms. But an iterative arm trains an ensemble at every
rung it passes through (30k, 55k, 80k, then 105k), and a one-shot arm trains two, so al54 as registered
produces a 30k-start curve with no interior. al53 will produce 300k, 400k, 500k, 600k on the same corpus
from its `it3_epi` arm, and the from-scratch ladder already has 10k through 1M, so the one missing piece
of the three-curve comparison is the interior of the 30k start on motif_shared_core.

Two cells, epistemic only, matching al53's arm set and al52's schedule. About 10 GPU-hours.

    python scripts/program_oct22/register_nov8k.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov8i as i  # noqa: E402

DRY = i.DRY


def all_cells():
    """al54's own cell builder, with only the schedule changed -- nothing else may drift between them."""
    out = []
    for rep in i.REPS:
        needs = [f"{i.PD}/cand_r{rep}.npz", f"{i.DONE}/pilot_s{rep}.done"]
        c = i.ladder(f"it3_epi_s{rep}", "epistemic", rep, needs)
        c["cmd"] = c["cmd"].replace("--schedule 75000", "--schedule 25000,25000,25000")
        out.append(c)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        for c in new:
            assert "--schedule 25000,25000,25000" in c["cmd"], "the schedule did not get swapped"
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        for c in new:
            print(" ", c["name"])
        if DRY:
            print()
            print(new[0]["cmd"] if new else "nothing to add")
            return
        shutil.copy(path, path + ".bak_nov8k")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
