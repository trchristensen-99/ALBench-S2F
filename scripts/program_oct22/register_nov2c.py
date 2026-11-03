"""Nov-2 night: the fixed harness plus single-pass reverse-complement augmentation ("H3").

pa39 (3 seeds each, tuned recipe, 30k) found where the harness loses the published LegNet 0.034 genomic r against the native trainer: **the reverse-complement consistency loss** (the average of the forward and RC losses, two forward
passes per step), not the per-sample shift. With AL_RC_MODE=aug (the oracle's and the bank's protocol: half of the samples flipped, one pass) the harness LegNet scores 0.8736 genomic / 0.387 OOD / 0.791 random against 0.8748 / 0.398 / 0.793
for the native bank, and 0.8416 / 0.338 / 0.759 with the consistency loss; per-sample shift alone: 0.8389; both: 0.8737.

  rr40_30k_r<k>   the standard managed loop of rr39 with AL_RC_MODE=aug   (3 replicate pools)
  lv40_30k_r<k>   the LegNet-variants loop of lv39 with AL_RC_MODE=aug    (3 replicate pools)

Everything else is H2 (patience 16, one-cycle never stopped, fp16, non-finite skip) and the isolated, medium-effort CLI transport. Compared with rr39 / lv39 (same pools) and the native bank.

    python scripts/program_oct22/register_nov2c.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov2b as nb  # noqa: E402

DRY = nb.DRY
H3 = {"AL_RC_MODE": "aug"}


def all_cells():
    reps, cap, budget = nb.RR_SIZE
    out = []
    for r in range(1, reps + 1):
        c = nb.loop_cell(
            f"rr40_30k_r{r}",
            "genomic_30k_rr40",
            r,
            30000,
            15,
            cap,
            budget,
            nb.RR | {"explore_frac": 0.4},
        )
        c["env"] = {**c["env"], **H3}
        out.append(c)
    _, rounds, cap, budget = nb.LV_SIZES[30000]
    for r in range(1, reps + 1):
        c = nb.loop_cell(
            f"lv40_30k_r{r}", "genomic_30k_lv40", r, 30000, rounds, cap, budget, nb.COORD_VARIANTS
        )
        c["env"] = {**c["env"], **H3}
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
        for c in new:
            print("  ", c["name"], c["env"])
        if DRY:
            return
        shutil.copy(path, path + ".bak_nov2c")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
