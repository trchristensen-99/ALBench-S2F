"""Nov-2 (registered again after patch_loop_bankbar.py: the bar is the seed banks' mean validation r, and the LLM credentials were fixed by linking the old home's .local, .claude and .bash_profile into the new home).

Nov-2: the loop against the tuned bar. The in-run LegNet baseline is now the per-size tuned recipe (patch_loop_tuned.py), with the out-of-the-box recipe reported beside it. These cells repeat the best manager arm of
Oct 30 (mgr38_ref: coordinator, explore the first 40% of the rounds then exploit, published MPRA-LegNet shown as the reference, known-good guidance, 15 rounds x 2 models, Sonnet 5.5, standard input, V0c) at all three sizes:

  rr38_<D>k_r<k>   D = 10k and 30k on replicate pools r1 to r3, 100k on r1 and r2 (per-model cap 45 H100-min at 100k, 25 otherwise)

At 30k they are paired by pool with mgr38_ref_r<k> (same arm, weaker bar), which isolates what the better reference does to the search; at all sizes they are compared with the 20-seed tuned LegNet banks and the
out-of-the-box LegNet banks (docs/TUNED_LEGNET.md), which is the question the loop has to answer: what does architecture search add to a LegNet that is already trained properly?

    python scripts/program_oct22/register_rr38.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_lr35 as lr  # noqa: E402

DRY = "--dry" in sys.argv
COORD = {
    "stages": ["propose", "coordinate", "code", "evaluate"],
    "ideas_per_round": 6,
    "manager": "schedule_ens",
    "explore_frac": 0.4,
}
SIZES = {
    10000: (3, 25, 10.5),
    30000: (3, 25, 10.5),
    100000: (2, 45, 22.0),
}  # replicate pools, per-model cap (H100-min), budget hours


def all_cells():
    out = []
    for D, (reps, cap, budget) in SIZES.items():
        for r in range(1, reps + 1):
            c = lr.cell(f"rr38_{D // 1000}k_r{r}", f"genomic_{D // 1000}k_rr38", r, D, 15, 2, COORD)
            assert " --val_cache " in c["cmd"] and " --base_cap_min 25 " in c["cmd"]
            c["cmd"] = (
                c["cmd"]
                .replace(" --val_cache ", " --reference_legnet --val_cache ", 1)
                .replace(" --base_cap_min 25 ", f" --base_cap_min {cap} ", 1)
            )
            c["cmd"] = c["cmd"].replace("{BUDGET}", str(budget))
            if D >= 100000:
                c["gpus"] = ["h100"]  # the tuned 100k baseline alone takes about 25 H100-minutes
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
            print(new[0]["cmd"][:600])
            return
        shutil.copy(path, path + ".bak_rr38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
