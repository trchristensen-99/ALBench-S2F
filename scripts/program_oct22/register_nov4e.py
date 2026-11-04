"""Nov-4: does a mixed validation set hurt the loop by itself, or only because it shortened training?

`vc39` (mixed validation B1 for stopping, proposer feedback and the stack) was -0.017 genomic and worse on 16 of 18 sets against `rr39`, but its proposer asked for a median 90 epochs against 200 and patience cut
29% of its models short against 16% (`analyze_loop_ood_gap.py` and the meta files): the validation set changed how long models were trained, and short training costs every set. This arm is `ar41`
(schedule_ens, explore 0.4, architecture-only, every model on the tuned recipe AdamW lr 0.01, wd 0.2, batch 1024, 300 epochs, patience off) with B1 in place of V0c, so the schedule cannot change and only
selection, feedback and the stack weights see the mixed set. Compare with `ar41` on the same pools.

  vc43_30k_r<k>   outputs/archsearch_reps_oct22/genomic_30k_vc43/rep<k>

    python scripts/program_oct22/register_nov4e.py [--dry]
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
import register_nov2b as nb  # noqa: E402

R, DRY = nb.R, nb.DRY
TUNED_30K = "adamw,0.01,0.2,1024,300,onecycle"


def all_cells():
    reps, cap, budget = nb.RR_SIZE
    out = []
    for r in range(1, reps + 1):
        c = nb.loop_cell(
            f"vc43_30k_r{r}", "genomic_30k_vc43", r, 30000, 15, cap, budget,
            {**nb.RR, "manager": "schedule_ens", "explore_frac": 0.4, "recipe_source": "harness"},
        )  # fmt: skip
        assert lr.V0C in c["cmd"]
        c["cmd"] = c["cmd"].replace(lr.V0C, nb.B1)
        c["needs"] = c["needs"] + ["outputs/val_variants/val_B1.npz"]
        c["env"] = {
            **c["env"],
            "AL_RC_MODE": "aug",
            "AL_RECIPE_MODE": "tuned",
            "AL_TUNED_RECIPE": TUNED_30K,
            "AL_PATIENCE": "9999",
        }
        c["gpus"] = [
            "v100"
        ]  # as ar41: keeps the LLM cell off the H100 queue the dispatcher strands
        if not DRY:  # the genomic bar does not apply to this validation set (as vc39)
            d = f"{R}/outputs/archsearch_reps_oct22/genomic_30k_vc43/rep{r}"
            os.makedirs(d, exist_ok=True)
            if not os.path.exists(f"{d}/baseline_legnet.txt"):
                open(f"{d}/baseline_legnet.txt", "w").write(nb.B1_BASELINE)
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
            print(new[0]["env"] if new else "", "\n", new[0]["cmd"][:600] if new else "nothing new")
            return
        shutil.copy(path, path + ".bak_nov4e")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
