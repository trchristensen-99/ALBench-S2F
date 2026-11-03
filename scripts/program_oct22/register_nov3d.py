"""Nov-3: does refining the best model of each architecture FAMILY (not only the best model overall) put diverse, near-optimal members into the ensemble?

The funnel analysis of the 21 standard 30k loop runs (`docs/DIVERSITY_FUNNEL_NOV3.md`) found the coordinator and proposer are not what loses diversity: independent (non-LegNet-trunk) designs are 43% of explore
proposals and are picked at the same rate as LegNet-named ones (35% against 32%). They are lost afterwards: they train with short recipes, end 0.054 validation r below the best LegNet-named design, and the exploit
phase only ever refines the best model overall, which is a LegNet descendant, so the best independent design is never improved (its best-so-far is unchanged from round 5 to the end in 17 of 21 runs).

  is40_30k_r<k>   manager "islands": explore rounds as in rr40; every exploit round refines ONE family's incumbent (family = mechanism x trunk, `agent_loop.island_of`), alternating between the best model overall
                  and the best model of another family (other trunk first); recipes written by the proposer, as in rr40
  ia40_30k_r<k>   islands + the tuned recipe for every model (AL_RECIPE_MODE=tuned, recipe_source harness): the 2x2 with rr40 (schedule, proposal recipes) and ar40 (schedule, tuned recipe)

Same pools, harness H3 and transport as rr40 (30k, 15 rounds x 2, explore 40%, cap 40 H100-minutes). Read-outs against rr40 and ar40: stack r (genomic, OOD, mean of 18); stack weight on members of other trunks
and of other mechanisms; best validation r of the best other-trunk model at the end of exploration against the end of the run; whether those models are in the final stack (`extract_diversity_funnel.py`).

    python scripts/program_oct22/register_nov3d.py [--dry]
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
TUNED_30K = "adamw,0.01,0.2,1024,300,onecycle"


def all_cells():
    reps, cap, budget = nb.RR_SIZE
    out = []
    for r in range(1, reps + 1):
        c = nb.loop_cell(
            f"is40_30k_r{r}", "genomic_30k_is40", r, 30000, 15, cap, budget,
            {**nb.RR, "manager": "islands", "explore_frac": 0.4},
        )  # fmt: skip
        c["env"] = {**c["env"], **H3}
        out.append(c)
        a = nb.loop_cell(
            f"ia40_30k_r{r}", "genomic_30k_ia40", r, 30000, 15, cap, budget,
            {**nb.RR, "manager": "islands", "explore_frac": 0.4, "recipe_source": "harness"},
        )  # fmt: skip
        a["env"] = {**a["env"], **H3, "AL_RECIPE_MODE": "tuned", "AL_TUNED_RECIPE": TUNED_30K}
        out.append(a)
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
            print(
                "  ",
                c["name"],
                c["gpus"],
                {
                    k: v
                    for k, v in c["env"].items()
                    if k.startswith("AL_RC") or k.startswith("AL_RECIPE")
                },
            )
        if DRY:
            return
        shutil.copy(path, path + ".bak_nov3e")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
