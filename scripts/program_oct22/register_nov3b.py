"""Nov-3: the diversity arms (30k, 3 replicate pools each, H3 = patience fix + fp16 + non-finite skip + single-pass RC augmentation + isolated medium-effort transport).

The H3 loop reaches the bank, but its good members are LegNet descendants (the stack of the LegNet-derived members is the whole stack; the other members alone reach 0.81) and the stack has 6 to 7 effective members against 16 for the bank.
The explored non-LegNet architectures may be weak because they are trained with proposal-written small-batch recipes, not because the architectures are weak. Two ways to push diversity:

  ar40_30k_r<k>   architecture-only search under the tuned recipe: AL_RECIPE_MODE=tuned, every model trains with AdamW lr 0.01, wd 0.2, batch 1024, 300 epochs, one-cycle (the 30k recipe of the LegNet bank, shortened to 300 epochs,
                  within 0.002 of 500), recipe_source "harness" (the proposer writes architectures only and is told the recipe), explore 40% then exploit, 15 rounds x 2.
  qt40_30k_r<k>   the quota manager: every round trains one exploit model and one explore model; the coordinator sees a coverage table of architectural features and which are untried, and the validation stack; proposal-written recipes.

Read-outs against rr40 (same pools): stack r, the share of stack weight on members not derived from the reference LegNet, effective members, the stack of the non-LegNet members alone, and the features trained.

    python scripts/program_oct22/register_nov3b.py [--dry]
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
            f"ar40_30k_r{r}", "genomic_30k_ar40", r, 30000, 15, cap, budget,
            nb.RR | {"explore_frac": 0.4, "recipe_source": "harness"},
        )  # fmt: skip
        c["env"] = {**c["env"], **H3, "AL_RECIPE_MODE": "tuned", "AL_TUNED_RECIPE": TUNED_30K}
        out.append(c)
        q = nb.loop_cell(
            f"qt40_30k_r{r}", "genomic_30k_qt40", r, 30000, 15, cap, budget,
            {"stages": nb.RR["stages"], "ideas_per_round": 6, "manager": "quota"},
        )  # fmt: skip
        q["env"] = {**q["env"], **H3}
        out.append(q)
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
            print("  ", c["name"], {k: v for k, v in c["env"].items() if k.startswith("AL_")})
        if DRY:
            return
        shutil.copy(path, path + ".bak_nov3b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
