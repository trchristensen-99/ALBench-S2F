"""Nov-6: the cheap loop changes - drop the coordinator, shorten the run, and show the proposer the code that failed.

Three measurements say the coordinator is not doing much: it gains +0.004 over a random pick of 2 from 6, only +0.001 over no selection at all, and moving from 3 to 12 ideas per round changes nothing. If
selection is that weak, the call is pure overhead. Separately, 10 rounds reaches within 0.002 of 15 and 30 adds nothing, so a shorter run frees budget for more POOLS, which is where the variance actually is.

  nc50_30k_r<k>   no coordinator (manager "none", stages propose/code/evaluate), 2 ideas per round, both trained, 15 rounds
  sh50_30k_r<k>   nc50 but 10 rounds, so 20 models: is a third of the budget enough?
  fc50_30k_r<k>   ar41 with AL_SHOW_FAILED_CODE=1: the proposer sees the first lines of each discarded model instead of a
                  one-line reason, which should stop the same broken construct reappearing. No extra compute.

All on Opus 5.5 and otherwise matched to `op49` (architecture-only, tuned 300-epoch recipe, patience off, three pools), so
each is read against that baseline rather than against the older Sonnet runs.

    python scripts/program_oct22/register_nov6f.py [--dry]
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

R, DRY = nb.R, nb.DRY
OPUS = "claude-opus-5-5"
TUNED_30K = "adamw,0.01,0.2,1024,300,onecycle"
BASE_ENV = {"AL_RC_MODE": "aug", "AL_RECIPE_MODE": "tuned", "AL_TUNED_RECIPE": TUNED_30K, "AL_PATIENCE": "9999", "LLM_MODEL": OPUS}  # fmt: skip
NO_COORD = {"stages": ["propose", "code", "evaluate"], "manager": "none", "ideas_per_round": 2}
ARMS = (
    ("nc50", 15, NO_COORD, {}),
    ("sh50", 10, NO_COORD, {}),
    ("fc50", 15, {"manager": "schedule_ens", "explore_frac": 0.4}, {"AL_SHOW_FAILED_CODE": "1"}),
)


def all_cells():
    reps, cap, budget = nb.RR_SIZE
    out = []
    for tag, rounds, cfg, extra in ARMS:
        for r in range(1, reps + 1):
            c = nb.loop_cell(
                f"{tag}_30k_r{r}", f"genomic_30k_{tag}", r, 30000, rounds, cap, budget,
                {**nb.RR, "recipe_source": "harness", **cfg},
            )  # fmt: skip
            c["env"] = {**c["env"], **BASE_ENV, **extra}
            c["gpus"] = ["v100"]
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
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            for c in (new[0], new[3], new[6]):
                i = c["cmd"].find("--agent_cfg")
                print(" ", c["name"], c["cmd"][i : i + 170])
            return
        shutil.copy(path, path + ".bak_nov6f")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
