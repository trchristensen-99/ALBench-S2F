"""Nov-5: rerun the three loop decisions that were measured while the harness was still truncating training.

Explore fraction, ideas per round and showing the reference were all decided on arms whose models were cut short: the explore-fraction arms asked for a median 200 / 120 / 80 epochs at 0.2 / 0.6 / 1.0 and were
truncated in 18 / 23 / 37% of models, so "more exploration is worse" is confounded with "more exploration trained less". The ideas-per-call result is older still (June, and the 1x60 vs 6x10 cadence test of
Oct 13), from before the patience fix. All three now rerun on the `ar41` footing - architecture-only search, every model on the tuned 300-epoch recipe, patience off - so the only thing that varies is the knob.

  ef45_{20,100}_30k_r<k>   explore fraction 0.2 and 1.0 against ar41's 0.4
  ip45_{3,12}_30k_r<k>     3 and 12 ideas per round against ar41's 6
  nr45_30k_r<k>            the reference LegNet and its bar NOT shown (ar41 shows both)

15 runs, 3 replicate pools each, 30 models per run. The LLM budget has been idle; these are read against `ar41` on the same pools.

    python scripts/program_oct22/register_nov5b.py [--dry]
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
H3 = {"AL_RC_MODE": "aug", "AL_RECIPE_MODE": "tuned", "AL_PATIENCE": "9999"}
TUNED_30K = "adamw,0.01,0.2,1024,300,onecycle"
ARMS = (
    ("ef45_20", {"explore_frac": 0.2}, True),
    ("ef45_100", {"explore_frac": 1.0}, True),
    ("ip45_3", {"explore_frac": 0.4, "ideas_per_round": 3}, True),
    ("ip45_12", {"explore_frac": 0.4, "ideas_per_round": 12}, True),
    ("nr45", {"explore_frac": 0.4}, False),  # reference hidden
)


def all_cells():
    reps, cap, budget = nb.RR_SIZE
    out = []
    for tag, cfg, show_ref in ARMS:
        for r in range(1, reps + 1):
            c = nb.loop_cell(
                f"{tag}_30k_r{r}", f"genomic_30k_{tag}", r, 30000, 15, cap, budget,
                {**nb.RR, "manager": "schedule_ens", "recipe_source": "harness", **cfg},
            )  # fmt: skip
            c["env"] = {**c["env"], **H3, "AL_TUNED_RECIPE": TUNED_30K}
            c["gpus"] = ["v100"]
            if (
                not show_ref
            ):  # the reference flag is added by nb.loop_cell; this arm is the control without it
                assert " --reference_legnet " in c["cmd"], c["name"]
                c["cmd"] = c["cmd"].replace(" --reference_legnet ", " ", 1)
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
            for c in new[:1] + new[-1:]:
                print(c["name"], "ref_shown=", " --reference_legnet " in c["cmd"])
                print(
                    " ", c["cmd"][c["cmd"].find("--agent_cfg") : c["cmd"].find("--agent_cfg") + 230]
                )
            return
        shutil.copy(path, path + ".bak_nov5b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
