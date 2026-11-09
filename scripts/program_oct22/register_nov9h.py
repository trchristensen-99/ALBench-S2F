"""Nov-9: take the free-recipe loop manager to 100k, the size the recommendation will be made at.

af42 (schedule_ens, proposer writes the whole TRAIN line, epochs fixed at 300) and ln42 (lanes) have
only ever run at 30k. Today's curves put the decision in the 100k+ region, and the architecture question
there is open: at 30k the loop reaches the tuned-LegNet level, and nobody has asked whether it still
does when the corpus is three times larger and LegNet's own per-size recipe has moved on.

Budget note, stated because it corrects the figure I quoted when asking: af42 AND ln42 at two replicates
each would be 4 x 36 = ~144 budget-hours, not the ~72 I said. So this registers one manager at two
replicates rather than two managers at one. That is also the better buy -- a single-draw loop arm is
precisely the defect lv39_100k had this morning, and a one-run answer at this size could not be
distinguished from the ~0.008 single-run sigma. af42 is the one taken because the free recipe is the
part that has not been tested above 30k; ln42 can follow and would need its own ask.

Settings follow lv39_100k exactly (6 rounds, 100 H100-min per model, 36 budget-hours) so the two 100k
loop arms differ in the manager and the recipe freedom, not in how much compute they were given.
H100-only: `loop_cell` pins that for D >= 100k.

    python scripts/program_oct22/register_nov9h.py [--dry]
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
D = 100_000
REPS = (1, 2)
TAG, MGR = "af42", "schedule_ens"
ROUNDS, CAP, BUDGET = nb.LV_SIZES[D][1:]  # 6 rounds, 100 H100-min per model, 36 budget hours
# patience off: AL_ONECYCLE_FULL protects only one-cycle schedules, and a proposer-chosen cosine
# schedule was stopped at 69-105 of 300 epochs when af42 first launched at 30k (Nov 3 17:30)
ENV = {"AL_RC_MODE": "aug", "AL_FIXED_EPOCHS": "300", "AL_PATIENCE": "9999"}


def all_cells():
    out = []
    for r in REPS:
        c = nb.loop_cell(
            f"{TAG}_{D // 1000}k_r{r}",
            f"genomic_{D // 1000}k_{TAG}",
            r,
            D,
            ROUNDS,
            CAP,
            BUDGET,
            {**nb.RR, "manager": MGR, "explore_frac": 0.4},
        )
        c["env"] = {**c["env"], **ENV}
        out.append(c)
    return out


def main():
    cells = all_cells()
    ref = {c["name"]: c for c in json.load(open(f"{R}/scratchpad/dispatch_registry.json"))}
    lv = ref[f"lv39_{D // 1000}k_r1"]
    for c in cells:
        assert c["gpus"] == ["h100"], c["gpus"]
        assert f'"manager": "{MGR}"' in c["cmd"], "the manager did not reach the agent config"
        assert f"--rounds {ROUNDS} " in c["cmd"] and f"--budget_hours {BUDGET}" in c["cmd"]
        assert f"--base_cap_min {CAP} " in c["cmd"], "the per-model cap did not get swapped"
        # same compute as the other 100k loop arm, so the manager is the only difference that matters
        for flag in ("--rounds", "--per_round", "--budget_hours", "--base_cap_min", "--max_params"):
            i, j = (
                c["cmd"].split(flag + " ")[1].split()[0],
                lv["cmd"].split(flag + " ")[1].split()[0],
            )
            assert i == j, f"{flag}: {i} against lv39_100k's {j}"
        # --resume means a non-empty dir continues someone else's run rather than starting this one
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(f"guard: {len(cells)} cells, h100, same compute as lv39_{D // 1000}k, output dirs empty")

    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        for c in new:
            print(" ", c["name"], c["out_dir"])
        if DRY:
            print()
            print(new[0]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov9h")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
