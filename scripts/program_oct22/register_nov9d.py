"""Nov-9: give the loop's 100k arm the replicates it never had.

`lv39_100k_r1` is COMPLETE and is the only run of the AutoResearch loop at 100k. Every claim about
whether the loop beats the tuned LegNet reference at that size rests on it, and a single loop run
carries sigma ~= 0.008 (project_autoresearch_variance_menu_oct19) -- the same size as the gaps being
claimed. This is exactly the defect the from-scratch curve had this morning and that sc56/sc57 fixed:
a number with no band compared against a number with one.

100k is the size that matters now rather than 30k. Today's acquisition reading (read_curves.py) puts
the recommendation in the 100k+ region: one-shot epistemic selection carries the gain (+0.071 over
random at 105k), rounds add +0.001 on top of a good pilot, and the from-scratch curve keeps climbing
to 1M. So the open architecture question is whether the loop adds anything over tuned LegNet where the
corpus recommendation will actually be made -- and at 30k it has three replicates while at 100k it has
one.

Two replicates, not more: three total matches the 30k arm, and it is 2 x 36 budget-hours = 72 GPU-h at
the cap (the budget is a ceiling, so likely less). H100-only, because `loop_cell` pins D >= 100k there.

The cells are the SAME arm, not a new one: name `lv39_100k_r{2,3}`, built by the same `loop_cell` with
the same COORD_VARIANTS config, differing only in the replicate pool. That is safe where al54 was not,
because the output directory is per-replicate (`genomic_100k_lv39/rep{r}`) -- so rep2 and rep3 are
empty and will really train. Both properties are asserted below rather than assumed: the command is
checked to be r1's command with only the replicate substituted, and the output directories are checked
to be empty.

    python scripts/program_oct22/register_nov9d.py [--dry]
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
REPS = (2, 3)
ROUNDS, CAP, BUDGET = nb.LV_SIZES[D][1:]  # 6 rounds, 100 H100-min per model, 36 budget hours


def all_cells():
    return [
        nb.loop_cell(
            f"lv39_{D // 1000}k_r{r}",
            f"genomic_{D // 1000}k_lv39",
            r,
            D,
            ROUNDS,
            CAP,
            BUDGET,
            nb.COORD_VARIANTS,
        )
        for r in REPS
    ]


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        by_name = {c["name"]: c for c in reg}
        r1 = by_name[f"lv39_{D // 1000}k_r1"]
        cells = all_cells()

        # 1. a replicate is only a replicate if nothing but the draw changed
        for c in cells:
            r = c["name"].rsplit("_r", 1)[1]
            want = r1["cmd"].replace("_r1_d", f"_r{r}_d").replace("/rep1", f"/rep{r}")
            assert c["cmd"] == want, f"{c['name']} differs from r1 beyond the replicate"
            assert c["env"] == r1["env"], f"{c['name']} env differs from r1"
            assert c["gpus"] == ["h100"], c["gpus"]
            # 2. an arm whose output dir already holds a run skips instead of training (the al54 lesson)
            od = os.path.join(R, c["out_dir"])
            assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
        print(f"guard: {len(cells)} cells identical to r1 bar the draw, output dirs empty")

        have = set(by_name)
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        for c in new:
            print(" ", c["name"], c["needs"])
        if DRY:
            print()
            print(new[0]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov9d")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
