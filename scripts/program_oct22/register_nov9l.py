"""Nov-9: run the manager that produces the best DIVERSITY at 100k, which neither 100k arm does.

Measured today, stacked genomic test r at 30k against the 20-seed tuned bank (0.8907), with the
effective member count the stack actually extracts:

    is40  0.8874  eff 6.8     ar41  0.8871  eff 12.3     ln42  0.8865  eff 7.6
    af42  0.8858  eff 9.6     lv39  0.8544  eff 5.6

ar41 and af42 are the same manager (schedule_ens, explore 0.4) and differ only in whether the proposer
may write the training recipe. They are level on solo quality, and not level on anything else: ar41's
non-LegNet members stack to 0.8673 against af42's 0.8372, and ar41 extracts 12.3 effective members
against 9.6. Fixing the recipe and freeing only the architecture is what produced usable diversity.

That matters most exactly where LegNet is strongest. At 10k the loop beats the bank outright (rr40 0.8029
against 0.7801); at 30k the frozen menu reaches parity; at 100k menu v2 was -0.018 behind. As D rises the
win stops coming from a better architecture and starts coming from members that are differently wrong, so
the arm to run at 100k is the one that generates decorrelated members -- and the two 100k arms running
are af42 (free recipe) and lv39, which is the weakest manager at 30k by -0.033.

The enforced recipe is the 100k entry of the per-size table, not the 30k one ar41 carried: at 100k that
is e500 at weight decay 0.1, taken from `register_nov2b.TUNED` rather than written out here, so it cannot
drift from the table the bank is trained with.

Compute is pinned to lv39_100k and af42_100k flag by flag (6 rounds, 100 H100-min per model, 36 budget
hours), so the three 100k arms differ in the manager and the recipe freedom and in nothing else.
About 72 GPU-hours over two replicates.

    python scripts/program_oct22/register_nov9l.py [--dry]
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
TAG, MGR = "ar41", "schedule_ens"
ROUNDS, CAP, BUDGET = nb.LV_SIZES[D][1:]  # 6 rounds, 100 H100-min per model, 36 budget hours
EP, WD = nb.TUNED[
    D
]  # the per-size table's 100k entry, the same one the reference bank is trained with
TUNED_100K = f"adamw,0.01,{WD},1024,{EP},onecycle"
# patience off for the same reason af42 needed it: a proposer-chosen schedule was being stopped early
ENV = {
    "AL_RC_MODE": "aug",
    "AL_PATIENCE": "9999",
    "AL_RECIPE_MODE": "tuned",
    "AL_TUNED_RECIPE": TUNED_100K,
}


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
            {**nb.RR, "manager": MGR, "explore_frac": 0.4, "recipe_source": "harness"},
        )
        c["env"] = {**c["env"], **ENV}
        out.append(c)
    return out


def main():
    cells = all_cells()
    ref = {c["name"]: c for c in json.load(open(f"{R}/scratchpad/dispatch_registry.json"))}
    for c in cells:
        assert c["gpus"] == ["h100"], c["gpus"]
        assert f'"manager": "{MGR}"' in c["cmd"], "the manager did not reach the agent config"
        assert f'"recipe_source": "harness"' in c["cmd"], (
            "architecture-only search did not reach it"
        )
        assert c["env"]["AL_TUNED_RECIPE"] == TUNED_100K, c["env"]["AL_TUNED_RECIPE"]
        assert (EP, WD) == (500, 0.1), f"the 100k table entry moved: {(EP, WD)}"
        # same compute as the other two 100k loop arms, so the manager is the only difference
        for other in (f"lv39_{D // 1000}k_r1", f"af42_{D // 1000}k_r1"):
            o = ref[other]
            for flag in (
                "--rounds",
                "--per_round",
                "--budget_hours",
                "--base_cap_min",
                "--max_params",
            ):
                i, j = (
                    c["cmd"].split(flag + " ")[1].split()[0],
                    o["cmd"].split(flag + " ")[1].split()[0],
                )
                assert i == j, f"{flag}: {i} against {other}'s {j}"
        # --resume means a non-empty dir continues someone else's run rather than starting this one
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(
        f"guard: {len(cells)} cells, h100, recipe {TUNED_100K}, compute matched to lv39+af42 at {D:,}"
    )

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
        shutil.copy(path, path + ".bak_nov9l")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
