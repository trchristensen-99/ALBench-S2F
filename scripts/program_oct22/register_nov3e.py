"""Nov-3 afternoon: replace the mis-specified ar40/ia40 arms and add the lanes arm.

BUG (found Nov 3 13:30): `agent_loop.run_agent_loop` overwrote AL_RECIPE_MODE=tuned with 'harness' whenever recipe_source was 'harness', so ar40 and ia40 trained Muon with a small random spread for 80 epochs, not the tuned
300-epoch AdamW recipe their registration said (the plan files showed empty train specs, which looked right; the trained epochs, 80, were never checked). Fixed in agent_loop (the mode is kept; a runtime guard refuses to train a model
whose assigned recipe differs from AL_TUNED_RECIPE; regression test `test_tuned_recipe_mode_survives_the_loop_and_is_applied`). The six cells are retired (outputs kept as `*_wrongrecipe`) and replaced:

  ar41_30k_r<k>   schedule_ens (explore 40% then exploit), architecture-only, every model trained with AdamW lr 0.01, wd 0.2, batch 1024, 300 epochs, one-cycle    (the user's hypothesis: give every design the long recipe)
  ln41_30k_r<k>   manager "lanes": every round trains one lane-A model (built on the reference LegNet's blocks) and one lane-B model (no EffBlock/SELayer), the same tuned recipe for both

Bake-off evidence behind it (register_nov3c.py, partial): the best independent designs from earlier runs gain +0.03 to +0.05 validation r (test +0.03 to +0.05) when retrained with the tuned recipe.

    python scripts/program_oct22/register_nov3e.py [--dry]
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
RETIRE = ("ar40_30k_", "ia40_30k_")


def all_cells():
    reps, cap, budget = nb.RR_SIZE
    out = []
    for r in range(1, reps + 1):
        for tag, mgr in (("ar41", "schedule_ens"), ("ln41", "lanes")):
            c = nb.loop_cell(
                f"{tag}_30k_r{r}", f"genomic_30k_{tag}", r, 30000, 15, cap, budget,
                {**nb.RR, "manager": mgr, "explore_frac": 0.4, "recipe_source": "harness"},
            )  # fmt: skip
            c["env"] = {**c["env"], **H3, "AL_RECIPE_MODE": "tuned", "AL_TUNED_RECIPE": TUNED_30K}
            c["gpus"] = ["v100"]
            out.append(c)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        keep = [c for c in reg if not c["name"].startswith(RETIRE)]
        have = {c["name"] for c in keep}
        new = [c for c in all_cells() if c["name"] not in have]
        reg2 = keep + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(
            "retire",
            len(reg) - len(keep),
            "| add",
            len(new),
            "| registry",
            len(reg),
            "->",
            len(reg2),
        )
        for c in new:
            print(
                "  ",
                c["name"],
                c["gpus"],
                {
                    k: v
                    for k, v in c["env"].items()
                    if k.startswith(("AL_RC", "AL_RECIPE", "AL_TUNED"))
                },
            )
        if DRY:
            return
        shutil.copy(path, path + ".bak_nov3g")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
