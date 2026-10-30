"""Oct-30: does an explore-then-exploit manager with a reference fallback beat the loop without it? Paired by pool with the Oct-29 arms (mgr35_c15: coordinator only; mgr35_e15: schedule_ens manager), same
settings (genomic 30k replicate pools r1..r3, 15 rounds x 2 models, known-good guidance, ensemble-aware, 6 ideas per round, proposer-chosen recipes, standard input, V0c validation, Sonnet 5.5).

  mgr38_ref_r<k>   e15 plus the published MPRA-LegNet as reference code and the harness's own LegNet baseline in the prompt (manager schedule_ens, no fallback rule): isolates the effect of showing the reference
  mgr38_fb_r<k>    the same with manager schedule_fb: explore (first 40% of the rounds) then exploit with a measured reference check; when no explored architecture beats the reference by more than 0.005 validation r
                   the exploit rounds make small modifications of the reference MPRA-LegNet instead of refining the explored winner

Contrasts: fb - ref (the fallback rule alone), ref - e15 (showing the reference), fb - c15 / fb - e15 (everything). One run is 30 models; pools r4..r6 can be added with --reps 6 when the first three look worth extending.

    python scripts/program_oct22/register_mgr38.py [--reps 3] [--dry]
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
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
COORD = {"stages": ["propose", "coordinate", "code", "evaluate"], "ideas_per_round": 6}
ARMS = {
    "ref": {"manager": "schedule_ens", "explore_frac": 0.4},
    "fb": {"manager": "schedule_fb", "explore_frac": 0.4},
}


def all_cells():
    out = []
    for arm, extra in ARMS.items():
        for r in range(1, REPS + 1):
            c = lr.cell(
                f"mgr38_{arm}_r{r}", f"genomic_30k_mgr38_{arm}", r, 30000, 15, 2, {**COORD, **extra}
            )
            assert " --val_cache " in c["cmd"]
            c["cmd"] = c["cmd"].replace(" --val_cache ", " --reference_legnet --val_cache ", 1)
            out.append(c)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        i = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells after index", i, "| registry", len(reg), "->", len(reg2))
        if DRY:
            print(new[0]["cmd"][:700])
            return
        shutil.copy(path, path + ".bak_mgr38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
