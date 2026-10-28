"""Oct-28 evening: does an explore-then-exploit manager in the multi-agent AutoResearch loop give a more diverse and better ensemble? (cells mgr34_)

Setting = the loop-control arms of Oct 25 (genomic 30k, replicate pools r1..r6, claude-sonnet-5-5, validation-only feedback, guidance + ensemble-aware prompt, coordinator picks 2 of 6 ideas) but 15 rounds
(30 trained models) instead of 10:
  c15  coordinator only (the control, no manager)
  s15  + manager "schedule": the first 6 rounds explore (coordinator and proposer are told to widen architecture classes and training recipes, the coordinator sees a coverage table), then 9 exploit rounds
  e15  + manager "schedule_ens": as s15, and the coordinator also sees the validation stack of everything trained so far (which models carry weight, correlation with the stack)
Compare with the existing 10-round arms at the first 20 models (default single call, coordinator only), and c15 against s15 / e15 at 30 models; outcome measures: stack r on the three test sets, spread of
architecture classes and recipes inside a run (analysis to write), pairwise prediction correlation of the run's models.

  python scripts/program_oct22/register_mgr34.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import REPS, R  # noqa: E402

os.chdir(R)
DRY = "--dry" in sys.argv
S55 = "claude-sonnet-5-5"
OUT22 = "outputs/archsearch_reps_oct22"
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
ARMS = {
    "c15": ("genomic_30k_mgr_c15", {}),
    "s15": ("genomic_30k_mgr_s15", {"manager": "schedule", "explore_frac": 0.4}),
    "e15": ("genomic_30k_mgr_e15", {"manager": "schedule_ens", "explore_frac": 0.4}),
}


def cell(arm, r):
    d, extra = ARMS[arm]
    pool = f"{REPS}/genomic_r{r}_d30000.npz"
    out = f"{OUT22}/{d}/rep{r}"
    cfg = {"stages": ["propose", "coordinate", "code", "evaluate"], "ideas_per_round": 6, **extra}
    return dict(
        name=f"mgr34_{arm}_r{r}",
        gpus=["h100", "v100"],
        ext=[],
        out_dir=out,
        env={"LLM_MODEL": S55},
        needs=[rel(pool)],
        cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D 30000 --out_dir {R}/{out} --rounds 15 --per_round 2 --max_params 40000000 --base_cap_min 25 "
        f"--budget_hours {{BUDGET}} --guidance known_good --ensemble_aware --loop agent --agent_cfg '{json.dumps(cfg)}' --resume",
        done=dict(type="file", path=f"{out}/COMPLETE"),
    )


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [cell(a, r) for r in range(1, 7) for a in ARMS if f"mgr34_{a}_r{r}" not in have]
        i = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len(reg2) == len(reg) + len(new) and len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells after index", i, "| registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_mgr34")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
