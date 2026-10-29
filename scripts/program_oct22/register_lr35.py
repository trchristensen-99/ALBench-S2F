"""Oct-29: loop experiments with the settled harness (cells mgr35_, long35_). Register AFTER the pk35_/fx35_ results, so the LLM budget is spent on the harness we will actually use.

mgr35_<arm>_r<k>   the explore-then-exploit manager comparison of mgr34_ again (genomic 30k replicate pools, claude-sonnet-5-5, validation-only feedback, guidance + ensemble-aware prompt, 15 rounds x 2 models)
                   with the harness switches on: AL_FLANK_SHIFT=1 and, with --ema, AL_EMA=0.9995 + AL_BN_RECALC=1. Arms c15 / s15 / e15 as in register_mgr34.py; pools r1..r<--reps>.
long35_r<k>        one LONG run per replicate pool (the collaborator's 200-model search got a much better single model with many more rounds): 50 rounds x 4 models, manager schedule_ens
                   (explore the first 40%), same switches. Resumable; one cell per pool.

  python scripts/program_oct22/register_lr35.py [--ema] [--reps 4] [--long 3] [--dry]
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
EMA = "--ema" in sys.argv
REPS_N = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 4
LONG_N = int(sys.argv[sys.argv.index("--long") + 1]) if "--long" in sys.argv else 3
S55 = "claude-sonnet-5-5"
OUT22 = "outputs/archsearch_reps_oct22"
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
ARMS = {
    "c15": ("genomic_30k_mgr35_c15", {}),
    "s15": ("genomic_30k_mgr35_s15", {"manager": "schedule", "explore_frac": 0.4}),
    "e15": ("genomic_30k_mgr35_e15", {"manager": "schedule_ens", "explore_frac": 0.4}),
}


def env():
    e = {"LLM_MODEL": S55, "AL_FLANK_SHIFT": "1"}
    if EMA:
        e.update({"AL_EMA": "0.9995", "AL_BN_RECALC": "1"})
    return e


def cell(name, d, r, rounds, per_round, extra):
    pool = f"{REPS}/genomic_r{r}_d30000.npz"
    out = f"{OUT22}/{d}/rep{r}"
    cfg = {"stages": ["propose", "coordinate", "code", "evaluate"], "ideas_per_round": 6, **extra}
    return dict(
        name=name,
        gpus=["h100", "v100"],
        ext=[],
        out_dir=out,
        env=env(),
        needs=[rel(pool)],
        cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D 30000 --out_dir {R}/{out} --rounds {rounds} --per_round {per_round} --max_params 40000000 --base_cap_min 25 "
        f"--budget_hours {{BUDGET}} --guidance known_good --ensemble_aware --loop agent --agent_cfg '{json.dumps(cfg)}' --resume",
        done=dict(type="file", path=f"{out}/COMPLETE"),
    )


def all_cells():
    out = [
        cell(f"mgr35_{a}_r{r}", ARMS[a][0], r, 15, 2, ARMS[a][1])
        for r in range(1, REPS_N + 1)
        for a in ARMS
    ]
    out += [
        cell(
            f"long35_r{r}",
            "genomic_30k_long35",
            r,
            50,
            4,
            {"manager": "schedule_ens", "explore_frac": 0.4},
        )
        for r in range(1, LONG_N + 1)
    ]
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
        print("add", len(new), "cells | env", env(), "| registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_lr35")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
