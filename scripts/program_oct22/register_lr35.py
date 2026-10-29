"""Oct-29 (afternoon): loop experiments with the flank-context harness (cells ctx35_, lng35_, mgr35_). The student sees the 200-bp insert plus 20 real plasmid bases on each side (--flank_ctx 20 --shift_max 10), the
proposer and the coder are told the input length (240) and asked for length-agnostic architectures, every architecture is checked at that length. Validation = V0c (--val_cache); patience stays at the loop's 8.

  ctx35_30k_r<k>      the standard loop (agent: proposer, coordinator picks 2 of 6, coder, evaluator; 10 rounds x 2 = 20 models; claude-sonnet-5-5) on genomic 30k replicate pools r1..r<--reps>, paired with the
                      Oct-25 coordinator arm on the same pools (without context): does the loop with context reach the collaborator's range, and does the menu it builds?
  lng35_<D>_r<k>      LONG runs (the collaborator's 200-model search got a much better single model with many more rounds): the single-call loop, 4 models per round, 50 rounds = 200 models, at 10k and 30k
                      on replicate pools r1, r2 (--long-reps). Cheap in LLM calls (one per round). Prefix analysis gives the curve of ensemble r against the number of models in one run.
  mgr35_<arm>_r<k>    the explore-then-exploit manager (arms c15 coordinator only, e15 schedule_ens) again, with context, 15 rounds x 2, pools r1..r3.

    python scripts/program_oct22/register_lr35.py [--reps 4] [--long-reps 2] [--mgr-reps 3] [--dry]
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


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


REPS_N, LONG_N, MGR_N = arg("--reps", 4), arg("--long-reps", 2), arg("--mgr-reps", 3)
S55 = "claude-sonnet-5-5"
OUT22 = "outputs/archsearch_reps_oct22"
V0C = "/grid/koo/home/shared/albench/data/val_V0c.npz"
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
CTX = "--flank_ctx 20 --shift_max 10"


def cell(name, d, r, D, rounds, per_round, agent_cfg=None):
    pool = f"{REPS}/genomic_r{r}_d{D}.npz"
    out = f"{OUT22}/{d}/rep{r}"
    loop = f"--loop agent --agent_cfg '{json.dumps(agent_cfg)}' " if agent_cfg is not None else ""
    return dict(
        name=name,
        gpus=["h100", "v100"],
        ext=[],
        out_dir=out,
        env={"LLM_MODEL": S55},
        needs=[rel(pool)],
        cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds {rounds} --per_round {per_round} --max_params 40000000 --base_cap_min 25 "
        f"--budget_hours {{BUDGET}} --guidance known_good --ensemble_aware {loop}{CTX} --val_cache {V0C} --resume",
        done=dict(type="file", path=f"{out}/COMPLETE"),
    )


def all_cells():
    coord = {"stages": ["propose", "coordinate", "code", "evaluate"], "ideas_per_round": 6}
    out = [
        cell(f"ctx35_30k_r{r}", "genomic_30k_ctx35", r, 30000, 10, 2, coord)
        for r in range(1, REPS_N + 1)
    ]
    out += [
        cell(f"lng35_{D // 1000}k_r{r}", f"genomic_{D // 1000}k_lng35", r, D, 50, 4)
        for D in (10000, 30000)
        for r in range(1, LONG_N + 1)
    ]
    for arm, extra in (("c15", {}), ("e15", {"manager": "schedule_ens", "explore_frac": 0.4})):
        out += [
            cell(
                f"mgr35_{arm}_r{r}", f"genomic_30k_mgr35_{arm}", r, 30000, 15, 2, {**coord, **extra}
            )
            for r in range(1, MGR_N + 1)
        ]
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
        for c in new:
            print(
                "  ",
                c["name"],
                "needs",
                c["needs"],
                "exists",
                all(os.path.exists(n) for n in c["needs"]),
            )
        if DRY:
            return
        shutil.copy(path, path + ".bak_lr35")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
