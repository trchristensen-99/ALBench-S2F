"""Oct-28 afternoon: the collaborator's model-based reservoirs through the same scaling design as the other strategies (menu v1, patience 16).

  d3_steps64    D3 (discrete diffusion for DNA), 64 sampling steps; his note: "use D3 uniform steps64 and steps256, try 64 first if compute is a bottleneck"
  d3_steps256   D3, 256 sampling steps
  evo2          Evo2-1B conditional regeneration of the K562 genomic pool (999,999 sequences)

For all three the oracle_v3 labels already existed: the earlier labelling of the same pools is exactly his seed-0 `nested_subsets` (checked set by set), so the nested pools
in k562_nested/ are written with `label_nested_pool.py --reuse` (no oracle call). Each: retrain cells at 10k (2 shards), 30k (4 shards) and 100k (6 shards, one model each, the first
six menu positions like the other 100k scale cells). Order: steps64 before steps256 before evo2's remaining cells. Registry order: 10k/30k directly after the last sc31 30k cell,
100k after the last sc31 100k cell. `--replace-gpa` removes the cells registered on a wrong guess (gpa_d3_student_*, guided diffusion arms) first. Registry writes are under flock.

  python scripts/program_oct22/register_yash.py [--dry] [--replace-gpa]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_oct30 as r30  # noqa: E402

R, NEST, rel = r30.R, r30.NEST, r30.rel
DRY = "--dry" in sys.argv
KOO = "/grid/koo/home/mundewadi"
ARMS = {
    "d3_steps64": f"{KOO}/ALBench-S2F/outputs/reservoir_cache/d3_uncond_steps64/pool_1M.npz",
    "d3_steps256": f"{KOO}/ALBench-S2F/outputs/reservoir_cache/d3_uncond_steps256/pool_1M.npz",
    "evo2": f"{KOO}/evo2_k562_uncond/samples/evo2_cond_regen_pool_1M.npz",
}
WRONG = ("pool32_gpa_", "sc32_gpa_")


def scale_cells(D):
    out = []
    for arm in ARMS:
        cells = r30.retrain(
            f"sc32_{arm}_{D // 1000}k",
            f"{NEST}/{arm}_d{D}.npz",
            D,
            f"outputs/menu_retrain_v0/sc32_{D // 1000}k_{arm}",
            r30.SHARDS[D],
        )
        if D >= 100000:
            for c in cells:
                c["env"]["MENU_MAX_N"] = "6"
        out += cells
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        if "--replace-gpa" in sys.argv:
            drop = [c["name"] for c in reg if c["name"].startswith(WRONG)]
            reg = [c for c in reg if not c["name"].startswith(WRONG)]
            print("removed", len(drop), "cells registered on the gpa guess")
        have = {c["name"] for c in reg}
        g_head = [c for c in scale_cells(10000) + scale_cells(30000) if c["name"] not in have]
        g_100 = [c for c in scale_cells(100000) if c["name"] not in have]
        last = lambda pat: max(i for i, c in enumerate(reg) if pat(c["name"]))  # noqa: E731
        i30 = last(
            lambda n: n.startswith("sc31_") and n.endswith(tuple(f"_30k_s{k}" for k in range(4)))
        )
        i100 = last(lambda n: n.startswith("sc31_") and "_100k_s" in n)
        assert i30 < i100
        reg2 = reg[: i30 + 1] + g_head + reg[i30 + 1 : i100 + 1] + g_100 + reg[i100 + 1 :]
        assert len(reg2) == len(reg) + len(g_head) + len(g_100) and len(
            {c["name"] for c in reg2}
        ) == len(reg2)
        print(
            "add",
            len(g_head),
            "cells after index",
            i30,
            "and",
            len(g_100),
            "after",
            i100,
            "| registry",
            len(reg),
            "->",
            len(reg2),
        )
        if DRY:
            return
        shutil.copy(path, path + ".bak_yash")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
