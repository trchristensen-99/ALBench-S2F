"""Oct-28 afternoon: the collaborator's three model-based reservoirs through the same scaling design as the other strategies (menu v1, patience 16).

  evo2                         Evo2-1B conditional regeneration of the K562 genomic pool (999,999 sequences); labels reused from the earlier labelling of
                               the same seed-0 subsets (the pool in k562_cumulative/evo2_d*.npz is identical to nested_subsets seed 0)
  gpa_d3_student_mean          D3 diffusion guided by the 10-member student's mean activity (1,000,016 sequences)
  gpa_d3_student_uncertainty   D3 diffusion guided by student-ensemble variance (1,000,000 sequences)

Each: labelling cell (the two diffusion pools; oracle_v3), then retrain cells at 10k (2 shards), 30k (4 shards) and 100k (6 shards, one model each, the
first six menu positions like the other 100k scale cells). Registry order: labelling and 10k/30k directly after the last sc31 30k cell, 100k after the last
sc31 100k cell. Registry writes are under flock so that two registrations cannot clobber each other.

  python scripts/program_oct22/register_yash.py [--dry]
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
    "evo2": f"{KOO}/evo2_k562_uncond/samples/evo2_cond_regen_pool_1M.npz",
    "gpa_d3_student_mean": f"{KOO}/ALBench-S2F/outputs/reservoir_cache/gpa_d3_student_mean/pool.npz",
    "gpa_d3_student_uncertainty": f"{KOO}/ALBench-S2F/outputs/reservoir_cache/gpa_d3_student_uncertainty/pool.npz",
}


def label_cells():
    out = []
    for arm in ("gpa_d3_student_mean", "gpa_d3_student_uncertainty"):
        out.append(
            dict(
                name=f"pool32_{arm}",
                gpus=["h100"],
                ext=[],
                out_dir=rel(NEST),
                env={"AG_S2_ORACLE_DIR": f"{R}/outputs/oracle_v3/s2"},
                cmd=f"uv run --no-sync python scripts/program_oct22/label_nested_pool.py --src {ARMS[arm]} --arm {arm} --seed 0",
                done=dict(type="file", path=rel(f"{NEST}/{arm}_d100000.npz")),
            )
        )
    return out


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
        have = {c["name"] for c in reg}
        g_head = [
            c
            for c in label_cells() + scale_cells(10000) + scale_cells(30000)
            if c["name"] not in have
        ]
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
