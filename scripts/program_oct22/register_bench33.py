"""Oct-28 afternoon: register the benchmark of our genomic ensembles against MPRA-LegNet and AlphaGenome-FT-MPRA students (cells bench33_*).

  --phase probe              12 LegNet cells (3 configurations x {10k, 30k} x 2 cells of 10 seeds) and the first AlphaGenome seed (index 0) of both Stage 2
                             recipes at 30k and 10k, so that the recipe can be chosen on V0c validation before the remaining seeds are trained
  --phase rest --recipe R    AlphaGenome seeds 1..4 at 30k and 10k for the chosen recipe R (oracle | winner)

Seeds are drawn once from os.urandom (outputs/bench33/seeds.json; Peter asked for fresh randomness instead of pinned seeds) and baked into each command line
and recorded in every meta file. LegNet cells may run on V100 (the code is the same fp16-autocast path on every GPU); AlphaGenome cells are H100 only and the
dispatcher never puts a cell without menu_retrain.py in its command on a MIG slice. Registry order: directly after the last sc32_ 30k cell.
Registry writes are under flock on scratchpad/.registry.lock; backup scratchpad/dispatch_registry.json.bak_bench33.

  python scripts/program_oct22/register_bench33.py --phase probe [--dry]
"""

import argparse
import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import R  # noqa: E402

os.chdir(R)
OUT = "outputs/menu_retrain_v0"
SEEDS = f"{R}/outputs/bench33/seeds.json"
DS = (30000, 10000)
CFGS = ("pub", "e100", "b256")
RECIPES = ("oracle", "winner")
N_LN, N_AG = 20, 5


def seeds() -> dict:
    if os.path.exists(SEEDS):
        return json.load(open(SEEDS))
    draw = lambda n: [int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF for _ in range(n)]  # noqa: E731
    s = {
        "ln": {str(D): draw(N_LN) for D in DS},
        "ag": {f"{D}_{r}": draw(N_AG) for D in DS for r in RECIPES},
        "drawn_from": "os.urandom",
    }
    os.makedirs(os.path.dirname(SEEDS), exist_ok=True)
    json.dump(s, open(SEEDS, "w"), indent=1)
    return s


def ln_cells(sd):
    cells = []
    for D in DS:
        for cfg in CFGS:
            out = f"{OUT}/bench33_ln_{cfg}_{D // 1000}k"
            for k in range(2):
                pairs = ",".join(f"{i}={sd['ln'][str(D)][i]}" for i in range(10 * k, 10 * k + 10))
                cells.append(
                    dict(
                        name=f"bench33_ln_{cfg}_{D // 1000}k_s{k}",
                        gpus=["h100", "v100"],
                        ext=[],
                        out_dir=out,
                        env={},
                        needs=[f"outputs/grid_pools/k562_nested/genomic_d{D}.npz"],
                        cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg} --seeds {pairs} && touch {R}/{out}/COMPLETE_{k}of2",
                        done=dict(type="file", path=f"{out}/COMPLETE_{k}of2"),
                    )
                )
    return cells


def ag_cells(sd, recipe, indices, Ds=DS):
    cells = []
    for D in Ds:
        for i in indices:
            cells.append(
                dict(
                    name=f"bench33_ag_{recipe}_{D // 1000}k_s{i}",
                    gpus=["h100"],
                    ext=[],
                    out_dir=f"{OUT}/bench33_agS2_{recipe}_{D // 1000}k",
                    env={},
                    needs=[f"outputs/grid_pools/k562_nested/genomic_d{D}.npz"],
                    cmd=f"uv run --no-sync python scripts/program_oct22/bench33_ag.py --D {D} --recipe {recipe} --seed {sd['ag'][f'{D}_{recipe}'][i]} --seed-index {i}",
                    done=dict(
                        type="file",
                        path=f"{OUT}/bench33_agS2_{recipe}_{D // 1000}k/r{i:02d}_menu_000_meta.json",
                    ),
                )
            )
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["probe", "rest"], required=True)
    ap.add_argument("--recipe", choices=RECIPES)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sd = seeds()
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        if a.phase == "probe":
            new = ag_cells(sd, "oracle", [0]) + ag_cells(sd, "winner", [0]) + ln_cells(sd)
        else:
            assert a.recipe, "--recipe is required for --phase rest"
            new = ag_cells(sd, a.recipe, range(1, N_AG))
        new = [c for c in new if c["name"] not in have]
        last = max(
            i for i, c in enumerate(reg) if c["name"].startswith("sc32_") and "_30k_s" in c["name"]
        )
        reg2 = reg[: last + 1] + new + reg[last + 1 :]
        assert len(reg2) == len(reg) + len(new) and len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells after index {last} | registry {len(reg)} -> {len(reg2)}")
        for c in new:
            print("  ", c["name"])
        if a.dry:
            return
        shutil.copy(path, path + ".bak_bench33")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
