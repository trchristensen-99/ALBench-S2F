"""Oct-29 (evening): the LegNet and AlphaGenome-FT baselines again with the context input the students now get (cells bench38_).

bench38_ln_<cfg>_<D>k_c<F>_s<k>    published MPRA-LegNet (schedules pub, e100, b256), 20 seeds as in bench33 (same seeds), input = 200-bp insert + F real flank bases each side (F = 20 and 28), shift +-10 over the real flank,
                                   at 10k and 30k on the canonical genomic pool.
bench38_ag_oracle_<D>k_w256_s<i>   AlphaGenome-FT-MPRA, the oracle recipe (Stage 1 then Stage 2), 10 seeds as in bench33 (same seeds), input 256 bp = 200-bp insert + 28 real flank bases each side (AlphaGenome reads
                                   multiples of 128 bp; the oracle itself uses 600 bp) at 10k and 30k.
Student-type check (does the ranking of reservoirs depend on the student?): the tuned LegNet (b256, F = 20, 5 seeds) and AlphaGenome-FT (256 bp, 2 seeds) on the random and CT-enriched motif pools at 30k:
bench38_ln_b256_30k_c20_<res>, bench38_ag_oracle_30k_w256_<res>.

  python scripts/program_oct22/register_bl38.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import R  # noqa: E402

os.chdir(R)
DRY = "--dry" in sys.argv
OUT = "outputs/menu_retrain_v0"
SEEDS = json.load(open(f"{R}/outputs/bench33/seeds.json"))
NEST = "outputs/grid_pools/k562_nested"
OTHER = ["random", "motif_ct_enriched"]


def ln_cell(cfg, D, F, k, res=None, seeds=None):
    pool = f"{NEST}/{res or 'genomic'}_d{D}.npz"
    out = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}" + (f"_{res}" if res else "")
    ids = seeds if seeds is not None else list(range(10 * k, 10 * k + 10))
    pairs = ",".join(f"{i}={SEEDS['ln'][str(D)][i]}" for i in ids)
    n = 2 if seeds is None else 1
    pool_arg = f" --pool {pool}" if res else ""
    return dict(
        name=f"{os.path.basename(out)}_s{k}",
        gpus=["h100", "v100"],
        ext=[],
        out_dir=out,
        env={},
        needs=[pool],
        cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg} --seeds {pairs} --ctx {F}{pool_arg} && touch {R}/{out}/COMPLETE_{k}of{n}",
        done=dict(type="file", path=f"{out}/COMPLETE_{k}of{n}"),
    )


def ag_cell(D, i, res=None):
    pool = f"{NEST}/{res or 'genomic'}_d{D}.npz"
    out = f"{OUT}/bench33_agS2_oracle_{D // 1000}k_w256" + (f"_{res}" if res else "")
    pool_arg = f" --pool {pool}" if res else ""
    return dict(
        name=f"bench38_ag_oracle_{D // 1000}k_w256" + (f"_{res}" if res else "") + f"_s{i}",
        gpus=["h100"],
        ext=[],
        out_dir=out,
        env={},
        needs=[pool],
        cmd=f"uv run --no-sync python scripts/program_oct22/bench33_ag.py --D {D} --recipe oracle --seed {SEEDS['ag'][f'{D}_oracle'][i]} --seed-index {i} --window 256{pool_arg}",
        done=dict(type="file", path=f"{out}/r{i:02d}_menu_000_meta.json"),
    )


def all_cells():
    out = []
    for D in (10000, 30000):
        for cfg in ("pub", "e100", "b256"):
            for F in (20, 28):
                out += [ln_cell(cfg, D, F, k) for k in range(2)]
        out += [ag_cell(D, i) for i in range(10)]
    for res in OTHER:
        out.append(ln_cell("b256", 30000, 20, 0, res=res, seeds=list(range(5))))
        out += [ag_cell(30000, i, res=res) for i in range(2)]
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        i = max(i for i, c in enumerate(reg) if c["name"].startswith("fx36_c20_tta8_30k"))
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells after index", i, "| registry", len(reg), "->", len(reg2))
        if DRY:
            for c in new[:3] + new[-3:]:
                print("  ", c["name"], "|", c["cmd"][-150:])
            return
        shutil.copy(path, path + ".bak_bl38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
