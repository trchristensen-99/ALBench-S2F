"""Oct-30: the published MPRA-LegNet with the study's context input at D = 100k, the size the earlier LegNet benchmarks (bench38_, 10k and 30k) did not cover. With context the published recipe is the strongest single
architecture we have at 30k (20-seed ensemble 0.863 to 0.869 genomic test against 0.857 for the 20-configuration menu), so the per-size comparison of the ensembles needs it at 100k too.

  bench38_ln_{e100,b256}_100k_c20_s<k>   10 seeds per schedule (2 per job, 5 jobs), canonical nested genomic 100k pool, input = insert + 20 real flank bases per side, shift +-10 over the real flank
                                         (e100: batch 1024, 100 epochs; b256: batch 256, 60 epochs; AdamW lr 0.01 one-cycle as the published recipe). Seeds are drawn once and stored in outputs/bench33/seeds100k.json.

Cells are placed before the recipe sweeps in the registry (they are on the critical path of the per-size comparison).

    python scripts/program_oct22/register_ln100.py [--dry]
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
NEST = "outputs/grid_pools/k562_nested"
SEED_FILE = f"{R}/outputs/bench33/seeds100k.json"
D, F = 100000, 20


def seeds():
    if os.path.exists(SEED_FILE):
        return json.load(open(SEED_FILE))
    s = [int.from_bytes(os.urandom(3), "big") for _ in range(10)]
    if not DRY:
        json.dump(s, open(SEED_FILE, "w"))
    return s


def all_cells():
    sd = seeds()
    out = []
    for cfg in ("e100", "b256"):
        odir = f"{OUT}/bench38_ln_{cfg}_100k_c{F}"
        for k in range(5):
            ids = [2 * k, 2 * k + 1]
            pairs = ",".join(f"{i}={sd[i]}" for i in ids)
            out.append(
                dict(
                    name=f"bench38_ln_{cfg}_100k_c{F}_s{k}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=odir,
                    env={},
                    needs=[f"{NEST}/genomic_d{D}.npz"],
                    cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg} --seeds {pairs} --ctx {F} && touch {R}/{odir}/COMPLETE_{k}of5",
                    done=dict(type="file", path=f"{odir}/COMPLETE_{k}of5"),
                )
            )
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        i = next(k for k, c in enumerate(reg) if c["name"].startswith("hp38_"))
        reg2 = reg[:i] + new + reg[i:]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells before index", i, "| registry", len(reg), "->", len(reg2))
        if DRY:
            print(new[0]["cmd"][:300])
            return
        shutil.copy(path, path + ".bak_ln100")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
