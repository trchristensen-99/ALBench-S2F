"""Nov-1: deepen the tuned-LegNet seed bank to 20 seeds per training size (the earlier agreement was at least 16 to 20 independent seeds per size; the lr38_ sweeps left 5 per arm). The per-size recipe is the best
arm of the recipe sweeps: 10k e800_wd4 (800 epochs, weight decay 0.4), 30k e500_wd2 (500 epochs, weight decay 0.2), 100k e500 (500 epochs, weight decay 0.1). Same pools, input standard, V0c selection and output
directories as the sweep arms; 15 new seeds (indices 5 to 19) drawn once into outputs/bench33/seeds_tuned.json, 5 per job.

  lr38s_<D>k_j<0,1,2>   D = 10k, 30k, 100k

    python scripts/program_oct22/register_lr38s.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, ln.NEST, ln.F, ln.DRY
SEED_FILE = f"{R}/outputs/bench33/seeds_tuned.json"
ARM = {10000: "e800_wd4", 30000: "e500_wd2", 100000: "e500"}


def seeds():
    if os.path.exists(SEED_FILE):
        return json.load(open(SEED_FILE))
    s = [int.from_bytes(os.urandom(3), "big") for _ in range(15)]
    if not DRY:
        json.dump(s, open(SEED_FILE, "w"))
    return s


def all_cells():
    sd, out = seeds(), []
    for D, arm in ARM.items():
        odir = f"{OUT}/bench38_ln_{arm}_{D // 1000}k_c{F}"
        for j in range(3):
            ids = range(5 + 5 * j, 10 + 5 * j)
            pairs = ",".join(f"{i}={sd[i - 5]}" for i in ids)
            out.append(
                dict(
                    name=f"lr38s_{D // 1000}k_j{j}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=odir,
                    env={},
                    needs=[f"{NEST}/genomic_d{D}.npz"],
                    cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {arm} --seeds {pairs} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
                    done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
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
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells at the end | registry", len(reg), "->", len(reg2))
        if DRY:
            print(new[0]["cmd"][:330] if new else "nothing new")
            return
        shutil.copy(path, path + ".bak_lr38s")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
