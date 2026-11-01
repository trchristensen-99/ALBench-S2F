"""Oct-31: can the published MPRA-LegNet recipe (e100) be improved, per training size? The tuned LegNet is the strongest student we have at 30k and 100k and the reference every AutoResearch run is measured
against, so its recipe deserves the same sweep the menu got. One knob changes per arm, everything else is the e100 recipe (AdamW, OneCycle, batch 1024, 100 epochs, lr 0.01, wd 0.1) with the study's input
standard (20 real flank bases, shift +-10), canonical nested genomic pools, V0c selection:

  e100_lr5 / e100_lr20   peak learning rate 0.005 / 0.02       e150          150 epochs
  e100_b2048             batch 2048                              e100_wd2 / e100_wd05   weight decay 0.2 / 0.05

  lr38_<arm>_<D>k   5 seeds per job and arm (the same five seeds in every arm), D = 10k, 30k, 100k; results land in bench38_ln_<arm>_<D>k_c20 next to the 20-seed e100 control

    python scripts/program_oct22/register_lr38.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, ln.NEST, ln.F, ln.DRY
SEED_FILE = f"{R}/outputs/bench33/seeds_recipe.json"
ARMS = ("e100_lr5", "e100_lr20", "e150", "e100_b2048", "e100_wd2", "e100_wd05")
SIZES = (10000, 30000, 100000)


def seeds():
    if os.path.exists(SEED_FILE):
        return json.load(open(SEED_FILE))
    s = [int.from_bytes(os.urandom(3), "big") for _ in range(5)]
    if not DRY:
        json.dump(s, open(SEED_FILE, "w"))
    return s


def all_cells():
    pairs, out = ",".join(f"{i}={s}" for i, s in enumerate(seeds())), []
    for D in SIZES:
        for arm in ARMS:
            odir = f"{OUT}/bench38_ln_{arm}_{D // 1000}k_c{F}"
            out.append(
                dict(
                    name=f"lr38_{arm}_{D // 1000}k",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=odir,
                    env={},
                    needs=[f"{NEST}/genomic_d{D}.npz"],
                    cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {arm} --seeds {pairs} --ctx {F} && touch {R}/{odir}/COMPLETE_0of1",
                    done=dict(type="file", path=f"{odir}/COMPLETE_0of1"),
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
        shutil.copy(path, path + ".bak_lr38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
