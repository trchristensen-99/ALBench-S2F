"""Nov-2: schedule length and early stopping of the tuned MPRA-LegNet recipe, the gaps of the lr38_ sweeps (5 seeds per arm, the same five seeds as every lr38_ arm, same pools, input standard and control).

The sweeps already show how much of the tuned recipe is epochs (docs/TUNED_LEGNET.md): at 30k, wd 0.2: e150 0.8654, e300 0.8730, e500 0.8748, e800 0.8732; at 100k: e200 0.9254, e300 0.9261, e500 0.9269. Missing:

  30k   e200_wd2          200 epochs at the tuned weight decay
  10k   e200_wd4, e300_wd4   200 and 300 epochs at the tuned weight decay (e500_wd4 0.7508, e800_wd4 0.7557)
  early stopping on the validation Pearson with patience 16 inside the full one-cycle schedule, at the tuned recipe of each size:
  10k   e800_wd4_es16       30k   e500_wd2_es16       100k   e500_es16

Stored per-epoch validation curves of the 20-seed banks say what to expect (the validation Pearson plateaus between 20% and 40% of the schedule, before the learning rate anneals, and the best epoch is always in the last 10%):
patience 16 would stop at epoch ~114 of 500 at 30k and give up ~0.06 validation r. These cells measure it on the test sets.

    python scripts/program_oct22/register_lr39.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402
import register_lr38b as lb  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, ln.NEST, ln.F, ln.DRY
ARMS = {
    10000: ("e200_wd4", "e300_wd4", "e800_wd4_es16"),
    30000: ("e200_wd2", "e500_wd2_es16"),
    100000: ("e500_es16",),
}


def all_cells():
    pairs, out = ",".join(f"{i}={s}" for i, s in enumerate(lb.seeds())), []
    for D, arms in ARMS.items():
        for arm in arms:
            odir = f"{OUT}/bench38_ln_{arm}_{D // 1000}k_c{F}"
            out.append(
                dict(
                    name=f"lr39_{arm}_{D // 1000}k",
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
        shutil.copy(path, path + ".bak_lr39")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
