"""Nov-4: can a tuned LegNet be trained for much less compute without losing accuracy, and can the search loop find a faster one?

1. `bs30_*` (no LLM): the tuned 30k recipe is 500 epochs at batch 1024, which is only 14.6k optimiser steps, so "500 epochs" overstates how much training is really happening. Compute scales with epochs and
   steps with epochs/batch, and the two have never been separated. Seven arms at 5 seeds each on the canonical 30k pool, against the tuned `e500_wd2` control already trained with 20 seeds:

     e125_wd2            1/4 compute, 1/4 steps   the honest control: simply train less
     e250_b512_wd2       1/2 compute, same steps  does a smaller batch buy back the lost compute?
     e125_b256_wd2       1/4 compute, same steps
     e500_b256_wd2       same compute, 4x steps   is the tuned recipe step-starved rather than epoch-starved?
     e250_b512_lr20_wd2, e125_b256_lr20_wd2, e125_b256_lr30_wd2   the cheap points at a higher peak learning rate

   About 3 GPU-hours. If a quarter-compute arm matches the control, every bank in the scaling program gets four times cheaper, which matters most at 300k (66 min a seed today).

2. `fast41_30k_r<k>` (LLM): the same AutoResearch loop restricted to LegNet modifications (`variants` manager), but with the per-model wall cap cut from 40 minutes to 6. Both the proposer and the coder are
   told the cap, so the objective becomes "the best LegNet-family model that trains in 6 minutes" rather than "the best model". Read-out: validation r against the 8.3 minutes a tuned seed takes, and whether
   the winning designs differ in kind from the ones the uncapped loop finds. Tuned recipe is NOT imposed: the recipe is what we want it to search over.

    python scripts/program_oct22/register_nov4h.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402
import register_nov2b as nb  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, "outputs/grid_pools/k562_nested", ln.F, ln.DRY
D = 30000
ARMS = (
    "e125_wd2",
    "e250_b512_wd2",
    "e125_b256_wd2",
    "e500_b256_wd2",
    "e250_b512_lr20_wd2",
    "e125_b256_lr20_wd2",
    "e125_b256_lr30_wd2",
)
SEED_FILE = f"{R}/outputs/bench33/seeds_nov4h.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def bs_cells():
    out, pool = [], f"{NEST}/genomic_d{D}.npz"
    for cfg in ARMS:
        odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}"
        sd = seeds(cfg, 5)
        for j in range(5):
            out.append(
                dict(
                    name=f"bs30_{cfg}_s{j}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=odir,
                    env={},
                    needs=[pool],
                    cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg} --seeds {j}={sd[j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
                    done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
                )
            )
    return out


def fast_cells():
    reps, _, budget = nb.RR_SIZE
    out = []
    for r in range(1, reps + 1):
        c = nb.loop_cell(
            f"fast41_30k_r{r}", "genomic_30k_fast41", r, D, 12, 6, budget,
            {**nb.RR, "manager": "variants", "explore_frac": 0.3},
        )  # fmt: skip
        c["env"] = {**c["env"], "AL_RC_MODE": "aug", "AL_PATIENCE": "9999"}
        c["gpus"] = ["v100"]
        out.append(c)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in bs_cells() + fast_cells() if c["name"] not in have]
        # the cheap-schedule sweep decides how expensive every later bank is, so it goes before the scaling cells
        i = next((k for k, c in enumerate(reg) if c["name"].startswith("sc44_")), len(reg))
        reg2 = reg[:i] + new + reg[i:]
        assert len({c["name"] for c in reg2}) == len(reg2) == len(reg) + len(new)
        print(f"add {len(new)} cells at index {i} | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print(new[0]["cmd"][:260])
            print([c["name"] for c in new if c["name"].startswith("fast41")])
            print(next(c["cmd"] for c in new if c["name"].startswith("fast41"))[:420])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov4h")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
