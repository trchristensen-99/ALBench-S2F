"""Oct-29 (evening): where is the optimum of the input context and of the shift augmentation, at 10k, 30k and 100k, on one reservoir per family? (cells sw38_)

One-factor sweeps around the current standard (20 real flank bases each side, shift +-10), with the first six menu-v1 configurations that take wider input (menu indices 1 to 6; index 0 fails to build at any
other input length and costs seconds), one seed, V0c validation, patience 16, nested pools, paired by configuration with the no-context arm of the same reservoir and size.

  context sweep (shift +-10): c12 c16 c20 c24 c28 c32 c40 c48 c60 real bases per side (input 224 ... 320 bp; every length is a multiple of 8)
  shift sweep (20 bases):     s0 s5 s15 s20 s30 (+-S per batch), orc = the oracle's schedule (+-15 per sample on half of the samples)
  rca                         20 bases, +-10, reverse-complement augmentation as in the oracle (one pass, flip p 0.5) instead of the forward + reverse-complement loss
  base                        no context (menu as it was)

Reservoirs, one per family: genomic, random, motif_ct_enriched, evoaug_paper2023, evo2. Sizes: 10k and 30k with every arm; 100k (the cells that are expensive) with genomic, random and motif_ct_enriched, four configurations
and the arms base, c16, c20, c28, c40, s5, s15, orc.

  sw38_<D>k_<reservoir>_<arm>   one cell each (a shard of the menu retrain, all configurations)

    python scripts/program_oct22/register_sw38.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_fx35b as fb  # noqa: E402
import register_oct30 as r30  # noqa: E402

R, NEST, V1 = r30.R, r30.NEST, r30.V1
DRY = "--dry" in sys.argv
RES = ["genomic", "random", "motif_ct_enriched", "evoaug_paper2023", "evo2"]
CAP = {10000: 15, 30000: 25, 100000: 60}
ARMS = {"base": ("", {})}
for F in (12, 16, 20, 24, 28, 32, 40, 48, 60):
    ARMS[f"c{F}"] = (f" --flank_ctx {F} --shift_max 10", {})
for S in (0, 5, 15, 20, 30):
    ARMS[f"s{S}"] = (f" --flank_ctx 20 --shift_max {S}", {})
ARMS["orc"] = (" --flank_ctx 20 --shift_max 15 --shift_per_sample --shift_prob 0.5", {})
ARMS["rca"] = (" --flank_ctx 20 --shift_max 10", {"AL_RC_MODE": "aug"})
ARMS_100K = ["base", "c16", "c20", "c28", "c40", "s5", "s15", "orc"]


def all_cells():
    out = []
    for D in (10000, 30000, 100000):
        res_d = RES if D < 100000 else RES[:3]
        arms = ARMS if D < 100000 else {k: ARMS[k] for k in ARMS_100K}
        n_arch = 7 if D < 100000 else 5  # MENU_MAX_N: indices 0..n-1 (0 cannot build with context)
        for res in res_d:
            pool = f"{NEST}/{res}_d{D}.npz"
            for arm, (extra, env) in arms.items():
                cap = CAP[D] * (1.4 if arm in ("c48", "c60") else 1.0)
                out += fb.cells(
                    f"sw38_{D // 1000}k_{res}_{arm}",
                    V1,
                    pool,
                    D,
                    {"MENU_MAX_N": str(n_arch), **env},
                    extra,
                    shards=1,
                    cap=cap,
                )
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
        missing = [n for c in new for n in c["needs"] if not os.path.exists(n)]
        print("missing needs:", sorted(set(missing))[:5])
        if DRY:
            return
        shutil.copy(path, path + ".bak_sw38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
