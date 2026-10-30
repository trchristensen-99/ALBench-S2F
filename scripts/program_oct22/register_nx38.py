"""Oct-30: three follow-ups chosen from the per-D procedure audit, all with the study's input standard (20 real flank bases per side, shift +-10), menu retrains with V0c validation,
patience 16, one seed per configuration, canonical nested genomic pools unless a replicate pool is named. Cells are paired with a control that has the same pool, menu and seeds.

  hp38_<arm>_genomic_<D>k   recipe at the two sizes that were never swept (menu v1c, all 20 configurations): arms base (the control), lrx05, lrx2, bs128, bs512, wd1e3 (+ wd1e2 at 10k)
  rc38_<arm>_<res>_<D>k     the single-pass reverse-complement augmentation (AL_RC_MODE=aug, arm rca) against the standard RC (arm c20) on the five standard cells: genomic 10k/30k/100k, random 30k, motif 30k;
                            controls: hp38_base (genomic 10k, 100k), mv1c_c20_30k (genomic 30k), rc38_c20 (random, motif 30k)
  v2m38_<res>_<D>k[_r<k>]   menu v2 (built from the standard-input searches by scripts/analysis/build_menu_v2.py; the cells wait for outputs/menus_oct30/V2.json) against menu v1c
  v1c38_genomic_<D>k_r<k>   menu v1c on the replicate pools r1, r2 at 10k and 30k (controls of the v2 cells there)

    python scripts/program_oct22/register_nx38.py [--dry]
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

R, NEST = r30.R, r30.NEST
DRY = "--dry" in sys.argv
REPS = f"{R}/outputs/grid_pools/k562_reps"
V1C = f"{R}/outputs/menus_oct29/V1c.json"
V2 = f"{R}/outputs/menus_oct30/V2.json"
STD = " --flank_ctx 20 --shift_max 10"
CAP = {10000: 15, 30000: 25, 100000: 60}
SHARDS = {10000: 4, 30000: 4, 100000: 10}
RECIPE = {
    "base": "",
    "lrx05": " --lr_scale 0.5",
    "lrx2": " --lr_scale 2",
    "bs128": " --bs_scale 0.5",
    "bs512": " --bs_scale 2",
    "wd1e3": " --wd_abs 1e-3",
}
RC_ENV = {"AL_RC_MODE": "aug"}


def cells(name, menu, pool, D, env=None, extra=""):
    return fb.cells(name, menu, pool, D, env or {}, STD + extra, shards=SHARDS[D], cap=CAP[D])


def all_cells():
    out = []
    for D in (10000, 100000):
        pool = f"{NEST}/genomic_d{D}.npz"
        arms = dict(RECIPE)
        if D == 10000:
            arms["wd1e2"] = " --wd_abs 1e-2"
        for arm, extra in arms.items():
            out += cells(f"hp38_{arm}_genomic_{D // 1000}k", V1C, pool, D, None, extra)
    for res, D in (
        ("genomic", 10000),
        ("genomic", 30000),
        ("genomic", 100000),
        ("random", 30000),
        ("motif_ct_enriched", 30000),
    ):
        pool = f"{NEST}/{res}_d{D}.npz"
        out += cells(f"rc38_rca_{res}_{D // 1000}k", V1C, pool, D, RC_ENV)
        if res != "genomic":
            out += cells(f"rc38_c20_{res}_{D // 1000}k", V1C, pool, D)
    for res, D in (
        ("genomic", 10000),
        ("genomic", 30000),
        ("genomic", 100000),
        ("random", 30000),
        ("motif_ct_enriched", 30000),
    ):
        out += cells(f"v2m38_{res}_{D // 1000}k", V2, f"{NEST}/{res}_d{D}.npz", D)
    for D in (10000, 30000):
        for r in (1, 2):
            pool = f"{REPS}/genomic_r{r}_d{D}.npz"
            out += cells(f"v2m38_genomic_{D // 1000}k_r{r}", V2, pool, D)
            out += cells(f"v1c38_genomic_{D // 1000}k_r{r}", V1C, pool, D)
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
            return
        shutil.copy(path, path + ".bak_nx38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
