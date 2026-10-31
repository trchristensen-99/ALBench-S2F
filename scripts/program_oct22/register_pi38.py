"""Oct-31: re-implementation of the five best architectures from the collaborator's open-world exploration (no LegNet in his prompt) in our harness, to see whether they reproduce his numbers in our
hands and whether they can be pushed further. Menus come from scripts/analysis/build_menu_pi.py (widths fitted to his stated parameter counts: 5.2M / 2.8M / 11M / 4.0M / 11.4M), two recipes per
architecture, study input standard (20 real flank bases per side, shift +-10), V0c validation, patience 16, canonical nested genomic pools.

  pi38_base_genomic_<D>k   the ten menu entries as built (3 seeds at 10k/30k, 2 at 100k): does an architecture found by open-world search match its stated test r at 30k (0.852-0.855) and how does it scale?
  tr38_<menu>_genomic_<D>k recipe transplant (scripts/analysis/transplant_recipe.py): every architecture of menu v2 (v2e100) or of the PI set (pie100, one entry per architecture) trained with the recipe of
                           the published MPRA-LegNet (AdamW, lr 0.01, wd 0.1, batch 1024, 100 epochs). At 100k the tuned LegNet beats the menu by 0.01 as an ensemble and 0.04 as a single model; this says whether the
                           gap is the recipe or the architecture. Control: v2m38_genomic_<D>k (same menu, own recipes)
  pi38_<arm>_genomic_30k   30k optimisation sweeps (2 seeds each): lrx05/lrx2 (peak learning rate), ep15 (1.5x epochs), bs128 (half batch), wd1e3 (weight decay 1e-3), x05/x2 (half/double parameter count)

    python scripts/program_oct22/register_pi38.py [--dry]
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
MENUS = f"{R}/outputs/menus_oct31"
PI, PI_X05, PI_X2, PI_E100, V2_E100 = (
    f"{MENUS}/{n}.json" for n in ("PI", "PI_x05", "PI_x2", "PI_e100", "V2_e100")
)
STD = " --flank_ctx 20 --shift_max 10"
# (menu, extra flags, seeds, shards, cap)
SWEEPS = {
    "lrx05": (PI, " --lr_scale 0.5", 2, 5, 40),
    "lrx2": (PI, " --lr_scale 2", 2, 5, 40),
    "ep15": (PI, " --epoch_scale 1.5", 2, 5, 60),
    "bs128": (PI, " --bs_scale 0.5", 2, 5, 60),
    "wd1e3": (PI, " --wd_abs 1e-3", 2, 5, 40),
    "x05": (PI_X05, "", 2, 5, 40),
    "x2": (PI_X2, "", 2, 5, 60),
}


def all_cells():
    out = []
    for D, seeds, shards, cap in ((30000, 3, 5, 40), (10000, 3, 5, 25), (100000, 2, 10, 120)):
        pool = f"{NEST}/genomic_d{D}.npz"
        out += fb.cells(f"pi38_base_genomic_{D // 1000}k", PI, pool, D, {}, STD, shards, cap, seeds)
    pool = f"{NEST}/genomic_d30000.npz"
    for arm, (menu, extra, seeds, shards, cap) in SWEEPS.items():
        out += fb.cells(
            f"pi38_{arm}_genomic_30k", menu, pool, 30000, {}, STD + extra, shards, cap, seeds
        )
    for name, menu, D, seeds, shards, cap in (
        ("v2e100", V2_E100, 10000, 1, 4, 25),
        ("v2e100", V2_E100, 30000, 1, 5, 40),
        ("v2e100", V2_E100, 100000, 1, 10, 120),
        ("pie100", PI_E100, 30000, 3, 5, 40),
        ("pie100", PI_E100, 100000, 2, 5, 120),
    ):
        pool = f"{NEST}/genomic_d{D}.npz"
        out += fb.cells(
            f"tr38_{name}_genomic_{D // 1000}k", menu, pool, D, {}, STD, shards, cap, seeds
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
            print(new[0]["cmd"][:400])
            return
        shutil.copy(path, path + ".bak_pi38")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
