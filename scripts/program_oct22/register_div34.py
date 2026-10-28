"""Oct-28 evening: experiments for architecture and recipe diversity (cells hp34_, fl34_, ds34_), all on the canonical nested GENOMIC pool, one training seed per configuration, patience 16, and validated on the
true V0c labels (--val_cache), which the earlier menu retrains were not (their labels.npz carry the earlier oracle's labels, r = 0.983 with V0c).

  hp34_<arm>_30k   menu v1 (20 configurations) at 30k with one recipe change applied to every configuration:
                   base (no change: the paired control, V0c validation) | lrx05, lrx2 (peak learning rate x0.5, x2) | bs128, bs512 (batch size x0.5, x2) | wd1e3 (weight decay 1e-3) |
                   adamw (AdamW, lr 1e-3, weight decay 1e-2: the recipe the proposer picks most often when it leaves Muon)
  fl34_flank_30k   the base arm with the shift augmentation taking real plasmid flank bases in instead of zeros (AL_FLANK_SHIFT=1; paired with hp34_base_30k: same seeds, same menu)
  hp34_base_10k, hp34_base_100k   menu v1 at 10k and 100k with V0c validation (controls for the next two)
  ds34_m10_10k, ds34_m100_100k    menus built from the AutoResearch runs done AT 10k (15 runs: genomic, random, motif CT-enriched, EvoAug, Evo2) and AT 100k (13 runs: genomic, GC-matched, motif syntax core,
                   mutagenesis) with the same recipe as menu v1 (top of every run, then the best of the rest), retrained on the genomic pool of that size: does searching at the target size beat the 30k menu?

Cells go directly after the last sc32_ 100k cell. Registry writes under flock; backup dispatch_registry.json.bak_div34.

  python scripts/program_oct22/register_div34.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_oct30 as r30  # noqa: E402

R, NEST, rel = r30.R, r30.NEST, r30.rel
DRY = "--dry" in sys.argv
OUT = "outputs/menu_retrain_v0"
V0C = "/grid/koo/home/shared/albench/data/val_V0c.npz"
V1 = r30.V1
M10, M100 = f"{R}/outputs/menus_oct28/M10.json", f"{R}/outputs/menus_oct28/M100.json"
CAP = {10000: 15, 30000: 25, 100000: 60}
SHARDS = {10000: 2, 30000: 4, 100000: 6}
ARMS30 = {
    "base": "",
    "lrx05": " --lr_scale 0.5",
    "lrx2": " --lr_scale 2",
    "bs128": " --bs_scale 0.5",
    "bs512": " --bs_scale 2",
    "wd1e3": " --wd_abs 1e-3",
    "adamw": " --optimizer adamw --lr_abs 1e-3 --wd_abs 1e-2",
}


def cells(name, menu, D, extra="", env_extra=None):
    out_dir = f"{OUT}/{name}"
    pool = f"{NEST}/genomic_d{D}.npz"
    env = {"AL_PATIENCE": "16", "MENU_MAX_FWD_MS": "1000000000", **(env_extra or {})}
    return [
        dict(
            name=f"{name}_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=out_dir,
            env=dict(env),
            needs=[rel(pool), rel(menu)],
            cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {menu} --reservoir_cache {pool} --D {D} --out_dir {R}/{out_dir} --shard {k}/{SHARDS[D]} --base_cap_min {CAP[D]} --val_cache {V0C}{extra}",
            done=dict(type="file", path=f"{out_dir}/COMPLETE_{k}of{SHARDS[D]}"),
        )
        for k in range(SHARDS[D])
    ]


def all_cells():
    out = []
    for arm, extra in ARMS30.items():
        out += cells(f"hp34_{arm}_30k", V1, 30000, extra)
    out += cells("fl34_flank_30k", V1, 30000, "", {"AL_FLANK_SHIFT": "1"})
    out += cells("hp34_base_10k", V1, 10000)
    out += cells("ds34_m10_10k", M10, 10000)
    out += cells("hp34_base_100k", V1, 100000)
    out += cells("ds34_m100_100k", M100, 100000)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        i = max(
            i for i, c in enumerate(reg) if c["name"].startswith("sc32_") and "_100k_s" in c["name"]
        )
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len(reg2) == len(reg) + len(new) and len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells after index", i, "| registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_div34")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
