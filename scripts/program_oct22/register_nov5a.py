"""Nov-5: tuned-LegNet banks for the genome-sourced reservoirs, the 300k recipe extension, and a cheaper route to the 10k recipe.

1. `gs45_<reservoir>_<D>` (110 cells): 10 seeds of the per-size tuned LegNet on each pool normalised by `scripts/normalise_new_pools.py`, at every size the pool can reach:

     accessible_tiles_k562   10k 30k   all K562 open chromatin, tiled (60,689 windows)
     accessible_k562_only    10k 30k   open in K562 and NOT HepG2 (53,839) -- the cell-type-specific arm
     accessible_shared       10k       open in both (28,330)
     accessible_hepg2_only   10k 30k 100k   open in HepG2 only (116,817) -- the control that separates "open chromatin" from "open in the assayed cell type"
     hg38_random             10k 30k 100k   uniformly random reference windows -- the no-selection control

2. `r300b_*` (15 cells): the 300k sweep said e300_wd05 (0.9508) beats e300/e200/e100 and was still rising, and weight decay has fallen at every size step, so test 500 epochs and wd 0.025 before freezing 300k.

3. `t10_*` (40 cells): at 10k, 1,200 epochs beat 800 but costs 1,200 passes. At batch 1024 a 10k epoch is only ~10 optimiser steps, so b512/b256/b128 reach the same ~12k steps in 600/300/150 epochs. The 30k sweep
   found smaller batches lose at matched steps, but 30k is not sample-starved the way 10k is, so it is worth one test. Also 20 seeds of e1200_wd4 to confirm the +0.0033 over e800_wd4 (it was only ~1.2 SE).

    python scripts/program_oct22/register_nov5a.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, "outputs/grid_pools/k562_nested", ln.F, ln.DRY
TUNED = {10000: "e800_wd4", 30000: "e500_wd2", 100000: "e500"}
GS = {
    "accessible_tiles_k562": (10000, 30000),
    "accessible_k562_only": (10000, 30000),
    "accessible_shared": (10000,),
    "accessible_hepg2_only": (10000, 30000, 100000),
    "hg38_random": (10000, 30000, 100000),
}
SEED_FILE = f"{R}/outputs/bench33/seeds_nov5a.json"
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def cells(name, D, cfg, pool, n, gpus, tag):
    odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}{tag}"
    sd = seeds(name, n)
    return [
        dict(
            name=f"{name}_s{j}",
            gpus=gpus,
            ext=[],
            out_dir=odir,
            env={},
            needs=[pool],
            cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg}"
            + (f" --pool {R}/{pool}" if tag else "")
            + f" --seeds {j}={sd[j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
            done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
        )
        for j in range(n)
    ]


def all_cells():
    out = []
    for res, sizes in GS.items():
        for D in sizes:
            pool = f"{NEST}/{res}_d{D}.npz"
            gpus = ["h100"] if D >= 100000 else ["h100", "v100"]
            out += cells(f"gs45_{res}_{D // 1000}k", D, TUNED[D], pool, 10, gpus, f"_{res}")
    g300 = f"{NEST}/genomic_d300000.npz"
    for cfg in ("e500_wd05", "e300_wd025", "e500_wd025"):
        out += cells(f"r300b_{cfg}", 300000, cfg, g300, 5, ["h100"], "")
    g10 = f"{NEST}/genomic_d10000.npz"
    for cfg in ("e600_b512_wd4", "e600_b256_wd4", "e300_b256_wd4", "e150_b128_wd4"):
        out += cells(f"t10_{cfg}", 10000, cfg, g10, 5, ["h100", "v100"], "")
    out += cells("t10_e1200_wd4_c20", 10000, "e1200_wd4", g10, 20, ["h100", "v100"], "")
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
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        for pre in ("gs45_", "r300b_", "t10_"):
            print(f"  {pre}: {sum(1 for c in new if c['name'].startswith(pre))}")
        if DRY:
            for pre in ("gs45_accessible_k562_only_10k", "r300b_e500_wd05", "t10_e300_b256_wd4"):
                c = next((c for c in new if c["name"].startswith(pre)), None)
                if c:
                    print(c["name"], c["needs"], "\n ", c["cmd"][:300])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov5a")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
