"""Oct-29 (evening): where is the optimum of the input context and of the shift augmentation, and does it depend on dataset size and reservoir? (cells sw38_)

One-factor sweeps around the current standard (20 real flank bases each side, shift +-10), with the first six menu-v1 configurations that take wider input (menu indices 1 to 6; index 0 fails to build at any other
input length and costs seconds), one seed, V0c validation, patience 16, nested pools, paired by configuration with the no-context arm of the same cell.

  context sweep (shift +-10): c12 c16 c20 c24 c28 c32 c40 c48 c60 real bases per side (input 224 ... 320 bp; every length is a multiple of 8)
  shift sweep (20 bases):     s0 s5 s15 s20 s30 (+-S per batch), orc = the oracle's schedule (+-15 per sample on half of the samples)
  rca                         20 bases, +-10, reverse-complement augmentation as in the oracle (one pass, flip p 0.5) instead of the forward + reverse-complement loss
  base                        no context (menu as it was)

The five standard cells (reservoir, size): genomic at 10k, 30k and 100k (dataset-size variance) and random and CT-enriched motif at 30k (reservoir variance). Every arm in every cell: 17 arms x 5 cells = 85 cell-arms.

  sw38_<D>k_<reservoir>_<arm>_s<k>   a shard of the menu retrain (one shard at 10k and 30k, two at 100k)

    python scripts/program_oct22/register_sw38.py [--dry] [--replace]      (--replace first removes every existing sw38_ cell from the registry; only safe while none has started)
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
CELLS = [
    ("genomic", 10000),
    ("genomic", 30000),
    ("genomic", 100000),
    ("random", 30000),
    ("motif_ct_enriched", 30000),
]
CAP = {10000: 15, 30000: 25, 100000: 60}
ARMS = {"base": ("", {})}
for F in (12, 16, 20, 24, 28, 32, 40, 48, 60):
    ARMS[f"c{F}"] = (f" --flank_ctx {F} --shift_max 10", {})
for S in (0, 5, 15, 20, 30):
    ARMS[f"s{S}"] = (f" --flank_ctx 20 --shift_max {S}", {})
ARMS["orc"] = (" --flank_ctx 20 --shift_max 15 --shift_per_sample --shift_prob 0.5", {})
ARMS["rca"] = (" --flank_ctx 20 --shift_max 10", {"AL_RC_MODE": "aug"})


def all_cells():
    out = []
    for res, D in CELLS:
        pool = f"{NEST}/{res}_d{D}.npz"
        for arm, (extra, env) in ARMS.items():
            cap = CAP[D] * (1.4 if arm in ("c48", "c60") else 1.0)
            out += fb.cells(
                f"sw38_{D // 1000}k_{res}_{arm}",
                V1,
                pool,
                D,
                {"MENU_MAX_N": "7", **env},
                extra,
                shards=2 if D >= 100000 else 1,
                cap=cap,
            )
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        if "--replace" in sys.argv:
            stale = [c["name"] for c in reg if c["name"].startswith("sw38_")]
            started = [
                n
                for n in stale
                if any(
                    os.path.exists(f"{R}/{OUT}/{n.rsplit('_s', 1)[0]}/r00_menu_001_meta.json")
                    for OUT in ("outputs/menu_retrain_v0",)
                )
            ]
            assert not started, f"cells already produced output: {started[:3]}"
            reg = [c for c in reg if not c["name"].startswith("sw38_")]
            print("removed", len(stale), "existing sw38_ cells")
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
