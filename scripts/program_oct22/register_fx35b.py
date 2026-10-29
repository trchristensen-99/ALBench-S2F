"""Oct-29 (afternoon): the real plasmid flank as student input is the largest effect found in this study (single configurations +0.05 to +0.07 test r at 20 real bases each side; the first
finished fx35_c20s10 models reach test 0.80 to 0.83, val 0.855 to 0.873: the range of the collaborator's best single model). Follow-ups, all on the canonical genomic 30k pool with V0c validation and
menu v1 (15 of the 20 configurations can take wider input):

  fx35_c{4,8,12}s10_30k   how little context is enough (208 / 216 / 224 bp input)
  fx35_c20s0_30k          20 bases of context, NO shift augmentation (does the shift still matter?)
  fx35_c20s30_30k         20 bases of context, shift +-30 over real bases
  fx35_c200s10_30k        the full 600-bp window the oracle sees (200 real bases each side)
  fe35_c20s10_emabn_30k   20 bases of context + weight averaging (0.9995) + BN recalculation
  lg35c_{c20,c20_emabn,c60_emabn}_30k   the LegNet-style reference (3 recipes x 3 seeds) with context, the closest thing we have to the collaborator's configuration
  fr35_<reservoir>_c20_30k  fairness: menu v1 with 20 bases of context on ten other reservoirs at 30k (genomic is fx35_c20s10); does the ranking of strategies change when students see the flank?

    python scripts/program_oct22/register_fx35b.py [--dry]
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
import register_pk35 as pk  # noqa: E402

R, NEST, rel, V1 = r30.R, r30.NEST, r30.rel, r30.V1
DRY = "--dry" in sys.argv
V0C = pk.V0C
OUT = pk.OUT
EMA = pk.EMA
ARMS = {  # name: (extra args, env, cap)
    "fx35_c4s10_30k": (" --flank_ctx 4 --shift_max 10", {}, 25),
    "fx35_c8s10_30k": (" --flank_ctx 8 --shift_max 10", {}, 25),
    "fx35_c12s10_30k": (" --flank_ctx 12 --shift_max 10", {}, 25),
    "fx35_c20s0_30k": (" --flank_ctx 20 --shift_max 0", {}, 25),
    "fx35_c20s30_30k": (" --flank_ctx 20 --shift_max 30", {}, 25),
    "fx35_c200s10_30k": (" --flank_ctx 200 --shift_max 10", {}, 90),
    "fe35_c20s10_emabn_30k": (" --flank_ctx 20 --shift_max 10", EMA, 30),
}
LGC = {
    "lg35c_c20_30k": (" --flank_ctx 20 --shift_max 10", {}),
    "lg35c_c20_emabn_30k": (" --flank_ctx 20 --shift_max 10", EMA),
    "lg35c_c60_emabn_30k": (" --flank_ctx 60 --shift_max 10", EMA),
}
RESERVOIRS = [
    "random",
    "motif_ct_enriched",
    "evoaug_paper2023",
    "zoonomia_orthologs",
    "motif_shared_core",
    "random_gc_sweep",
    "mutagenesis_mr10",
    "d3_steps64",
    "d3_steps256",
    "evo2",
]


def cells(name, menu, pool, D, env_extra, extra, shards=4, cap=25, seeds=1):
    out_dir = f"{OUT}/{name}"
    env = {"AL_PATIENCE": "16", "MENU_MAX_FWD_MS": "1000000000", **(env_extra or {})}
    return [
        dict(
            name=f"{name}_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=out_dir,
            env=dict(env),
            needs=[rel(pool), rel(menu)],
            cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {menu} --reservoir_cache {pool} --D {D} --out_dir {R}/{out_dir} --shard {k}/{shards} --base_cap_min {cap} --val_cache {V0C}{extra}"
            + (f" --seeds {seeds}" if seeds != 1 else ""),
            done=dict(type="file", path=f"{out_dir}/COMPLETE_{k}of{shards}"),
        )
        for k in range(shards)
    ]


def all_cells():
    out = []
    g30 = f"{NEST}/genomic_d30000.npz"
    for name, (extra, env, cap) in ARMS.items():
        out += cells(name, V1, g30, 30000, env, extra, cap=cap)
    for name, (extra, env) in LGC.items():
        out += cells(name, pk.LG_MENU, g30, 30000, env, extra, shards=3, seeds=3)
    for res in RESERVOIRS:
        out += cells(
            f"fr35_{res}_c20_30k",
            V1,
            f"{NEST}/{res}_d30000.npz",
            30000,
            {},
            " --flank_ctx 20 --shift_max 10",
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
        print("add", len(new), "cells | registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_fx35b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
