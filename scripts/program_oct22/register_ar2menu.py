"""Per-D ensembling of the ar2 menus: train each search's exported menu (`ar2 export-menu` -> ws/menu/menu_ar2.json) on the pool the search used, with our
menu_retrain harness (RC-averaged MSE, shift augmentation, early stopping on V0c) and stack the members with ElasticNetCV, so the result is comparable with the
v1 menu retrained on the same pools (ind30m_v1_*, patience 8). Three seeds per configuration; cells wait (dispatcher `needs`) until the menu file exists.

  python scripts/program_oct22/register_ar2menu.py [--dry]
"""

import json
import os
import shutil
import sys

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
os.chdir(R)
REPS = f"{R}/outputs/grid_pools/k562_reps"
CELLS = {  # cell -> (pool, D, shards)
    "genomic_10k": (f"{REPS}/genomic_r1_d10000.npz", 10000, 3),
    "genomic_30k": (f"{REPS}/genomic_r1_d30000.npz", 30000, 3),
    "random_30k": (f"{REPS}/random_r1_d30000.npz", 30000, 3),
    "motif_30k": (f"{REPS}/motif_ct_enriched_r1_d30000.npz", 30000, 3),
}
CAP = {10000: 15, 30000: 25, 100000: 60}
reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
new = []
for cell, (pool, D, sh) in CELLS.items():
    menu = f"outputs/ar2/ws_{cell}/menu/menu_ar2.json"
    out = f"outputs/menu_retrain_v0/ar2m_{cell}"
    for k in range(sh):
        name = f"ar2m_{cell}_s{k}"
        if name in have:
            continue
        new.append(
            dict(
                name=name,
                gpus=["h100", "v100"],
                ext=[],
                out_dir=out,
                env={"AL_PATIENCE": "8", "MENU_MAX_FWD_MS": "1000000000"},
                needs=[menu, os.path.relpath(pool, R)],
                cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {R}/{menu} --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --shard {k}/{sh} --seeds 3 --base_cap_min {CAP[D]}",
                done=dict(type="file", path=f"{out}/COMPLETE_{k}of{sh}"),
            )
        )
i = max(k for k, c in enumerate(reg) if c["name"].startswith("ar2w_"))
reg2 = reg[: i + 1] + new + reg[i + 1 :]
print("new cells", len(new), "registry", len(reg), "->", len(reg2))
if "--dry" not in sys.argv:
    shutil.copy("scratchpad/dispatch_registry.json", "scratchpad/dispatch_registry.json.bak_ar2m")
    json.dump(reg2, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
    os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
    print("registered")
