"""Coverage completion while the queue drains: 100k pools and menu retrains for the two unseen reservoirs that had only 10k and 30k
(motif_shared_core, dinuc_shuffle; Zoonomia is left out: its sampler's capacity at 100k was never checked), three replicates each.
The evoaug 300k menu retrain is registered by register_oct26.py --only a300.

    python scripts/program_oct22/register_oct26c.py
"""

import json
import os
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import GP, REPS, SEED, R  # noqa: E402

os.chdir(R)
reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"
RAW100 = f"{GP}/k562_reps_raw100k"
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
new = []
for st in ("motif_shared_core", "dinuc_shuffle"):
    loop = " ".join(f'"{st} {SEED(r)}"' for r in (1, 2, 3))
    cmd = (
        f"for spec in {loop}; do set -- $spec; uv run --no-sync python scripts/build_pool.py --strategy $1 --size 100000 --generation-seed $2 --out-dir {RAW100} || true; done; "
        f"uv run --no-sync python -c \"import sys; sys.path.insert(0, 'scratchpad'); from stage2_lib import *; print(slice_unseen(100000, ['{st}']))\""
    )
    new.append(
        dict(
            name=f"pool100_{st}",
            gpus=["h100"],
            ext=[],
            out_dir=rel(RAW100),
            env={"AG_S2_ORACLE_DIR": f"{R}/outputs/oracle_v3/s2"},
            cmd=cmd,
            done=dict(type="file", path=rel(f"{REPS}/{st}_r3_d100000.npz")),
        )
    )
    for r in (1, 2, 3):
        pool = f"{REPS}/{st}_r{r}_d100000.npz"
        out = f"outputs/menu_retrain_v0/un100k_{st}_r{r}"
        for k in range(3):
            new.append(
                dict(
                    name=f"un100_{st}_r{r}_s{k}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=out,
                    env={},
                    needs=[rel(pool)],
                    cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {MENU} --reservoir_cache {pool} --D 100000 --out_dir {R}/{out} --shard {k}/3 --base_cap_min 60",
                    done=dict(type="file", path=f"{out}/COMPLETE_{k}of3"),
                )
            )
new = [c for c in new if c["name"] not in have]
reg += new
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new), "cells; registry", len(reg))
