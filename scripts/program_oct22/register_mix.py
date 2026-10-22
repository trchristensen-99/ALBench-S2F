import json, os, sys
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad"); sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import *
os.chdir(R)
reg = json.load(open("scratchpad/dispatch_registry.json")); have = {c["name"] for c in reg}
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"
names = list(json.load(open(f"{GP}/k562_mix/manifest.json")))
new = []
for nm in names:
    pool = f"{GP}/k562_mix/{nm}_d30000.npz"
    for k in range(3):
        cn = f"mix_{nm}_s{k}"
        if cn in have: continue
        out = f"outputs/menu_retrain_v0/mix30k_{nm}"
        new.append(dict(name=cn, gpus=["h100", "v100"], ext=[], out_dir=out, env={}, needs=[os.path.relpath(pool, R)],
                        cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {MENU} --reservoir_cache {pool} --D 30000 --out_dir {R}/{out} --shard {k}/3 --base_cap_min 25",
                        done=dict(type="file", path=f"{out}/COMPLETE_{k}of3")))
# mixture retrains go before the LLM runs' tail so they are submitted early (they only need GPUs)
i = next(i for i, c in enumerate(reg) if c["name"].startswith("g30_r1"))
reg = reg[:i] + new + reg[i:]
json.dump(reg, open("scratchpad/dispatch_registry.json", "w"), indent=1); print("registered", len(new), "mixture cells; registry", len(reg))
