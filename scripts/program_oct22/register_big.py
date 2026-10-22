import json, os, sys
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad"); sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import *
os.chdir(R)
from albench.pools import PoolSpec
reg = json.load(open("scratchpad/dispatch_registry.json")); have = {c["name"] for c in reg}
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"; S55 = "claude-sonnet-5-5"
rel = lambda p: os.path.relpath(p, R)
pools_cells, llm_cells, gpu_cells = [], [], []
# ---- 1. build the missing 30k replicate pools (random, motif r4-15; unseen reservoirs r1-3), grouped 6 per job
todo = [(st, r) for st in ("random", "motif_ct_enriched") for r in range(4, 16)] + [(st, r) for st in UNSEEN for r in (1, 2, 3)]
for gi in range(0, len(todo), 6):
    grp = todo[gi:gi + 6]
    nm = f"pool30_{gi // 6}"
    if nm in have: continue
    last = str(PoolSpec(strategy=grp[-1][0], params={}, size=30000, generation_seed=SEED(grp[-1][1]), task="k562").path(RAW30))
    loop = " ".join(f'"{st} {SEED(r)}"' for st, r in grp)
    pools_cells.append(dict(name=nm, gpus=["h100"], ext=[], out_dir="outputs/grid_pools/k562_reps_raw30k", env={"AG_S2_ORACLE_DIR": f"{R}/outputs/oracle_v3/s2"},
                            cmd=f'for spec in {loop}; do set -- $spec; uv run --no-sync python scripts/build_pool.py --strategy $1 --size 30000 --generation-seed $2 --out-dir {RAW30} || true; done',
                            done=dict(type="file", path=rel(last))))
# ---- 2. LLM runs: 20 models each (10 rounds x 2), every run on its own replicate pool
def llm(name, st, r, D, cap, gpus, order):
    pool = f"{REPS}/{st}_r{r}_d{D}.npz"
    out = f"outputs/archsearch_reps_oct22/{st}_{D // 1000}k/rep{r}"
    return dict(name=name, gpus=gpus, ext=[], out_dir=out, env={"LLM_MODEL": S55}, needs=[rel(pool)], _o=order,
                cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
                    f"--base_cap_min {cap} --budget_hours {{BUDGET}} --guidance known_good --ensemble_aware --resume",
                done=dict(type="file", path=f"{out}/COMPLETE"))
HV = ["h100", "v100"]
for r in range(1, 16): llm_cells.append(llm(f"g30_r{r}", "genomic", r, 30000, 25, HV, 1))
for r in range(1, 16): llm_cells.append(llm(f"rnd30_r{r}", "random", r, 30000, 25, HV, 2))
for st in UNSEEN:
    for r in (1, 2, 3): llm_cells.append(llm(f"u_{st}_r{r}", st, r, 30000, 25, HV, 3))
for r in range(1, 16): llm_cells.append(llm(f"mot30_r{r}", "motif_ct_enriched", r, 30000, 25, HV, 4))
for r in range(1, 11): llm_cells.append(llm(f"g10_r{r}", "genomic", r, 10000, 15, HV, 5))
for r in range(4, 11): llm_cells.append(llm(f"g100x_r{r}", "genomic", r, 100000, 60, ["h100"], 6))
# ---- 3. GPU cells: menu retrain on the unseen reservoirs' replicate pools (3 pools x 3 shards each)
for st in UNSEEN:
    for r in (1, 2, 3):
        pool = f"{REPS}/{st}_r{r}_d30000.npz"
        for k in range(3):
            out = f"outputs/menu_retrain_v0/un30k_{st}_r{r}"
            gpu_cells.append(dict(name=f"un30_{st}_r{r}_s{k}", gpus=HV, ext=[], out_dir=out, env={}, needs=[rel(pool)],
                                  cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {MENU} --reservoir_cache {pool} --D 30000 --out_dir {R}/{out} --shard {k}/3 --base_cap_min 25",
                                  done=dict(type="file", path=f"{out}/COMPLETE_{k}of3")))
new = [c for c in pools_cells + llm_cells + gpu_cells if c["name"] not in have]
for c in new: c.pop("_o", None)
json.dump(new + reg if False else reg + new, open("scratchpad/dispatch_registry.json", "w"), indent=1)
# put pool builds first, then LLM runs in priority order, then the rest (registry order = submission priority)
reg = json.load(open("scratchpad/dispatch_registry.json"))
pools_first = [c for c in reg if c["name"].startswith("pool30_")]
rest = [c for c in reg if not c["name"].startswith("pool30_")]
json.dump(pools_first + rest, open("scratchpad/dispatch_registry.json", "w"), indent=1)
print("registered", len(new), "cells:", len(pools_cells), "pool-build,", len(llm_cells), "LLM runs,", len(gpu_cells), "menu-retrain; registry", len(reg))
