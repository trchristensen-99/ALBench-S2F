"""Unseen-reservoir bespoke-vs-menu comparison at 10k (all six) and 100k (three), extending the 30k design (u_ / un30_ cells).

Why: weakness 3 of the brief asks whether strategies the menu was not built from are handicapped, and whether that depends on size.
10k pools are nested prefixes of each replicate's 30k pool; 100k pools are new independent replicate builds. Registry order = priority:
the new LLM runs go after every existing LLM run so they never delay the core program.
"""
import json, os, sys
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad"); sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import *
os.chdir(R)
reg = json.load(open("scratchpad/dispatch_registry.json")); have = {c["name"] for c in reg}
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"; S55 = "claude-sonnet-5-5"
rel = lambda p: os.path.relpath(p, R)
ST100 = ["motif_syntax_core", "mutagenesis", "gc_matched"]
RAW100 = f"{GP}/k562_reps_raw100k"
HV = ["h100", "v100"]

pools, llm_cells, gpu_cells = [], [], []
for st in ST100:
    loop = " ".join(f'"{st} {SEED(r)}"' for r in (1, 2, 3))
    cmd = (f'for spec in {loop}; do set -- $spec; uv run --no-sync python scripts/build_pool.py --strategy $1 --size 100000 --generation-seed $2 --out-dir {RAW100} || true; done; '
           f'uv run --no-sync python -c "import sys; sys.path.insert(0, \'scratchpad\'); from stage2_lib import *; print(slice_unseen(100000, [\'{st}\']))"')
    pools.append(dict(name=f"pool100_{st}", gpus=["h100"], ext=[], out_dir=rel(RAW100), env={"AG_S2_ORACLE_DIR": f"{R}/outputs/oracle_v3/s2"}, cmd=cmd,
                      done=dict(type="file", path=rel(f"{REPS}/{st}_r3_d100000.npz"))))


def llm(name, st, r, D, cap, gpus):
    pool = f"{REPS}/{st}_r{r}_d{D}.npz"
    out = f"outputs/archsearch_reps_oct22/{st}_{D // 1000}k/rep{r}"
    return dict(name=name, gpus=gpus, ext=[], out_dir=out, env={"LLM_MODEL": S55}, needs=[rel(pool)],
                cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
                    f"--base_cap_min {cap} --budget_hours {{BUDGET}} --guidance known_good --ensemble_aware --resume",
                done=dict(type="file", path=f"{out}/COMPLETE"))


def retrain(name, st, r, D, tag, cap, k):
    pool = f"{REPS}/{st}_r{r}_d{D}.npz"; out = f"outputs/menu_retrain_v0/{tag}_{st}_r{r}"
    return dict(name=f"{name}_{st}_r{r}_s{k}", gpus=HV, ext=[], out_dir=out, env={}, needs=[rel(pool)],
                cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {MENU} --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --shard {k}/3 --base_cap_min {cap}",
                done=dict(type="file", path=f"{out}/COMPLETE_{k}of3"))


for st in UNSEEN:
    for r in (1, 2, 3):
        llm_cells.append(llm(f"u10_{st}_r{r}", st, r, 10000, 15, HV))
        for k in range(3): gpu_cells.append(retrain("un10", st, r, 10000, "un10k", 15, k))
for st in ST100:
    for r in (1, 2, 3):
        llm_cells.append(llm(f"u100_{st}_r{r}", st, r, 100000, 60, ["h100"]))
        for k in range(3): gpu_cells.append(retrain("un100", st, r, 100000, "un100k", 60, k))

new_pools = [c for c in pools if c["name"] not in have]; new_llm = [c for c in llm_cells if c["name"] not in have]; new_gpu = [c for c in gpu_cells if c["name"] not in have]
last_pool = max(i for i, c in enumerate(reg) if c["name"].startswith("pool30_"))
reg = reg[:last_pool + 1] + new_pools + reg[last_pool + 1:]
last_llm = max(i for i, c in enumerate(reg) if c["name"].startswith("g100x_"))
reg = reg[:last_llm + 1] + new_llm + reg[last_llm + 1:] + new_gpu
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1); os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new_pools), "pool builds,", len(new_llm), "LLM runs,", len(new_gpu), "menu-retrain shards; registry", len(reg))
