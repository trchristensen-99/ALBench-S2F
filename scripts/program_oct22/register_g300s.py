"""HP-transfer check at 300k: three independent AutoResearch runs (20 models each, default single-call loop, guidance known_good) on genomic 300k replicate pools r1-r3,
to compare with the retrained menu at 300k (menu minus own search at equal model counts, as done at 10k, 30k and 100k)."""
import json, os, sys
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad"); sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import *
os.chdir(R)
reg = json.load(open("scratchpad/dispatch_registry.json")); have = {c["name"] for c in reg}
S55 = "claude-sonnet-5-5"; rel = lambda p: os.path.relpath(p, R)
new = []
for r in (1, 2, 3):
    pool = f"{REPS}/genomic_r{r}_d300000.npz"; assert os.path.exists(pool), pool
    out = f"outputs/archsearch_reps_oct22/genomic_300k/rep{r}"
    new.append(dict(name=f"g300s_r{r}", gpus=["h100"], ext=[], out_dir=out, env={"LLM_MODEL": S55}, needs=[rel(pool)],
                    cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D 300000 --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
                        f"--base_cap_min 120 --budget_hours {{BUDGET}} --guidance known_good --ensemble_aware --resume",
                    done=dict(type="file", path=f"{out}/COMPLETE")))
new = [c for c in new if c["name"] not in have]
last_llm = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
# these go FIRST among pending LLM cells: put them right after the last already-finished/running group (front of the Oct-25 block)
first_new = min(i for i, c in enumerate(reg) if c["name"].startswith("ac30_"))
reg = reg[:first_new] + new + reg[first_new:]
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1); os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new), "300k search cells; registry", len(reg))
