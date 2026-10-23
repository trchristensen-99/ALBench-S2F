"""Pilot of the multi-agent search loop: 5 runs on genomic 30k pools r1-r5, paired with the existing default-loop runs g30_r1-r5."""
import json, os, sys
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad"); sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import *
os.chdir(R)
reg = json.load(open("scratchpad/dispatch_registry.json")); have = {c["name"] for c in reg}
S55 = "claude-sonnet-5-5"; rel = lambda p: os.path.relpath(p, R)
new = []
for r in range(1, 6):
    pool = f"{REPS}/genomic_r{r}_d30000.npz"; out = f"outputs/archsearch_reps_oct22/genomic_30k_agent/rep{r}"
    new.append(dict(name=f"ag30_r{r}", gpus=["v100"], ext=[], out_dir=out, env={"LLM_MODEL": S55}, needs=[rel(pool)],
                    cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D 30000 --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
                        f"--base_cap_min 25 --budget_hours {{BUDGET}} --guidance known_good --ensemble_aware --loop agent --resume",
                    done=dict(type="file", path=f"{out}/COMPLETE")))
new = [c for c in new if c["name"] not in have]
last_llm = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
reg = reg[:last_llm + 1] + new + reg[last_llm + 1:]
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1); os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new), "agent-loop pilot cells; registry", len(reg))
