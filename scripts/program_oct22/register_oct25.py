"""Oct-25 search-loop variants (LLM runs, V100-only cells).

Tier 1, genomic 30k pools r1-r6 (paired with the default single-call runs g30_r1-r6):
  ac30_  agent loop with a COORDINATOR: propose 6 ideas, coordinator keeps 2, coder, evaluator, reflector
  an30_  the same without the reflector (isolates the coordinator)
  ae30_  agent loop (no coordinator) with an explore-style proposer
Tier 2, the single-call loop with --style explore against the default prompt across the reservoir grid (genomic, random, motif_ct_enriched, evoaug, evo2) at 10k and 30k.
"""
import json, os, sys
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad"); sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import *
os.chdir(R)
reg = json.load(open("scratchpad/dispatch_registry.json")); have = {c["name"] for c in reg}
S55 = "claude-sonnet-5-5"; rel = lambda p: os.path.relpath(p, R)
V = ["v100"]
NEST = f"{GP}/k562_nested"
# 10k replicate pools for random and motif r1-r6 (nested prefix of each replicate's 30k pool)
print("10k pools sliced:", slice10k_from30k(strategies=("random", "motif_ct_enriched"), reps=tuple(range(1, 7))))


def llm(name, pool, out, D, extra=""):
    cap = 15 if D == 10000 else 25
    return dict(name=name, gpus=V, ext=[], out_dir=out, env={"LLM_MODEL": S55}, needs=[rel(pool)],
                cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
                    f"--base_cap_min {cap} --budget_hours {{BUDGET}} --guidance known_good --ensemble_aware{extra} --resume",
                done=dict(type="file", path=f"{out}/COMPLETE"))


O = "outputs/archsearch_reps_oct22"
cfg = lambda stages, **kw: " --loop agent --agent_cfg '" + json.dumps(dict(stages=stages, **kw)) + "'"
new = []
gp = lambda r, D=30000: f"{REPS}/genomic_r{r}_d{D}.npz"
for r in range(1, 7):
    new.append(llm(f"ac30_r{r}", gp(r), f"{O}/genomic_30k_agent_coord/rep{r}", 30000, cfg(["propose", "coordinate", "code", "evaluate", "reflect"], ideas_per_round=6)))
for r in range(1, 7):
    new.append(llm(f"an30_r{r}", gp(r), f"{O}/genomic_30k_agent_coord_noref/rep{r}", 30000, cfg(["propose", "coordinate", "code", "evaluate"], ideas_per_round=6)))
# tier 2: explore vs default across the grid
GRID = {  # tag -> (strategy dir name, pool path fn, n_runs, has_default_already {D: n})
    "rnd": ("random", lambda r, D: f"{REPS}/random_r{r}_d{D}.npz", 6, {30000: 6}),
    "mot": ("motif_ct_enriched", lambda r, D: f"{REPS}/motif_ct_enriched_r{r}_d{D}.npz", 6, {30000: 6}),
    "gen": ("genomic", lambda r, D: gp(r, D), 6, {30000: 6, 10000: 6}),
    "eva": ("evoaug_paper2023", lambda r, D: f"{REPS}/evoaug_paper2023_r{r}_d{D}.npz", 3, {}),
    "ev2": ("evo2", lambda r, D: f"{GP}/k562_cumulative/evo2_d{D}.npz", 3, {}),  # legacy per-size pool: no nested or replicate Evo2 pools exist; both arms use the same pool
}
for D, dtag in ((30000, "30"), (10000, "10")):
    for tag, (st, pf, n, have_def) in GRID.items():
        if tag == "gen" and D == 30000:
            continue  # genomic 30k explore already done (9 runs)
        for r in range(1, n + 1):
            pool = pf(r, D)
            assert os.path.exists(pool), pool
            new.append(llm(f"xe{dtag}_{tag}_r{r}", pool, f"{O}/{st}_{D // 1000}k_explore/rep{r}", D, " --style explore"))
            if D not in have_def:
                new.append(llm(f"xd{dtag}_{tag}_r{r}", pool, f"{O}/{st}_{D // 1000}k/rep{r}", D))
for r in range(1, 7):
    new.append(llm(f"ae30_r{r}", gp(r), f"{O}/genomic_30k_agent_explore/rep{r}", 30000, cfg(["propose", "code", "evaluate", "reflect"], proposer_style="explore")))
new = [c for c in new if c["name"] not in have]
last_llm = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
reg = reg[:last_llm + 1] + new + reg[last_llm + 1:]
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1); os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
import collections
print("registered", len(new), "LLM cells:", dict(collections.Counter(c["name"].split("_")[0] for c in new)), "registry", len(reg))
