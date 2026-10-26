"""Oct-26 coordinator grid: does the coordinator's +0.005 on genomic 30k (6/6 paired wins) hold across reservoirs and sizes?

Agent loop without the reflector (propose 6 ideas, coordinator keeps 2, coder, evaluator), the simplest variant that matched the full one on genomic 30k,
on the replicate pools that already have default-loop runs: 30k random 6, motif_ct_enriched 6, evoaug 3, evo2 3; 10k genomic 6, random 6, motif_ct_enriched 6 (36 runs).

    python scripts/program_oct22/register_oct26b.py [--dry]
"""

import json
import os
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import GP, REPS, R  # noqa: E402

os.chdir(R)
reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
S55 = "claude-sonnet-5-5"
OUT22 = "outputs/archsearch_reps_oct22"
CFG = (
    " --loop agent --agent_cfg '"
    + json.dumps(dict(stages=["propose", "coordinate", "code", "evaluate"], ideas_per_round=6))
    + "'"
)
POOL = {
    "genomic": lambda r, D: f"{REPS}/genomic_r{r}_d{D}.npz",
    "random": lambda r, D: f"{REPS}/random_r{r}_d{D}.npz",
    "motif_ct_enriched": lambda r, D: f"{REPS}/motif_ct_enriched_r{r}_d{D}.npz",
    "evoaug_paper2023": lambda r, D: f"{REPS}/evoaug_paper2023_r{r}_d{D}.npz",
    "evo2": lambda r, D: f"{GP}/k562_cumulative/evo2_d{D}.npz",
}
TAG = {
    "genomic": "g",
    "random": "r",
    "motif_ct_enriched": "m",
    "evoaug_paper2023": "e",
    "evo2": "v",
}
GRID = [
    (30000, "random", 6),
    (30000, "motif_ct_enriched", 6),
    (30000, "evoaug_paper2023", 3),
    (30000, "evo2", 3),
    (10000, "genomic", 6),
    (10000, "random", 6),
    (10000, "motif_ct_enriched", 6),
]
new = []
for D, st, n in GRID:
    for r in range(1, n + 1):
        pool = POOL[st](r, D)
        assert os.path.exists(pool), pool
        out = f"{OUT22}/{st}_{D // 1000}k_agent_coord_noref/rep{r}"
        new.append(
            dict(
                name=f"an30_{TAG[st]}{D // 1000}_r{r}",
                gpus=["v100"],
                ext=[],
                out_dir=out,
                env={"LLM_MODEL": S55},
                needs=[os.path.relpath(pool, R)],
                cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
                f"--base_cap_min {15 if D == 10000 else 25} --budget_hours {{BUDGET}} --guidance known_good --ensemble_aware{CFG} --resume",
                done=dict(type="file", path=f"{out}/COMPLETE"),
            )
        )
new = [c for c in new if c["name"] not in have]
print(len(new), "new coordinator-grid runs")
if "--dry" in sys.argv:
    sys.exit(0)
# keep them with the other LLM cells (after the last LLM cell), before the Oct-26 retrain cells
last_llm = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
reg = reg[: last_llm + 1] + new + reg[last_llm + 1 :]
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new), "registry", len(reg))
