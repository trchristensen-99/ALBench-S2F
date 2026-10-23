"""Oct-23 additions: (1) search on mixed-corpus pools (GRM = 10k genomic + 10k random + 10k motif per replicate) x15, (2) prompt-style arms
explore / reflect / explore_reflect on genomic 30k pools r1-r9 (paired with the existing default-prompt runs g30_r1-r9), (3) precision A/B retrains
of the 33-architecture menu on genomic r1 30k (fp32 baseline = ind30k_genomic_r1)."""
import json, os, sys
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad"); sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import *
import numpy as np
os.chdir(R)
reg = json.load(open("scratchpad/dispatch_registry.json")); have = {c["name"] for c in reg}
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"; S55 = "claude-sonnet-5-5"
rel = lambda p: os.path.relpath(p, R)
HV = ["h100", "v100"]

# ---- mixed replicate pools: first 10k of each component's replicate pool (nested prefixes), shuffled together
n_new = 0
for r in range(1, 16):
    dst = f"{REPS}/GRM_r{r}_d30000.npz"
    if os.path.exists(dst):
        continue
    S, Y = [], []
    for st in ("genomic", "random", "motif_ct_enriched"):
        z = np.load(f"{REPS}/{st}_r{r}_d30000.npz", allow_pickle=True)
        S += list(z["sequences"])[:10000]; Y.append(z["oracle_labels"][:10000])
    Y = np.concatenate(Y); perm = np.random.default_rng(SEED(r) + 1000).permutation(len(S))
    save_pool(dst, [S[i] for i in perm], Y[perm], "oracle_v3", "GRM", SEED(r)); n_new += 1
print("mixed pools built:", n_new)


def llm(name, pool, out, D, cap, gpus, extra=""):
    return dict(name=name, gpus=gpus, ext=[], out_dir=out, env={"LLM_MODEL": S55}, needs=[rel(pool)],
                cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
                    f"--base_cap_min {cap} --budget_hours {{BUDGET}} --guidance known_good --ensemble_aware{extra} --resume",
                done=dict(type="file", path=f"{out}/COMPLETE"))


new = []
for r in range(1, 16):
    new.append(llm(f"mx30_r{r}", f"{REPS}/GRM_r{r}_d30000.npz", f"outputs/archsearch_reps_oct22/GRM_30k/rep{r}", 30000, 25, HV))
for tag, style in (("ex30", "explore"), ("rf30", "reflect"), ("xr30", "explore_reflect")):
    for r in range(1, 10):
        new.append(llm(f"{tag}_r{r}", f"{REPS}/genomic_r{r}_d30000.npz", f"outputs/archsearch_reps_oct22/genomic_30k_{style}/rep{r}", 30000, 25, HV, f" --style {style}"))
pool = f"{REPS}/genomic_r1_d30000.npz"
for mode in ("bf16", "fp16", "tf32"):
    out = f"outputs/menu_retrain_v0/prec30k_{mode}_genomic_r1"
    for k in range(3):
        new.append(dict(name=f"ind30_prec_{mode}_s{k}", gpus=["h100"], ext=[], out_dir=out, env={"AL_PRECISION": mode}, needs=[rel(pool)],
                        cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {MENU} --reservoir_cache {pool} --D 30000 --out_dir {R}/{out} --shard {k}/3 --base_cap_min 25",
                        done=dict(type="file", path=f"{out}/COMPLETE_{k}of3")))
new = [c for c in new if c["name"] not in have]
# LLM cells go ahead of retrains (registry order = priority): insert after the last existing LLM cell
last_llm = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
llm_new = [c for c in new if "LLM_MODEL" in c["env"]]; gpu_new = [c for c in new if "LLM_MODEL" not in c["env"]]
reg = reg[:last_llm + 1] + llm_new + reg[last_llm + 1:] + gpu_new
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1); os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(llm_new), "LLM cells and", len(gpu_new), "retrain cells; registry", len(reg))
