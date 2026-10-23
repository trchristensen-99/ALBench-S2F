"""Idempotent follow-up for the Oct-22 program; run from cron every 10 minutes.
 1. slice newly built raw 30k pools into the standard replicate files
 2. when every run of an architecture-source condition is COMPLETE: build its menu (top 3 per run by validation, first 20 models) and register the retraining cells
"""
import glob, json, os, subprocess, sys, time
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad"); sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import *
os.chdir(R)
PY = f"{R}/.venv/bin/python"
log = lambda *a: print(time.strftime("%m-%d %H:%M"), *a, flush=True)  # noqa: E731
n = slice30k()
if n: log("sliced", n, "new 30k pools")
O22 = "outputs/archsearch_reps_oct22"
FAM = [f"outputs/archsearch_final_oct20/son55_wide/rep{r}" for r in (1, 2, 3)] + [f"outputs/archsearch_family_oct20/{f}/rep{r}" for f in ("random", "motif_ct_enriched", "evo2", "evoaug_paper2023") for r in (1, 2, 3)]
CONDS = {  # name -> run dirs
    "MF": FAM,                                                                 # existing mixed condition: 5 reservoirs x 3 runs (shared pool per reservoir)
    "MG": [f"{O22}/genomic_30k/rep{r}" for r in range(1, 16)],                 # genomic only, 15 runs on 15 independent pools
    "MR": [f"{O22}/random_30k/rep{r}" for r in range(1, 16)],                  # random only, 15 runs
    "MGR": [f"{O22}/genomic_30k/rep{r}" for r in range(1, 9)] + [f"{O22}/random_30k/rep{r}" for r in range(1, 8)],   # mixed search: 8 genomic + 7 random
    "MX": [f"{O22}/GRM_30k/rep{r}" for r in range(1, 16)],                     # searches run ON the mixed GRM corpus (10k genomic + 10k random + 10k motif), 15 runs on 15 independent pools
}
TARGETS = {"genomic": f"{REPS}/genomic_r1_d30000.npz", "random": f"{REPS}/random_r1_d30000.npz", "motif": f"{REPS}/motif_ct_enriched_r1_d30000.npz", "grm": f"{REPS}/GRM_r1_d30000.npz"}
reg = json.load(open("scratchpad/dispatch_registry.json")); have = {c["name"] for c in reg}
new = []
os.makedirs("outputs/menus_oct22", exist_ok=True)
for cond, dirs in CONDS.items():
    menu = f"outputs/menus_oct22/{cond}.json"
    if not all(os.path.exists(d + "/COMPLETE") for d in dirs):
        continue
    if not os.path.exists(menu):
        r = subprocess.run([PY, "scratchpad/build_menu.py", "--cells", *dirs, "--k", "3", "--first_n", "20", "--no_reference", "--out", menu], capture_output=True, text=True)
        log("built menu", cond, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])
    for tname, pool in TARGETS.items():
        for k in range(3):
            nm = f"as_{cond}_{tname}_s{k}"
            if nm in have: continue
            out = f"outputs/menu_retrain_v0/as30k_{cond}_{tname}"
            new.append(dict(name=nm, gpus=["h100", "v100"], ext=[], out_dir=out, env={}, needs=[os.path.relpath(pool, R)],
                            cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {R}/{menu} --reservoir_cache {pool} --D 30000 --out_dir {R}/{out} --shard {k}/3 --base_cap_min 25",
                            done=dict(type="file", path=f"{out}/COMPLETE_{k}of3")))
if new:
    reg = json.load(open("scratchpad/dispatch_registry.json"))
    reg += [c for c in new if c["name"] not in {x["name"] for x in reg}]
    json.dump(reg, open("scratchpad/dispatch_registry.json", "w"), indent=1)
    log("registered", len(new), "retrain cells")
