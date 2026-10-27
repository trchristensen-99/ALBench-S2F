"""Oct-27 overnight experiments: close the gaps in which the earlier assumptions were tested at one size or one reservoir.

llmctl    loop-element controls on genomic 30k pools r1-r6, paired with the default single-call runs (g30_) and the coordinator runs (an30_):
            ag30_p_   agent loop with a separate coder but no coordinator and no reflector (propose 2, code, evaluate)
            ag30_x_   the coordinator arm's over-generation (6 ideas, 2 kept) with the 2 picked AT RANDOM, no LLM call (control for the coordinator's judgment)
          prompt content: --guidance none against the default known_good guidance, random 30k (6), motif 30k (6), genomic 10k (6)   (xd30_gn_)
recipe    training-recipe sensitivity of the retrained menu: patience 4 / 16 (AL_PATIENCE), epoch budget x0.5 / x2, on genomic, random, motif at 30k
          and genomic at 10k and 100k                                                                                                  (ind30e_)
valdep    validation labels: the oracle_v3-relabelled validation set (V0c) against the default one at 10k and 100k, three reservoirs        (ind30w_)
hgdep     HepG2 and differential students at 10k and 100k (K562 arm = the V0c arms above)                                                  (ind30h_)
hptune    does retuning the recipe at 600k close the gap to the extrapolation? best 3 architectures of random 300k, lr x0.5 / x2, batch x4, batch x4 + lr x2  (ind30t_)

  python scripts/program_oct22/register_oct27.py [--only llmctl,recipe,...] [--dry]
"""

import glob
import json
import os
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import GP, REPS, R  # noqa: E402

os.chdir(R)
ONLY = (
    sys.argv[sys.argv.index("--only") + 1].split(",")
    if "--only" in sys.argv
    else ["llmctl", "recipe", "valdep", "hgdep", "hptune"]
)
DRY = "--dry" in sys.argv
S55 = "claude-sonnet-5-5"
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"
OUT22 = "outputs/archsearch_reps_oct22"
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
RES = ["genomic", "random", "motif_ct_enriched"]
CAP = {10000: 15, 30000: 25, 100000: 60, 600000: 240}
P = lambda st, D, r=1: f"{REPS}/{st}_r{r}_d{D}.npz"  # noqa: E731
new = []


def llm(name, pool, out, D, extra=""):
    cap = 15 if D == 10000 else 25
    return dict(
        name=name,
        gpus=["v100"],
        ext=[],
        out_dir=out,
        env={"LLM_MODEL": S55},
        needs=[rel(pool)],
        cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
        f"--base_cap_min {cap} --budget_hours {{BUDGET}} --ensemble_aware{extra} --resume",
        done=dict(type="file", path=f"{out}/COMPLETE"),
    )


def retrain(
    name, pool, D, out, n_sh, extra="", gpus=("h100", "v100"), menu=MENU, env=None, needs=()
):
    return [
        dict(
            name=f"{name}_s{k}",
            gpus=list(gpus),
            ext=[],
            out_dir=out,
            env=dict(env or {}),
            needs=sorted({rel(pool), *needs}),
            cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {menu} --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --shard {k}/{n_sh} --base_cap_min {CAP[D]}{extra}",
            done=dict(type="file", path=f"{out}/COMPLETE_{k}of{n_sh}"),
        )
        for k in range(n_sh)
    ]


cfg = lambda **kw: " --guidance known_good --loop agent --agent_cfg '" + json.dumps(kw) + "'"  # noqa: E731
if "llmctl" in ONLY:
    for r in range(1, 7):
        gp = P("genomic", 30000, r)
        new.append(
            llm(
                f"ag30_p_r{r}",
                gp,
                f"{OUT22}/genomic_30k_agent_plain/rep{r}",
                30000,
                cfg(stages=["propose", "code", "evaluate"]),
            )
        )
        new.append(
            llm(
                f"ag30_x_r{r}",
                gp,
                f"{OUT22}/genomic_30k_agent_randpick/rep{r}",
                30000,
                cfg(
                    stages=["propose", "coordinate", "code", "evaluate"],
                    ideas_per_round=6,
                    coordinator_mode="random",
                ),
            )
        )
    for tag, st, D in (
        ("rnd", "random", 30000),
        ("mot", "motif_ct_enriched", 30000),
        ("gen10", "genomic", 10000),
    ):
        for r in range(1, 7):
            new.append(
                llm(
                    f"xd30_gn_{tag}_r{r}",
                    P(st, D, r),
                    f"{OUT22}/{st}_{D // 1000}k_noguid/rep{r}",
                    D,
                )
            )  # no --guidance flag = guidance none

if "recipe" in ONLY:
    arms = {
        "pat4": (" ", {"AL_PATIENCE": "4"}),
        "pat16": (" ", {"AL_PATIENCE": "16"}),
        "ep05": (" --epoch_scale 0.5", {}),
        "ep2": (" --epoch_scale 2.0", {}),
    }
    for arm, (extra, env) in arms.items():
        for st, D, sh in [(s, 30000, 3) for s in RES] + [
            ("genomic", 10000, 1),
            ("genomic", 100000, 3),
        ]:
            new += retrain(
                f"ind30e_{arm}_{st}_{D // 1000}k",
                P(st, D),
                D,
                f"outputs/menu_retrain_v0/tr{D // 1000}k_{arm}_{st}_r1",
                sh,
                extra.rstrip(),
                env=env,
            )

if "valdep" in ONLY or "hgdep" in ONLY:
    VAL = {
        "": f"{R}/outputs/val_variants/val_V0.npz",
        "hepg2": f"{R}/outputs/val_variants/val_V0_hepg2.npz",
        "diff": f"{R}/outputs/val_variants/val_V0_diff.npz",
    }
if "valdep" in ONLY:
    for D, sh in ((10000, 1), (100000, 3)):
        for st in RES:
            new += retrain(
                f"ind30w_{st}_{D // 1000}k",
                P(st, D),
                D,
                f"outputs/menu_retrain_v0/vv{D // 1000}k_V0c_{st}_r1",
                sh,
                f" --val_cache {VAL['']}",
                needs=[rel(VAL[""])],
            )
if "hgdep" in ONLY:
    for D, sh in ((10000, 1), (100000, 3)):
        for tgt in ("hepg2", "diff"):
            for st in RES:
                pool = f"{GP}/k562_hepg2/{tgt}/{st}_r1_d{D}.npz"
                cs = retrain(
                    f"ind30h_{tgt}_{st}_d{D // 1000}k",
                    pool,
                    D,
                    f"outputs/menu_retrain_v0/hg{D // 1000}k_{tgt}_{st}_r1",
                    sh,
                    f" --val_cache {VAL[tgt]}",
                    env={"HP_ALLOW_UNSTAMPED": "1"},
                    needs=[rel(VAL[tgt])],
                )
                new += cs
if "hptune" in ONLY:
    k3 = f"{R}/outputs/acq_oct26/menu_k3_random600k.json"
    if not os.path.exists(k3):
        rows = sorted(
            (
                (json.load(open(f))["val_pearson"], json.load(open(f))["menu_index"])
                for f in glob.glob(
                    f"{R}/outputs/menu_retrain_v0/nest_random_r1_d300k/r00_menu_*_meta.json"
                )
            ),
            reverse=True,
        )
        menu = json.load(open(MENU))
        top = [i for _, i in rows[:3]]
        json.dump([menu[i] for i in sorted(top)], open(k3, "w"), indent=1)
        print("hptune menu", sorted(top), [round(v, 4) for v, _ in rows[:3]])
    for arm, extra in {
        "lr05": " --lr_scale 0.5",
        "lr2": " --lr_scale 2.0",
        "bs4": " --bs_scale 4",
        "bs4lr2": " --bs_scale 4 --lr_scale 2.0",
    }.items():
        new += retrain(
            f"ind30t_{arm}_random600k",
            P("random", 600000),
            600000,
            f"outputs/menu_retrain_v0/hptune600k_{arm}_random_r1",
            3,
            extra,
            gpus=("h100",),
            menu=k3,
        )

reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
new = [c for c in new if c["name"] not in have]
for c in new:
    for p in c.get("needs", []):
        if not os.path.exists(p) and "k562_hepg2" not in p:
            raise SystemExit(f"{c['name']}: missing input {p}")
import collections  # noqa: E402

print(
    len(new),
    "new cells:",
    dict(collections.Counter(c["name"].split("_")[0] + "_" + c["name"].split("_")[1] for c in new)),
)
if DRY:
    sys.exit(0)
last_llm = max(i for i, c in enumerate(reg) if "LLM_MODEL" in c.get("env", {}))
llm_new = [c for c in new if "LLM_MODEL" in c["env"]]
gpu_new = [c for c in new if "LLM_MODEL" not in c["env"]]
reg = reg[: last_llm + 1] + llm_new + reg[last_llm + 1 :] + gpu_new
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new), "registry", len(reg))
