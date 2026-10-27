"""Oct-28 follow-ups to the overnight program: replicate the findings that rest on one seed, one reservoir or one pool series, and train the genomic
ensembles that are handed to collaborators.

ensemble  genomic 10k and 30k menu ensembles with saved weights (3 seeds, V0c validation, patience 8 and patience 16)        (ind30z_)
recipe2   patience 16 on replicate pools r2 and r3 (genomic 10k/30k/100k, random and motif 30k) and random/motif at 10k r1-r3   (ind30e_)
series    random and motif_ct_enriched at 10k and 100k on pools r1-r3, the series the unseen families were trained on           (un10_, un100_)
sc300     motif_shared_core at 300k: pool build + menu retrain, to compare with random / motif / evoaug / genomic at 300k      (pool100_sc300, mrr_)
llm2      separate-coder arm (agent loop without coordinator and reflector) on random 30k, motif 30k, genomic 10k pools;
          coordinator arm on evoaug and evo2 10k pools                                                                         (ag30_p_, an30_)

libcheck  the library (albench menu train / stack) retrains 3 entries on the genomic 10k pool, to compare with the experiment script's members        (ind30z_)

  python scripts/program_oct22/register_oct28.py [--only ensemble,recipe2,series,sc300,llm2,libcheck] [--dry]
"""

import collections
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
    else ["ensemble", "recipe2", "series", "sc300", "llm2", "libcheck"]
)
DRY = "--dry" in sys.argv
S55 = "claude-sonnet-5-5"
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"
OUT22 = "outputs/archsearch_reps_oct22"
VAL0C = f"{R}/outputs/val_variants/val_V0.npz"
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
RES = ["genomic", "random", "motif_ct_enriched"]
CAP = {10000: 15, 30000: 25, 100000: 60, 300000: 120}
P = lambda st, D, r=1: f"{REPS}/{st}_r{r}_d{D}.npz"  # noqa: E731
new = []


def retrain(name, pool, D, out, n_sh, extra="", gpus=("h100", "v100"), env=None, needs=()):
    return [
        dict(
            name=f"{name}_s{k}",
            gpus=list(gpus),
            ext=[],
            out_dir=out,
            env=dict(env or {}),
            needs=sorted({rel(pool), *needs}),
            cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {MENU} --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --shard {k}/{n_sh} --base_cap_min {CAP[D]}{extra}",
            done=dict(type="file", path=f"{out}/COMPLETE_{k}of{n_sh}"),
        )
        for k in range(n_sh)
    ]


def llm(name, pool, out, D, extra):
    return dict(
        name=name,
        gpus=["v100"],
        ext=[],
        out_dir=out,
        env={"LLM_MODEL": S55},
        needs=[rel(pool)],
        cmd=f"uv run --no-sync python scratchpad/arch_autoresearch.py --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --rounds 10 --per_round 2 --max_params 40000000 "
        f"--base_cap_min {15 if D == 10000 else 25} --budget_hours {{BUDGET}} --guidance known_good --ensemble_aware{extra} --resume",
        done=dict(type="file", path=f"{out}/COMPLETE"),
    )


if "ensemble" in ONLY:
    for D in (10000, 30000):
        for pat in (8, 16):
            new += retrain(
                f"ind30z_ens_genomic_{D // 1000}k_p{pat}",
                f"{GP}/k562_nested/genomic_d{D}.npz",
                D,
                f"outputs/menu_retrain_v0/ens{D // 1000}k_genomic_p{pat}",
                3,
                f" --seeds 3 --save_models --val_cache {VAL0C}",
                env={"AL_PATIENCE": str(pat), "HP_ALLOW_UNSTAMPED": "1"}
                if pat != 8
                else {"HP_ALLOW_UNSTAMPED": "1"},
                needs=[rel(VAL0C)],
            )

if "recipe2" in ONLY:
    cells = [(s, 30000, 3) for s in RES] + [("genomic", 10000, 1), ("genomic", 100000, 3)]
    cells += [(s, 10000, 1) for s in ("random", "motif_ct_enriched")]
    for st, D, sh in cells:
        for r in (2, 3):
            new += retrain(
                f"ind30e_pat16_{st}_{D // 1000}k_r{r}",
                P(st, D, r),
                D,
                f"outputs/menu_retrain_v0/tr{D // 1000}k_pat16_{st}_r{r}",
                sh,
                env={"AL_PATIENCE": "16"},
            )
    # the 10k random / motif arms also need pool r1 (the genomic 10k arm already has it)
    for st in ("random", "motif_ct_enriched"):
        new += retrain(
            f"ind30e_pat16_{st}_10k_r1",
            P(st, 10000, 1),
            10000,
            f"outputs/menu_retrain_v0/tr10k_pat16_{st}_r1",
            1,
            env={"AL_PATIENCE": "16"},
        )

if "series" in ONLY:
    for D, tag, pre in ((10000, "un10k", "un10"), (100000, "un100k", "un100")):
        for st in ("random", "motif_ct_enriched"):
            for r in (1, 2, 3):
                new += retrain(
                    f"{pre}_{st}_r{r}",
                    P(st, D, r),
                    D,
                    f"outputs/menu_retrain_v0/{tag}_{st}_r{r}",
                    3,
                )

if "sc300" in ONLY:
    RAW300 = f"{GP}/k562_reps_raw300k"
    pool = P("motif_shared_core", 300000, 1)
    new.append(
        dict(
            name="pool100_sc300_motif_shared_core",
            gpus=["h100"],
            ext=[],
            out_dir=rel(RAW300),
            env={"AG_S2_ORACLE_DIR": f"{R}/outputs/oracle_v3/s2"},
            cmd=f"uv run --no-sync python scripts/build_pool.py --strategy motif_shared_core --size 300000 --generation-seed 43 --out-dir {RAW300}; "
            f"uv run --no-sync python -c \"import sys; sys.path.insert(0, 'scratchpad'); from stage2_lib import *; print(slice_unseen(300000, ['motif_shared_core'], reps=(1,)))\"",
            done=dict(type="file", path=rel(pool)),
        )
    )
    new += retrain(
        "mrr_nest_motif_shared_core_300k",
        pool,
        300000,
        "outputs/menu_retrain_v0/nest_motif_shared_core_r1_d300k",
        12,
    )

if "llm2" in ONLY:
    PLAIN = (
        " --loop agent --agent_cfg '"
        + json.dumps(dict(stages=["propose", "code", "evaluate"]))
        + "'"
    )
    COORD = (
        " --loop agent --agent_cfg '"
        + json.dumps(dict(stages=["propose", "coordinate", "code", "evaluate"], ideas_per_round=6))
        + "'"
    )
    for tag, st, D in (
        ("rnd", "random", 30000),
        ("mot", "motif_ct_enriched", 30000),
        ("gen10", "genomic", 10000),
    ):
        for r in range(1, 7):
            new.append(
                llm(
                    f"ag30_p_{tag}_r{r}",
                    P(st, D, r),
                    f"{OUT22}/{st}_{D // 1000}k_agent_plain/rep{r}",
                    D,
                    PLAIN,
                )
            )
    for tag, st, pool_of in (
        ("e10", "evoaug_paper2023", lambda r: P("evoaug_paper2023", 10000, r)),
        ("v10", "evo2", lambda r: f"{GP}/k562_cumulative/evo2_d10000.npz"),
    ):
        for r in (1, 2, 3):
            new.append(
                llm(
                    f"an30_{tag}_r{r}",
                    pool_of(r),
                    f"{OUT22}/{st}_10k_agent_coord_noref/rep{r}",
                    10000,
                    COORD,
                )
            )

if "libcheck" in ONLY:
    # the library (albench menu train / stack) against the experiment script on the same nested genomic 10k pool: the first 3 entries of the priority order,
    # seed 0 and the same entry indices, so each member should match its r00 counterpart in ens10k_genomic_p8 up to GPU non-determinism
    out = "outputs/menu_retrain_v0/libcheck_10k"
    new.append(
        dict(
            name="ind30z_libcheck_10k",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=out,
            env={},
            needs=[rel(VAL0C)],
            cmd=f"uv run --no-sync python -m albench.cli menu train --size 10000 --pool {GP}/k562_nested/genomic_d10000.npz --val {VAL0C} --out {R}/{out} --k 3 --seeds 1 --patience 8 && "
            f"uv run --no-sync python -m albench.cli menu stack --out {R}/{out} --val {VAL0C} --size 10000",
            done=dict(type="file", path=f"{out}/ensemble.json"),
        )
    )

reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
new = [c for c in new if c["name"] not in have]
for c in new:
    for p in c.get("needs", []):
        if not os.path.exists(p) and "motif_shared_core_r1_d300000" not in p:
            raise SystemExit(f"{c['name']}: missing input {p}")
print(
    len(new),
    "new cells:",
    dict(collections.Counter("_".join(c["name"].split("_")[:2]) for c in new)),
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
