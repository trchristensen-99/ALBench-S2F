"""Oct-29 experiments from the review of the ledger: how far patience helps, and whether the architecture search should be spread over reservoirs.

patience2  patience 24 and 32 against 16 and 8 (genomic 10k/30k/100k, random 30k, motif 30k; pool r1)                         (ind30e_)
archsrc2   the 15 motif_ct_enriched search runs of the 30k program become a third menu (MM, 45 candidates, top 3 of each run), retrained on the
           four 30k pools the genomic (MG) and random (MR) menus were retrained on (genomic, random, motif, GRM), and all three menus are retrained
           on two reservoirs none of the 45 searches used (evoaug, evo2) to see what carries over                                (as_)

scale      do larger models help? width x2 and x3 copies (about 4x and 9x the parameters) of the best 8 configurations, written by the proposer model, retrained on genomic,
           random and motif at 100k, and the best 4 at x2 on random and motif at 300k, with a time cap that is not the binding constraint, paired with the unscaled
           retrains of the same configurations                                                                                  (scl30_)

  python scripts/program_oct22/register_oct29.py [--only patience2,archsrc2,scale] [--dry]
"""

import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import GP, REPS, R  # noqa: E402

os.chdir(R)
ONLY = (
    sys.argv[sys.argv.index("--only") + 1].split(",")
    if "--only" in sys.argv
    else ["patience2", "archsrc2", "scale"]
)
DRY = "--dry" in sys.argv
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"
O22 = "outputs/archsearch_reps_oct22"
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
P = lambda st, D, r=1: f"{REPS}/{st}_r{r}_d{D}.npz"  # noqa: E731
CAP = {10000: 15, 30000: 25, 100000: 60, 300000: 120}
new = []


def retrain(
    name, pool, D, out, n_sh, menu, extra="", env=None, gpus=("h100", "v100"), cap=None, needs=()
):
    return [
        dict(
            name=f"{name}_s{k}",
            gpus=list(gpus),
            ext=[],
            out_dir=out,
            env=dict(env or {}),
            needs=sorted({rel(pool), *needs}),
            cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {menu} --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --shard {k}/{n_sh} --base_cap_min {cap or CAP[D]}{extra}",
            done=dict(type="file", path=f"{out}/COMPLETE_{k}of{n_sh}"),
        )
        for k in range(n_sh)
    ]


if "patience2" in ONLY:
    for pat in (24, 32):
        for st, D, sh in (
            ("genomic", 10000, 1),
            ("genomic", 30000, 3),
            ("random", 30000, 3),
            ("motif_ct_enriched", 30000, 3),
            ("genomic", 100000, 3),
        ):
            new += retrain(
                f"ind30e_pat{pat}_{st}_{D // 1000}k",
                P(st, D),
                D,
                f"outputs/menu_retrain_v0/tr{D // 1000}k_pat{pat}_{st}_r1",
                sh,
                MENU,
                env={"AL_PATIENCE": str(pat)},
            )

if "archsrc2" in ONLY:
    mm = "outputs/menus_oct22/MM.json"
    if not os.path.exists(mm):
        dirs = [f"{O22}/motif_ct_enriched_30k/rep{r}" for r in range(1, 16)]
        assert all(os.path.exists(d + "/COMPLETE") for d in dirs)
        r = subprocess.run(
            [
                f"{R}/.venv/bin/python",
                "scratchpad/build_menu.py",
                "--cells",
                *dirs,
                "--k",
                "3",
                "--first_n",
                "20",
                "--no_reference",
                "--out",
                mm,
            ],
            capture_output=True,
            text=True,
        )
        print((r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1])
    targets = {
        "genomic": P("genomic", 30000),
        "random": P("random", 30000),
        "motif": P("motif_ct_enriched", 30000),
        "grm": P("GRM", 30000),
    }
    held_out = {
        "evoaug": P("evoaug_paper2023", 30000),
        "evo2": f"{GP}/k562_cumulative/evo2_d30000.npz",
    }
    jobs = [("MM", t, pool) for t, pool in targets.items()] + [
        (c, t, pool) for c in ("MG", "MR", "MM") for t, pool in held_out.items()
    ]
    for cond, t, pool in jobs:
        assert os.path.exists(pool), pool
        new += retrain(
            f"as_{cond}_{t}",
            pool,
            30000,
            f"outputs/menu_retrain_v0/as30k_{cond}_{t}",
            3,
            f"{R}/outputs/menus_oct22/{cond}.json",
        )

if "scale" in ONLY:
    new.append(
        dict(
            name="scl30_gen",
            gpus=["v100"],
            ext=[],
            out_dir="outputs/scaleup",
            env={"LLM_MODEL": "claude-sonnet-5-5"},
            cmd="uv run --no-sync python scripts/program_oct22/scale_configs.py --out outputs/scaleup",
            done=dict(type="file", path="outputs/scaleup/DONE"),
        )
    )
    for f in (2, 3):
        for t, st in (("genomic", "genomic"), ("random", "random"), ("motif", "motif_ct_enriched")):
            new += retrain(
                f"scl30_x{f}_{t}_100k",
                P(st, 100000),
                100000,
                f"outputs/menu_retrain_v0/scl100k_x{f}_{t}_r1",
                4,
                f"{R}/outputs/scaleup/menu_x{f}.json",
                cap=120,
                needs=[f"outputs/scaleup/menu_x{f}.json"],
            )
    for t, st in (("random", "random"), ("motif", "motif_ct_enriched")):
        new += retrain(
            f"scl30_x2_{t}_300k",
            P(st, 300000),
            300000,
            f"outputs/menu_retrain_v0/scl300k_x2_{t}_r1",
            4,
            f"{R}/outputs/scaleup/menu_x2_top4.json",
            cap=240,
            needs=["outputs/scaleup/menu_x2_top4.json"],
        )

reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
new = [c for c in new if c["name"] not in have]
print(
    len(new),
    "new cells:",
    dict(collections.Counter("_".join(c["name"].split("_")[:3]) for c in new)),
)
if DRY:
    sys.exit(0)
reg = reg + new
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new), "registry", len(reg))
