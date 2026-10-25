"""Oct-26 non-LLM experiments (menu retrains and pool builds; they need no LLM budget, so they run while the usage limit is paused).

noise    assay-noise test: menu retrained on labels with Gosai-sized measurement error (scale 1) at 10k/30k/100k and double error (scale 2) at 30k,
         genomic / random / motif_ct_enriched replicate r1; test labels stay clean oracle values.
seeds    nine seeds on one pool: seeds 3-8 added to the three seeds already trained on pool r1 at 30k (ind30k_*_r1, *_seeds12 equivalents), 3 reservoirs.
ladder   nested three-seed ladder on pool r1 at 10k and 100k (30k exists): seeds 0-2 (genomic: seeds 1-2, seed 0 is nest_genomic_r1_*).
zoo      Zoonomia rebuilt after the validation-chromosome fix: raw 30k pools (seeds 43-45), slice 10k/30k as zoonomia_orthologs_v2, menu retrain at both.
anchor   measured high-D anchors: random and motif_ct_enriched pools of 1M (seed 43; 600k is its prefix), menu retrain at 600k and 1M. H100 only.
acq      acquisition dry run: candidates, scoring cells (menu retrained at 30k genomic r1 with --score_pool). Downstream cells are registered after scoring.

  python scripts/program_oct22/register_oct26.py [--only noise,seeds,...] [--dry]
"""

import collections
import json
import os
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import GP, REPS, SEED, R  # noqa: E402

from albench.pools import PoolSpec  # noqa: E402

os.chdir(R)
DRY = "--dry" in sys.argv
ONLY = (
    sys.argv[sys.argv.index("--only") + 1].split(",")
    if "--only" in sys.argv
    else ["noise", "seeds", "ladder", "zoo", "anchor", "acq"]
)
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"
HV, H = ["h100", "v100"], ["h100"]
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
RES = ["genomic", "random", "motif_ct_enriched"]
CAP = {10000: 15, 30000: 25, 100000: 60, 600000: 240, 1000000: 360}


def retrain(name, pool, D, out, n_sh, extra="", gpus=HV, menu=MENU, cap=None, needs=()):
    cells = []
    for k in range(n_sh):
        cells.append(
            dict(
                name=f"{name}_s{k}",
                gpus=gpus,
                ext=[],
                out_dir=out,
                env={},
                needs=sorted({rel(pool), *needs}),
                cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {menu} --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --shard {k}/{n_sh} "
                f"--base_cap_min {cap or CAP[D]}{extra}",
                done=dict(type="file", path=f"{out}/COMPLETE_{k}of{n_sh}"),
            )
        )
    return cells


P = lambda st, D, r=1: f"{REPS}/{st}_r{r}_d{D}.npz"  # noqa: E731
new = []
if "noise" in ONLY:
    for st in RES:
        for D, sh in ((30000, 3), (10000, 1), (100000, 3)):
            new += retrain(
                f"ind30n1_{st}_{D // 1000}k",
                P(st, D),
                D,
                f"outputs/menu_retrain_v0/nz1_{st}_r1_d{D // 1000}k",
                sh,
                " --label_noise_scale 1",
            )
    for st in RES:
        new += retrain(
            f"ind30n2_{st}_30k",
            P(st, 30000),
            30000,
            f"outputs/menu_retrain_v0/nz2_{st}_r1_d30k",
            3,
            " --label_noise_scale 2",
        )
if "seeds" in ONLY:
    for st in RES:
        new += retrain(
            f"ind30s_{st}",
            P(st, 30000),
            30000,
            f"outputs/menu_retrain_v0/ind30k_{st}_r1_seeds3to8",
            12,
            " --seeds 6 --seed_offset 3",
        )
if "ladder" in ONLY:
    for st in RES:
        for D in (10000, 100000):
            if st == "genomic":
                new += retrain(
                    f"ind30l_{st}_{D // 1000}k",
                    P(st, D),
                    D,
                    f"outputs/menu_retrain_v0/lad_{st}_r1_d{D // 1000}k_seeds12",
                    4 if D == 10000 else 6,
                    " --seeds 2 --seed_offset 1",
                )
            else:
                new += retrain(
                    f"ind30l_{st}_{D // 1000}k",
                    P(st, D),
                    D,
                    f"outputs/menu_retrain_v0/lad_{st}_r1_d{D // 1000}k",
                    4 if D == 10000 else 6,
                    " --seeds 3",
                )
if "zoo" in ONLY:
    RAW = f"{GP}/k562_reps_raw30k_zoo2"
    build = " ; ".join(
        f"uv run --no-sync python scripts/build_pool.py --strategy zoonomia_orthologs --size 30000 --generation-seed {SEED(r)} --out-dir {RAW}"
        for r in (1, 2, 3)
    )
    new.append(
        dict(
            name="pool30_zoo2",
            gpus=H,
            ext=[],
            out_dir=rel(RAW),
            env={"AG_S2_ORACLE_DIR": f"{R}/outputs/oracle_v3/s2"},
            cmd=f"{build} ; uv run --no-sync python scripts/program_oct22/slice_pool.py --raw {RAW} --strategy zoonomia_orthologs --name zoonomia_orthologs_v2 --size 30000 --reps 1 2 3 --prefix 10000 30000",
            done=dict(
                type="file", path="outputs/grid_pools/k562_reps/zoonomia_orthologs_v2_r3_d30000.npz"
            ),
        )
    )
    for r in (1, 2, 3):
        for D, sh in ((30000, 3), (10000, 1)):
            new += retrain(
                f"un{D // 1000}_zoov2_r{r}",
                P("zoonomia_orthologs_v2", D, r),
                D,
                f"outputs/menu_retrain_v0/un{D // 1000}k_zoonomia_v2_r{r}",
                sh,
            )
if "anchor" in ONLY:
    RAW = f"{GP}/k562_reps_raw1m"
    for st in ("random", "motif_ct_enriched"):
        new.append(
            dict(
                name=f"pool1m_{st}",
                gpus=H,
                ext=[],
                out_dir=rel(RAW),
                env={"AG_S2_ORACLE_DIR": f"{R}/outputs/oracle_v3/s2"},
                cmd=f"uv run --no-sync python scripts/build_pool.py --strategy {st} --size 1000000 --generation-seed {SEED(1)} --out-dir {RAW} ; "
                f"uv run --no-sync python scripts/program_oct22/slice_pool.py --raw {RAW} --strategy {st} --size 1000000 --reps 1 --prefix 600000 1000000",
                done=dict(type="file", path=f"outputs/grid_pools/k562_reps/{st}_r1_d1000000.npz"),
            )
        )
    for D, tag in ((600000, "a600"), (1000000, "a1m")):
        for st in ("random", "motif_ct_enriched"):
            new += retrain(
                f"{tag}_{st}",
                P(st, D),
                D,
                f"outputs/menu_retrain_v0/anchor_{st}_r1_d{D // 1000}k",
                17,
                gpus=H,
            )
if "acq" in ONLY:
    # candidates: first 100k of the random, motif_ct_enriched and evoaug r1 pools (labelled by oracle_v3), built once on the login node by acq_dryrun.py candidates
    cand = f"{GP}/k562_acq/candidates_r1.npz"
    new += retrain(
        "ind30a_score_genomic",
        P("genomic", 30000),
        30000,
        "outputs/menu_retrain_v0/acq30k_genomic_r1",
        3,
        f" --score_pool {cand}",
        needs=[rel(cand)],
    )

reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
new = [c for c in new if c["name"] not in have]
PRIO = (
    "ind30a_",
    "pool",
    "a600_",
    "a1m_",
    "ind30n",
    "un10_zoov2",
    "un30_zoov2",
    "ind30s",
    "ind30l",
)  # earlier = first in the registry = first to get a GPU slot
new.sort(key=lambda c: next(i for i, pfx in enumerate(PRIO) if c["name"].startswith(pfx)))
for c in new:
    for p in c.get("needs", []):
        if not os.path.exists(p) and not any(
            x in p for x in ("_d600000", "_d1000000", "zoonomia_orthologs_v2", "candidates_r1")
        ):
            raise SystemExit(f"{c['name']}: missing input {p}")
print(
    "new cells:",
    dict(collections.Counter(c["name"].rsplit("_s", 1)[0] for c in new)),
    "total",
    len(new),
)
if DRY:
    sys.exit(0)
reg += new
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new), "cells; registry", len(reg))
