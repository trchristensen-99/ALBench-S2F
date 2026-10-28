"""Oct-28 night: preliminary scaling curves and the 30k-genomic top-up (acquisition) experiment, with the one menu (v1, 20 configurations) for every strategy.

Two designs, both trained with menu v1 at patience 16, one seed, validation V0c, test sets genomic / designed OOD / random 32k:

scale31   each strategy trained on its own canonical nested pool at 10k, 30k, 100k (10k and 30k: all 20 configurations; 100k: the 12 of highest
          priority, MENU_MAX_N=12, extendable later because model files are named by menu position)
          strategies: genomic, zoonomia_orthologs, evoaug_paper2023, mutagenesis_mr10 (10% per-base random mutagenesis), motif_ct_enriched,
          motif_shared_core, random (uniform), random_gc_sweep (equal batches at GC 25/35/45/55/65/75%)                  (sc31_)
al31      30k genomic (replicate 1) + 70k = 100k, 12 configurations, patience 16: 70k more genomic (control), 70k from one reservoir
          (random, motif_ct_enriched, evoaug_paper2023, motif_shared_core, zoonomia_orthologs, random_gc_sweep, mutagenesis_mr10), 70k at random
          from the even mix of random / motif_ct_enriched / evoaug_paper2023 (the candidate pool), and 70k selected from that mix by the
          30k-genomic ensemble: highest mean, highest SD, SD excess over the activity trend, SD within activity bins, cluster-balanced (diversity), SD then
          diversity, BADGE-like k-means++ on member disagreement                                                         (al31_)

  python scripts/program_oct22/register_oct30.py [--dry]

Registry order = priority: new pools, scale 10k, the AL arms, scale 30k go ahead of the remaining Oct-28/29 retrains, 100k scale cells and the large
width-scaled / 300k cells go after the motif-menu cells (they matter less for the decisions the PI will question). Backup: dispatch_registry.json.bak_oct30.
"""

import json
import os
import re
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import GP, REPS, R  # noqa: E402

os.chdir(R)
DRY = "--dry" in sys.argv
rel = lambda p: os.path.relpath(p, R)  # noqa: E731
V1 = f"{R}/outputs/menus_oct22/V1.json"
NEST, ACQ = f"{GP}/k562_nested", f"{GP}/k562_acq"
RAW = f"{GP}/k562_reps_raw100k"
CAP = {10000: 15, 30000: 25, 100000: 60}
SHARDS = {10000: 2, 30000: 4, 100000: 6}
STRATS = [
    "genomic",
    "random",
    "motif_ct_enriched",
    "evoaug_paper2023",
    "zoonomia_orthologs",
    "motif_shared_core",
    "random_gc_sweep",
    "mutagenesis_mr10",
]
NEW_POOLS = {  # name -> (strategy, extra build args)
    "random_gc_sweep": ("random_gc_sweep", ""),
    "mutagenesis_mr10": ("mutagenesis", " --set mutation_rate=0.1"),
}
AL_ARMS = {  # arm -> training pool
    "genomic": f"{REPS}/genomic_r1_d100000.npz",
    **{
        f"res_{s}": f"{ACQ}/res_{s}_d100000.npz"
        for s in (
            "random",
            "motif_ct_enriched",
            "evoaug_paper2023",
            "motif_shared_core",
            "zoonomia_orthologs",
            "random_gc_sweep",
            "mutagenesis_mr10",
        )
    },
    "mix_random": f"{ACQ}/sel_rand_d100000.npz",
    "mix_mean": f"{ACQ}/sel_fullmean_d100000.npz",
    "mix_std": f"{ACQ}/sel_fullstd_d100000.npz",
    **{
        f"mix_{a}": f"{ACQ}/sel_{a}_d100000.npz"
        for a in ("resid", "binned", "diverse", "stdiv", "badge")
    },
}


def pool_cells():
    out = []
    for name, (strat, extra) in NEW_POOLS.items():
        cmd = (
            f"uv run --no-sync python scripts/build_pool.py --strategy {strat} --size 100000 --generation-seed 42 --out-dir {RAW}{extra} && "
            f"uv run --no-sync python scripts/program_oct22/slice_nested_new.py --strategy {strat} --name {name} --raw {RAW}{extra}"
        )
        out.append(
            dict(
                name=f"pool31_{name}",
                gpus=["h100"],
                ext=[],
                out_dir=rel(RAW),
                env={"AG_S2_ORACLE_DIR": f"{R}/outputs/oracle_v3/s2"},
                cmd=cmd,
                done=dict(type="file", path=rel(f"{ACQ}/res_{name}_d100000.npz")),
            )
        )
    return out


def retrain(name, pool, D, out, n_sh):
    env = {"AL_PATIENCE": "16"}
    if D >= 100000:
        env["MENU_MAX_N"] = "12"
    return [
        dict(
            name=f"{name}_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=out,
            env=dict(env),
            needs=[rel(pool)],
            cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {V1} --reservoir_cache {pool} --D {D} --out_dir {R}/{out} --shard {k}/{n_sh} --base_cap_min {CAP[D]}",
            done=dict(type="file", path=f"{out}/COMPLETE_{k}of{n_sh}"),
        )
        for k in range(n_sh)
    ]


def scale_cells(D):
    out = []
    for st in STRATS:
        out += retrain(
            f"sc31_{st}_{D // 1000}k",
            f"{NEST}/{st}_d{D}.npz",
            D,
            f"outputs/menu_retrain_v0/sc31_{D // 1000}k_{st}",
            SHARDS[D],
        )
    return out


def al_cells():
    out = []
    for arm, pool in AL_ARMS.items():
        out += retrain(f"al31_{arm}", pool, 100000, f"outputs/menu_retrain_v0/al31_{arm}", 6)
    return out


def main():
    reg = json.load(open("scratchpad/dispatch_registry.json"))
    have = {c["name"] for c in reg}
    groups = {
        "pools": [c for c in pool_cells() if c["name"] not in have],
        "sc10": [c for c in scale_cells(10000) if c["name"] not in have],
        "sc30": [c for c in scale_cells(30000) if c["name"] not in have],
        "sc100": [c for c in scale_cells(100000) if c["name"] not in have],
        "al": [c for c in al_cells() if c["name"] not in have],
    }
    # pull the cells whose priority changes out of the registry, then put everything back in the intended order
    pick = lambda pat: [c for c in reg if re.match(pat, c["name"])]  # noqa: E731
    moved_as = pick(
        r"as_M"
    )  # motif-menu cells and held-out reservoir retrains (and the earlier as_ cells, kept together)
    moved_x2 = pick(r"scl30_x2_")
    moved_x3 = pick(r"scl30_x3_")
    moved_mrr = pick(r"mrr_nest_motif_shared_core")
    moved = {c["name"] for c in moved_as + moved_x2 + moved_x3 + moved_mrr}
    rest = [c for c in reg if c["name"] not in moved]
    i_v1 = max(i for i, c in enumerate(rest) if c["name"].startswith("ind30m_v1_"))
    head, tail = rest[: i_v1 + 1], rest[i_v1 + 1 :]
    reg2 = (
        head
        + groups["pools"]
        + groups["sc10"]
        + groups["al"]
        + groups["sc30"]
        + tail
        + moved_as
        + groups["sc100"]
        + moved_x2
        + moved_mrr
        + moved_x3
    )
    assert len(reg2) == len(reg) + sum(len(v) for v in groups.values())
    assert len({c["name"] for c in reg2}) == len(reg2)
    print(
        {k: len(v) for k, v in groups.items()},
        "moved",
        len(moved),
        "registry",
        len(reg),
        "->",
        len(reg2),
    )
    if DRY:
        return
    shutil.copy("scratchpad/dispatch_registry.json", "scratchpad/dispatch_registry.json.bak_oct30")
    json.dump(reg2, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
    os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
    print("registered")


if __name__ == "__main__":
    main()
