"""Nov-4 (afternoon): the experiments that decide the per-size student before the reservoir scaling runs. All cells use harness H3 (patience off, fp16, single-pass RC augmentation) and the standard input.

1. C219 across sizes and pools (`c219_<D>k_<pool>_s<k>`): the collaborator's ShuffleNet-style C219 is the one non-LegNet architecture that adds to the tuned bank on the same data (+0.004 genomic at 30k,
   `analyze_pi_stack.py`). 5 seeds on the canonical pool and on replicate pools r1-r3, at 10k (600 epochs), 30k and 100k (300 epochs), gentler recipe (AdamW lr 0.003, wd 0.1, batch 512, one-cycle).
   With the same-pool banks (`register_nov4c.py`) this says whether the gain holds per pool and per size. 100k cells are named `mrr_` so they stay off MIG slices.
2. Reservoir fairness at 30k (`fb30_*`, `m3_30k_*`, `rcr30_*`): does the procedure favour genomic training data? On 3 replicate pools each of random, motif_ct_enriched and GRM: the tuned bank (20 seeds, e500 wd 0.2)
   and menu v3 (`build_menu_v3.py`: 10 loop architectures + C219 + C213, 1 seed); menu v3 also on the genomic canonical and r1-r3 pools. Read-out: the winning student family per test set and whether reservoir
   rankings agree between the bank, the menu and their union. Plus a recipe check of the bank on random_r1 (300 / 800 epochs at wd 0.2, 500 at wd 0.1 and 0.4; 5 seeds), since the recipe was tuned on genomic data.
3. The bank recipe at the edges (`rce10_*`, `rce300_*`): 10k at 1,200 / 1,600 epochs (wd 0.4) and 800 / 1,200 (wd 0.6); 300k at 100 / 200 / 300 epochs (wd 0.1) and 300 (wd 0.05). 5 seeds, canonical pool.

About 140 H100-hours in all. The menu-v3 cells wait for `outputs/menus_nov4/MENU_V3_30k.json` (their `needs`).

    python scripts/program_oct22/register_nov4d.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_fx35b as fb  # noqa: E402
import register_nov2b as nb  # noqa: E402

R, NEST, DRY = nb.R, nb.NEST, nb.DRY
OUT = "outputs/menu_retrain_v0"
REPS = "outputs/grid_pools/k562_reps"
F = 20
H3 = {"AL_PATIENCE": "9999", "AL_PRECISION": "fp16", "AL_RC_MODE": "aug"}
SEED_FILE = f"{R}/outputs/bench33/seeds_nov4d.json"
MENU_V3 = f"{R}/outputs/menus_nov4/MENU_V3_30k.json"
FAIR = ("random", "motif_ct_enriched", "GRM")
_SEEDS = json.load(open(SEED_FILE)) if os.path.exists(SEED_FILE) else {}


def seeds(key, n):
    if key not in _SEEDS:
        _SEEDS[key] = [int.from_bytes(os.urandom(3), "big") for _ in range(n)]
    return _SEEDS[key]


def rel(p):
    return p[len(R) + 1 :] if p.startswith(R + "/") else p


def bank(name, D, cfg, pool, n, gpus):
    """One seed per job of the published LegNet with a bench33 recipe (native trainer), as register_nov4a/c."""
    tag = f"_{os.path.basename(pool).split('_d')[0]}" if pool else ""
    odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}{tag}"
    sd = seeds(f"{name}", n)
    return [
        dict(
            name=f"{name}_s{j}",
            gpus=gpus,
            ext=[],
            out_dir=odir,
            env={},
            needs=[pool or f"{NEST}/genomic_d{D}.npz"],
            cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {cfg}"
            + (f" --pool {R}/{pool}" if pool else "")
            + f" --seeds {j}={sd[j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
            done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
        )
        for j in range(n)
    ]


def c219_cells():
    base = next(
        e for e in json.load(open(f"{R}/outputs/menus_nov3/PI41_mid_30k.json")) if "c219" in e["id"]
    )
    out = []
    for D, epochs, cap in ((10000, 600, 150), (30000, 300, 200), (100000, 300, 480)):
        menu = f"{R}/outputs/menus_nov4/C219_{D // 1000}k.json"
        if not DRY:
            os.makedirs(os.path.dirname(menu), exist_ok=True)
            json.dump(
                [{**base, "train_config": {**base["train_config"], "epochs": epochs}}],
                open(menu, "w"),
                indent=1,
            )
        pools = {
            "can": f"{NEST}/genomic_d{D}.npz",
            **{f"r{r}": f"{REPS}/genomic_r{r}_d{D}.npz" for r in (1, 2, 3)},
        }
        for tag, pool in pools.items():
            odir = f"{OUT}/c219_{D // 1000}k_{tag}"
            for k in range(5):
                out.append(
                    dict(
                        name=f"{'mrr_' if D >= 100000 else ''}c219_{D // 1000}k_{tag}_s{k}",
                        gpus=["h100"] if D >= 100000 else ["h100", "v100"],
                        ext=[],
                        out_dir=odir,
                        env={**H3, "MENU_MAX_FWD_MS": "1000000000"},
                        needs=[pool, rel(menu)],
                        cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {menu} --reservoir_cache {pool} --D {D} --out_dir {R}/{odir} --shard 0/1 --base_cap_min {cap} --val_cache {fb.V0C}{nb.STD} --seed_offset {k} && touch {R}/{odir}/COMPLETE_seed{k}",
                        done=dict(type="file", path=f"{odir}/COMPLETE_seed{k}"),
                    )
                )
    return out


def fairness_cells():
    D, out = 30000, []
    for res in FAIR:
        for r in (1, 2, 3):
            pool = f"{REPS}/{res}_r{r}_d{D}.npz"
            out += bank(f"fb30_{res}_r{r}", D, "e500_wd2", pool, 20, ["h100", "v100"])
    menu_pools = {f"{res}_r{r}": f"{REPS}/{res}_r{r}_d{D}.npz" for res in FAIR for r in (1, 2, 3)}
    menu_pools.update(
        {
            "genomic_can": f"{NEST}/genomic_d{D}.npz",
            **{f"genomic_r{r}": f"{REPS}/genomic_r{r}_d{D}.npz" for r in (1, 2, 3)},
        }
    )  # python 3.6 on the login node
    for tag, pool in menu_pools.items():
        out += fb.cells(f"m3_30k_{tag}", MENU_V3, pool, D, H3, nb.STD, shards=6, cap=200, seeds=1)
    for cfg in ("e300_wd2", "e800_wd2", "e500", "e500_wd4"):
        out += bank(f"rcr30_{cfg}", D, cfg, f"{REPS}/random_r1_d{D}.npz", 5, ["h100", "v100"])
    return out


def edge_cells():
    out = []
    for cfg in ("e1200_wd4", "e1600_wd4", "e800_wd6", "e1200_wd6"):
        out += bank(f"rce10_{cfg}", 10000, cfg, None, 5, ["h100", "v100"])
    for cfg in ("e100", "e200", "e300", "e300_wd05"):
        out += bank(f"rce300_{cfg}", 300000, cfg, None, 5, ["h100"])
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        cells = c219_cells() + fairness_cells() + edge_cells()
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells at the end | registry", len(reg), "->", len(reg2))
        for pre in ("c219_", "mrr_c219_", "fb30_", "m3_30k_", "rcr30_", "rce10_", "rce300_"):
            print(f"  {pre}: {sum(1 for c in new if c['name'].startswith(pre))}")
        if DRY:
            for pre in ("c219_30k", "m3_30k_random_r1", "rce300_e300_s0"):
                c = next(c for c in new if c["name"].startswith(pre))
                print(c["name"], c["gpus"], c["needs"], "\n ", c["cmd"][:460])
            return
        json.dump(_SEEDS, open(SEED_FILE, "w"))
        shutil.copy(path, path + ".bak_nov4d")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
