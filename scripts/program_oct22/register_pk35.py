"""Oct-29: where does a collaborator's single-model gain come from, and does real flank context help? (cells pk35_, fx35_, lg35_)

A collaborator's own AutoResearch search reached a best single model of 0.8165 on the 30k genomic test (ours: 0.760 by validation). His best model is a LegNet with weight decay 2e-4, weight averaging
(decay 0.9995) with BatchNorm recalculation, and half the dilation in the later stages. Weight averaging and BN recalculation are harness options we did not have; the arms below measure each on our own
menu, one switch at a time, paired with hp34_base_30k (same 20 configurations, same seed, same V0c validation, same canonical genomic 30k pool). Seed-noise floor: hp34s1_base_30k minus hp34_base_30k.

  pk35_<arm>_30k   menu v1 with a harness switch: ema (AL_EMA=0.9995) | bn (AL_BN_RECALC=1 alone) | emabn (both: the collaborator's combination) | ema999bn (decay 0.999 + recalculation)
  fx35_<arm>_30k   menu v1 with real plasmid flank as INPUT (the oracle sees 600 bp: 200 insert + plasmid on each side): c0s30 = no extra input but the training shift runs +-30 over real bases;
                   c20s10 / c60s10 / c100s10 = 20 / 60 / 100 real bases on each side (input 240 / 320 / 400 bp), shift +-10 over real bases. Five of the 20 configurations are hard-wired to 200 positions
                   (indices 0, 10, 12, 13, 15) and cannot take wider input, so the context arms hold 15 configurations; compare them with hp34_base_30k on those 15.
  lg35_<arm>_30k   a LegNet-style reference (the proposer's reference tower, 670k parameters) under three recipes (menu recipe, AdamW lr 1e-3 wd 2e-4, AdamW lr 1e-2 wd 0.1) x 3 seeds, with the same
                   switches: base | emabn | flank (AL_FLANK_SHIFT=1) | emabn_flank. A clean ablation on one architecture.

Cells are appended to the registry under flock (backup dispatch_registry.json.bak_pk35).

  python scripts/program_oct22/register_pk35.py [--dry]
"""

import ast
import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_oct30 as r30  # noqa: E402

R, NEST, rel, V1 = r30.R, r30.NEST, r30.rel, r30.V1
DRY = "--dry" in sys.argv
OUT = "outputs/menu_retrain_v0"
V0C = "/grid/koo/home/shared/albench/data/val_V0c.npz"
LG_MENU = f"{R}/outputs/menus_oct29/legnet_ref.json"
D = 30000
EMA = {"AL_EMA": "0.9995", "AL_BN_RECALC": "1"}
PK = {
    "ema": ({"AL_EMA": "0.9995"}, ""),
    "bn": ({"AL_BN_RECALC": "1"}, ""),
    "emabn": (EMA, ""),
    "ema999bn": ({"AL_EMA": "0.999", "AL_BN_RECALC": "1"}, ""),
}
FX = {  # arm: (extra args, per-model cap in H100-minutes)
    "c0s30": (" --flank_ctx 0 --shift_max 30", 25),
    "c20s10": (" --flank_ctx 20 --shift_max 10", 25),
    "c60s10": (" --flank_ctx 60 --shift_max 10", 35),
    "c100s10": (" --flank_ctx 100 --shift_max 10", 45),
}
LG = {
    "base": ({}, ""),
    "emabn": (EMA, ""),
    "flank": ({"AL_FLANK_SHIFT": "1"}, ""),
    "emabn_flank": ({**EMA, "AL_FLANK_SHIFT": "1"}, ""),
}


def build_legnet_menu():
    """Three recipes of the proposer's LegNet-style reference tower (REFERENCE_SRC), as a menu file."""
    src = open(f"{R}/albench/arch_search/arch_autoresearch.py").read()
    ref = next(
        ast.literal_eval(n.value)
        for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "REFERENCE_SRC"
    )
    recipes = {
        "menu recipe (Muon lr 2e-3 wd 1e-4)": dict(optimizer="muon", lr=2e-3, weight_decay=1e-4),
        "AdamW lr 1e-3 wd 2e-4": dict(optimizer="adamw", lr=1e-3, weight_decay=2e-4),
        "AdamW lr 1e-2 wd 0.1 (LegNet recipe)": dict(optimizer="adamw", lr=1e-2, weight_decay=0.1),
    }
    menu = [
        {
            "id": f"legnet_ref_{i}",
            "cell": "reference",
            "idea": f"LegNet-style reference tower, {name}",
            "orig_val": None,
            "source": ref,
            "train_config": {**tc, "batch_size": 256, "epochs": 80, "lr_schedule": "onecycle"},
        }
        for i, (name, tc) in enumerate(recipes.items())
    ]
    return menu


def cells(name, menu, env_extra=None, extra="", shards=4, cap=25, seeds=1):
    out_dir = f"{OUT}/{name}"
    pool = f"{NEST}/genomic_d{D}.npz"
    env = {"AL_PATIENCE": "16", "MENU_MAX_FWD_MS": "1000000000", **(env_extra or {})}
    return [
        dict(
            name=f"{name}_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=out_dir,
            env=dict(env),
            needs=[rel(pool), rel(menu)],
            cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {menu} --reservoir_cache {pool} --D {D} --out_dir {R}/{out_dir} --shard {k}/{shards} --base_cap_min {cap} --val_cache {V0C}{extra}"
            + (f" --seeds {seeds}" if seeds != 1 else ""),
            done=dict(type="file", path=f"{out_dir}/COMPLETE_{k}of{shards}"),
        )
        for k in range(shards)
    ]


def all_cells():
    out = []
    for arm, (env, extra) in PK.items():
        out += cells(f"pk35_{arm}_30k", V1, env, extra)
    for arm, (extra, cap) in FX.items():
        out += cells(f"fx35_{arm}_30k", V1, None, extra, cap=cap)
    for arm, (env, extra) in LG.items():
        out += cells(f"lg35_{arm}_30k", LG_MENU, env, extra, shards=3, seeds=3)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    os.makedirs(os.path.dirname(LG_MENU), exist_ok=True)
    if not os.path.exists(LG_MENU) and not DRY:
        json.dump(build_legnet_menu(), open(LG_MENU, "w"), indent=1)
    elif DRY:
        print("legnet menu entries:", len(build_legnet_menu()))
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells | registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_pk35")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
