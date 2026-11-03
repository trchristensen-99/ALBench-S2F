"""Nov-3: are the architectures that are NOT LegNet descendants weaker, or just trained with the wrong recipe?

The funnel analysis of the 21 standard 30k loop runs (`extract_diversity_funnel.py`) found that designs which do not name the LegNet trunk (22% of trained models) end 0.054 validation r below the best LegNet-named
design, rank 26th of ~30, and carry about 1% of the stack weight. But they were all trained with short recipes (median 58 epochs, batch 128-256, lr about 0.002), while the LegNet descendants got long AdamW schedules
(median 150 epochs, batch 512-1024): the loop never gave a different architecture the treatment that made LegNet strong. No LLM is needed to separate the two explanations: retrain the best of them under matched recipes.

  bake_own_30k    every entry with the recipe the loop gave it, full schedule (patience off)        the control: same models, harness H3
  bake_tuned_30k  AdamW lr 0.01, wd 0.2, batch 1024, 300 epochs, one-cycle (the 30k LegNet recipe, e300)
  bake_mid_30k    AdamW lr 0.003, wd 0.1, batch 512, 300 epochs, one-cycle (gentler: attention and recurrent layers dislike lr 0.01)

Entries (validation-side selection only, runs rr40/rr39/ef39_*, at most 2 per run and 1 per idea): the 8 best designs that do not name LegNet/EffBlock, the 4 best LegNet hybrids (a BiGRU, attention, U-Net or
state-space part on the trunk) and the 2 best plain LegNet variants as controls. Harness H3 (patience off, fp16, single-pass RC augmentation), 2 seeds, V100 only (the H100s are saturated by other groups).
Compare with `pa39_rc_30k` (the published LegNet, tuned recipe, same harness: val 0.9037, genomic 0.8736) through `compare_legnet_vs_loop.py`-style stacks.

    python scripts/program_oct22/register_nov3c.py [--dry]
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
D = 30000
AR = "outputs/archsearch_reps_oct22"
FUNNEL = f"{R}/outputs/justification_oct22/data_diversity_funnel_30k.json"
ARMS = (
    "rr40",
    "rr39",
    "ef39_20",
    "ef39_60",
    "ef39_100",
)  # V0c-validated runs only (vc39 used another validation set)
RECIPES = {
    "own": None,
    "tuned": {
        "optimizer": "adamw",
        "lr": 0.01,
        "weight_decay": 0.2,
        "batch_size": 1024,
        "epochs": 300,
        "lr_schedule": "onecycle",
    },
    "mid": {
        "optimizer": "adamw",
        "lr": 0.003,
        "weight_decay": 0.1,
        "batch_size": 512,
        "epochs": 300,
        "lr_schedule": "onecycle",
    },
}
SHARDS = 7


def hybrid(m: dict) -> bool:
    return m["text_legnet"] and any(c not in ("plain conv", "dilated") for c in m["classes"])


def pick(pool: list, cond, n: int, per_run: int = 2) -> list:
    out, cnt, seen = [], {}, set()
    for arm, rep, m in sorted(pool, key=lambda x: -x[2]["val"]):
        key = (arm, rep)
        if not cond(m) or m["idea"][:60] in seen or cnt.get(key, 0) >= per_run:
            continue
        out.append((arm, rep, m))
        cnt[key] = cnt.get(key, 0) + 1
        seen.add(m["idea"][:60])
        if len(out) == n:
            break
    return out


def entries() -> dict:
    """One menu per recipe: the same 14 architectures, only `train_config` differs."""
    d = json.load(open(FUNNEL))
    pool = [(a, r, m) for a in ARMS for r, v in d[a].items() for m in v["models"]]
    chosen = (
        [("independent",) + x for x in pick(pool, lambda m: not m["text_legnet"], 8)]
        + [("hybrid",) + x for x in pick(pool, hybrid, 4)]
        + [
            ("plain LegNet",) + x
            for x in pick(pool, lambda m: m["text_legnet"] and not hybrid(m), 2)
        ]
    )
    assert len(chosen) == 14, len(chosen)
    base = []
    for kind, arm, rep, m in chosen:
        meta = json.load(open(f"{R}/{AR}/genomic_30k_{arm}/{rep}/{m['id']}_meta.json"))
        tc = meta["train_config"]
        base.append(
            {
                "id": f"{arm}_{rep}_{m['id']}",
                "cell": f"{arm}/{rep}",
                "kind": kind,
                "idea": meta["idea"],
                "orig_val": float(meta["val_pearson"]),
                "n_params": int(meta["n_params"]),
                "source": meta["source"],
                "own_train_config": tc,
            }
        )
    out = {}
    for name, recipe in RECIPES.items():
        out[name] = [
            {
                **{k: v for k, v in e.items() if k != "own_train_config"},
                "train_config": dict(recipe or e["own_train_config"]),
            }
            for e in base
        ]
    return out


def all_cells() -> list:
    menus, out = entries(), []
    for name, ents in menus.items():
        path = f"{R}/outputs/menus_nov3/BAKE_{name}_30k.json"
        if not DRY:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(ents, open(path, "w"), indent=1)
        cells = fb.cells(
            f"bake_{name}_30k",
            path,
            f"{NEST}/genomic_d{D}.npz",
            D,
            {"AL_PATIENCE": "9999", "AL_PRECISION": "fp16", "AL_RC_MODE": "aug"},
            nb.STD,
            shards=SHARDS,
            cap=90,
            seeds=2,
        )
        for c in cells:
            c["gpus"] = [
                "v100"
            ]  # H100 queues are full of other groups' jobs; these are cheap on V100s
        out += cells
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells at the end | registry", len(reg), "->", len(reg2))
        if DRY:
            print(new[0]["cmd"][:520] if new else "nothing new")
            return
        shutil.copy(path, path + ".bak_nov3d")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
