"""Nov-3 evening: let the loop choose any architecture AND any recipe, with the epoch budget fixed at 300; and give the collaborator's architectures the long schedules he trains them with.

1. `AL_FIXED_EPOCHS=300` (new in agent_loop): the proposer writes the full TRAIN line (optimizer, lr, weight decay, batch size, schedule) and the harness sets the epochs to 300 and never stops a one-cycle schedule early. The
   enforced AdamW lr 0.01 / wd 0.2 / batch 1024 of ar41/ln41 removed the recipe as a confound, but it also forces one recipe on every design (it diverged for one dilated tower in the bake-off, and the 5 LegNet-family designs
   prefer it); this arm lets each design pick its own.

  af42_30k_r<k>   schedule_ens manager (explore 40% then exploit), free recipe, 300 epochs      (vs ar41: fixed recipe)
  ln42_30k_r<k>   lanes manager (one LegNet-trunk and one independent-trunk model per round), free recipe, 300 epochs     (vs ln41)

2. `mrr_pi41_<recipe>_30k`: the five architectures re-implemented from the collaborator's descriptions (C219 ShuffleNet-style, C213 ResNeXt, C209 Volterra, C210 plain + 2D pairwise map, C218 Volterra + Fourier;
   `build_menu_pi.py`) trained with 300 epochs under harness H3 (single-pass RC augmentation, patience off, fp16), 2 seeds each. pi38 trained them for 60-80 epochs under the old harness, while his models train 160-360 epochs
   (his tables, Nov 2): so the earlier "0.843 against his 0.852-0.855" may be under-training. Recipes: tuned (AdamW 0.01, wd 0.2, b1024), mid (AdamW 0.003, wd 0.1, b512), ownmuon (Muon 2e-3, wd 1e-4, b256) and
   ownadamw (AdamW 0.003, wd 0.05, b256) at 300 epochs. Named `mrr_` so the dispatcher keeps them off MIG slices (11M-parameter models).

    python scripts/program_oct22/register_nov3f.py [--dry]
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
H3 = {"AL_RC_MODE": "aug"}
D = 30000
RECIPES = {
    "tuned": {"optimizer": "adamw", "lr": 0.01, "weight_decay": 0.2, "batch_size": 1024, "epochs": 300, "lr_schedule": "onecycle"},
    "mid": {"optimizer": "adamw", "lr": 0.003, "weight_decay": 0.1, "batch_size": 512, "epochs": 300, "lr_schedule": "onecycle"},
    "ownmuon": {"optimizer": "muon", "lr": 0.002, "weight_decay": 0.0001, "batch_size": 256, "epochs": 300, "lr_schedule": "onecycle"},
    "ownadamw": {"optimizer": "adamw", "lr": 0.003, "weight_decay": 0.05, "batch_size": 256, "epochs": 300, "lr_schedule": "onecycle"},
}  # fmt: skip


def llm_cells():
    reps, cap, budget = nb.RR_SIZE
    out = []
    for r in range(1, reps + 1):
        for tag, mgr in (("af42", "schedule_ens"), ("ln42", "lanes")):
            c = nb.loop_cell(
                f"{tag}_30k_r{r}", f"genomic_30k_{tag}", r, D, 15, cap, budget,
                {**nb.RR, "manager": mgr, "explore_frac": 0.4},
            )  # fmt: skip
            c["env"] = {**c["env"], **H3, "AL_FIXED_EPOCHS": "300"}
            out.append(c)
    return out


def pi_cells():
    base = [
        e for e in json.load(open(f"{R}/outputs/menus_oct31/PI.json")) if e["id"].endswith("_adamw")
    ]
    assert len(base) == 5, len(base)
    out = []
    for name, recipe in RECIPES.items():
        ents = [
            {
                **{k: v for k, v in e.items() if k != "train_config"},
                "id": e["id"].replace("_adamw", "") + f"_{name}",
                "train_config": dict(recipe),
            }
            for e in base
        ]
        path = f"{R}/outputs/menus_nov3/PI41_{name}_30k.json"
        if not DRY:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(ents, open(path, "w"), indent=1)
        out += fb.cells(
            f"mrr_pi41_{name}_30k", path, f"{NEST}/genomic_d{D}.npz", D,
            {"AL_PATIENCE": "9999", "AL_PRECISION": "fp16", **H3}, nb.STD, shards=5, cap=120, seeds=2,
        )  # fmt: skip
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in llm_cells() + pi_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells | registry", len(reg), "->", len(reg2))
        if DRY:
            print([c["name"] for c in new][:8], "...")
            return
        shutil.copy(path, path + ".bak_nov3i")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
