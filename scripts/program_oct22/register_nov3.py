"""Nov-3: the fixed loop (H3 = patience fix + fp16 + non-finite skip + single-pass RC augmentation + the isolated, medium-effort CLI transport) at the other sizes and with more rounds.

At 30k the H3 arms (rr40, lv40) reached the bank: best single models 0.871 to 0.878 in every run, loop stack 0.886 against the bank's 0.891, loop + bank 0.8975 above the bank alone (docs/LEGNET_VS_LOOP_AND_NEXT_ARMS.md).

  rr40_10k_r<k>, lv40_10k_r<k>     the standard managed loop (coordinator, 40% explore, reference LegNet and bank bar; 15 rounds x 2) and the LegNet-variants loop (8 rounds x 2) at 10k, 3 replicate pools each
  rr40_100k_r<k>, lv40_100k_r<k>   the same at 100k, 2 replicate pools each; 10 rounds x 2 and 6 rounds x 2, per-model cap 100 H100-minutes, H100 only (the tuned 100k recipe alone is about 25 H100-minutes)
  rl40_30k_r<k>                    the standard loop at 30k with twice the rounds (30 x 2 = 60 models), 2 replicate pools: does the stack keep improving with more models now that the models are good?

Compared with the native banks (bench38_ln_*) and, at 30k, with rr40 (same pools).

    python scripts/program_oct22/register_nov3.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov2b as nb  # noqa: E402

DRY = nb.DRY
H3 = {"AL_RC_MODE": "aug"}
# size -> (replicates, rr rounds, lv rounds, per-model cap in H100-minutes, rr budget h, lv budget h)
SIZES = {10000: (3, 15, 8, 30, 14.0, 12.0), 100000: (2, 10, 6, 100, 40.0, 36.0)}


def cell(name, dirname, r, D, rounds, cap, budget, cfg):
    c = nb.loop_cell(name, dirname, r, D, rounds, cap, budget, cfg)
    c["env"] = {**c["env"], **H3}
    return c


def all_cells():
    out = []
    for D, (reps, rr_rounds, lv_rounds, cap, rr_budget, lv_budget) in SIZES.items():
        k = D // 1000
        for r in range(1, reps + 1):
            out.append(
                cell(
                    f"rr40_{k}k_r{r}",
                    f"genomic_{k}k_rr40",
                    r,
                    D,
                    rr_rounds,
                    cap,
                    rr_budget,
                    nb.RR | {"explore_frac": 0.4},
                )
            )
            out.append(
                cell(
                    f"lv40_{k}k_r{r}",
                    f"genomic_{k}k_lv40",
                    r,
                    D,
                    lv_rounds,
                    cap,
                    lv_budget,
                    nb.COORD_VARIANTS,
                )
            )
    for r in (1, 2):
        out.append(
            cell(
                f"rl40_30k_r{r}",
                "genomic_30k_rl40",
                r,
                30000,
                30,
                40,
                30.0,
                nb.RR | {"explore_frac": 0.4},
            )
        )
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
        for c in new:
            print(
                "  ", c["name"], c["gpus"], "cap", c["cmd"].split("--base_cap_min ")[1].split()[0]
            )
        if DRY:
            return
        shutil.copy(path, path + ".bak_nov3")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
