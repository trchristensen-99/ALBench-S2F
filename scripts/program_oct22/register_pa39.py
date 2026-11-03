"""Nov-2 night: which part of the harness costs the published LegNet 0.02 to 0.04 genomic r?

pa38_ref (done): the published MPRA-LegNet with the bank's recipes through the menu-library harness (patience off, fp16) scores 0.034 below the native bank at 30k (tuned recipe 0.8416 against 0.8748; the
gap is 0.022 at 10k and 0.039 at 100k). The remaining protocol differences (audit F17): the harness shifts the whole batch once, the bank every sample; the harness trains on the average of the forward and reverse-complement
losses (two forward passes), the bank flips half of the samples (one pass); the harness keeps the epoch with the best RC-averaged validation MSE, the bank the best forward-only Pearson.

  pa39_rc_30k    AL_RC_MODE=aug: oracle-style single-pass reverse-complement augmentation (half of the samples flipped)
  pa39_sh_30k    --shift_per_sample: one shift per sample
  pa39_both_30k  both: the closest the harness gets to the native protocol

30k, the tuned recipe only (e500, wd 0.2, AdamW lr 0.01, batch 1024, one-cycle), 3 seeds each, patience off, fp16; compared with pa38_ref_30k (neither change) and the native bank bench38_ln_e500_wd2_30k_c20.

    python scripts/program_oct22/register_pa39.py [--dry]
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
ARMS = {
    "rc": ({"AL_RC_MODE": "aug"}, ""),
    "sh": ({}, " --shift_per_sample"),
    "both": ({"AL_RC_MODE": "aug"}, " --shift_per_sample"),
}


def tuned_menu() -> str:
    src = f"{R}/outputs/menus_nov2/PARITY_30k.json"
    path = "outputs/menus_nov2/PARITY_30k_tuned.json"
    if not DRY:
        entries = [e for e in json.load(open(src)) if "tuned" in e["id"]]
        assert len(entries) == 1
        json.dump(entries, open(f"{R}/{path}", "w"), indent=1)
    return f"{R}/{path}"


def all_cells():
    menu, out = tuned_menu(), []
    for arm, (env, extra) in ARMS.items():
        out += fb.cells(
            f"pa39_{arm}_30k",
            menu,
            f"{NEST}/genomic_d{D}.npz",
            D,
            {"AL_PATIENCE": "9999", "AL_PRECISION": "fp16", **env},
            nb.STD + extra,
            shards=1,
            cap=120,
            seeds=3,
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
        if DRY:
            print(new[0]["cmd"][:420] if new else "nothing new")
            return
        shutil.copy(path, path + ".bak_pa39")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
