"""Oct-29 (evening): augmentation closer to how the oracle was trained, on top of 20 real flank bases (menu v1, genomic 30k pool, V0c validation, 15 configurations that take wider input).

The oracle (train_oracle_s2_v2.py) shifts the 600-bp window by up to +-15 real bases on HALF of its samples, draws the shift per sample, and flips the strand on half of the samples. Our students shift +-10 on every batch (one
shift for the whole batch) and average forward and reverse-complement predictions in the loss. Arms:

  fx36_c20_ps10_30k      per-sample shift +-10 on every sample (isolates per-sample against per-batch)
  fx36_c20_oracle_30k    the oracle's shift: per sample, +-15, on half of the samples
  fx36_c20_tta8_30k      the current training (per-batch +-10) and, at evaluation, the mean over real-flank windows at offsets -8, -4, 0, 4, 8

Compared with fx35_c20s10_30k (same configurations, seeds and data).

    python scripts/program_oct22/register_fx36.py [--dry]
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
import register_oct30 as r30  # noqa: E402

R, NEST, V1 = r30.R, r30.NEST, r30.V1
DRY = "--dry" in sys.argv
ARMS = {
    "fx36_c20_ps10_30k": " --flank_ctx 20 --shift_max 10 --shift_per_sample",
    "fx36_c20_oracle_30k": " --flank_ctx 20 --shift_max 15 --shift_per_sample --shift_prob 0.5",
    "fx36_c20_tta8_30k": " --flank_ctx 20 --shift_max 10 --tta 8",
}


def all_cells():
    out = []
    for name, extra in ARMS.items():
        out += fb.cells(name, V1, f"{NEST}/genomic_d30000.npz", 30000, {}, extra)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        i = max(i for i, c in enumerate(reg) if c["name"].startswith("fx35_c200s10_30k"))
        reg2 = reg[: i + 1] + new + reg[i + 1 :]
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells after index", i, "| registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_fx36")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
