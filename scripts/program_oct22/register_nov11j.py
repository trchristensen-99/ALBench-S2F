#!/usr/bin/env python3
"""A second seed of the 300k reference bracket, so the top rung has a range.

The 300k bracket is registered at --seeds 1, which makes every non-LegNet point on the
scaling figure a single model at the rung the slope claim rests on -- and the slope is
exactly what Peter asked about ("DREAM-RNN might scale a bit better but that's not
certain"). An endpoint with no spread cannot answer that.

Only ONE extra seed offset is added, not two. These are h100-only, ~5 GPU-h each, and the
first six have not run yet; a second full set would compete with them for the scarcest
resource before we know what they cost. n=2 is enough for a range, and the error-decay
exponent separates replicates even at n=2. A third goes in during monitoring if the first
two land quickly.

6 cells, 6 models, ~30 GPU-h.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sys

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = "scratchpad/dispatch_registry.json"
DRY = "--dry" in sys.argv
OFFSET, NSHARD = 1, 6


def main() -> None:
    os.chdir(R)
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(REG))
        have = {c["name"] for c in reg}
        src = {c["name"]: c for c in reg}

        new = []
        for k in range(NSHARD):
            base = src.get(f"refbank300k_300k_genomic_can_e{k}_d0")
            assert base is not None, f"shard {k} of the 300k bracket is not registered"
            cmd = base["cmd"]
            assert "--seed_offset 0" in cmd, f"shard {k} is not at seed_offset 0"
            assert f"--shard {k}/{NSHARD}" in cmd, f"shard {k} numbering moved"
            cmd = cmd.replace("--seed_offset 0", f"--seed_offset {OFFSET}")
            assert "--seeds 1 " in cmd, "the bracket is no longer one seed per cell"

            name = f"refbank300k_300k_genomic_can_e{k}_d0_o{OFFSET}"
            if name in have:
                continue
            # menu_retrain names models r{seed+offset:02d}_menu_{idx:03d}, so the new seed
            # writes beside the old one in the same out_dir without touching it
            done = f"{base['out_dir']}/r{OFFSET:02d}_menu_{k:03d}_meta.json"
            assert not os.path.exists(os.path.join(R, done)), f"{done} already exists"
            new.append(dict(
                name=name, gpus=base["gpus"], ext=base.get("ext", []),
                env=dict(base.get("env", {})), out_dir=base["out_dir"],
                needs=list(base.get("needs", [])), cmd=cmd,
                done=dict(type="file", path=done),
            ))

        print(f"{len(new)} new cells (~{len(new) * 5} GPU-h, h100-only)")
        for c in new:
            print("  ", c["name"], "->", c["done"]["path"])
        if DRY or not new:
            return
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2), "duplicate cell name"
        shutil.copy(REG, REG + ".bak_nov11j")
        json.dump(reg2, open(REG + ".tmp", "w"), indent=1)
        os.replace(REG + ".tmp", REG)
        print(f"registered | registry {len(reg)} -> {len(reg2)}")


if __name__ == "__main__":
    main()
