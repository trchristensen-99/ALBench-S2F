"""Nov-4: the control for the 10k loop result. At 10k the H3 loop stack (0.8029 genomic) beats the 20-seed tuned LegNet bank (0.7801), but the bank's recipe (e800, wd 0.4) was still improving at 800 epochs and the loop's best
models all used another one: AdamW, peak lr 0.006-0.01, batch 512, weight decay 0.3, 400 epochs. This trains 20 seeds of the published MPRA-LegNet with that recipe (native trainer, same pool, input standard and V0c selection as the
bank), one seed per job, to see whether the gap is a recipe the bank was never given.

  bc10_s0..s19   outputs/menu_retrain_v0/bench38_ln_e400_b512_wd3_lr8_10k_c<F>

    python scripts/program_oct22/register_nov4a.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, NEST, F, DRY = ln.R, ln.OUT, ln.NEST, ln.F, ln.DRY
D, ARM = 10000, "e400_b512_wd3_lr8"
SEED_FILE = f"{R}/outputs/bench33/seeds_ctrl_10k_nov4.json"


def seeds() -> list:
    if os.path.exists(SEED_FILE):
        return json.load(open(SEED_FILE))
    s = [int.from_bytes(os.urandom(3), "big") for _ in range(20)]
    if not DRY:
        json.dump(s, open(SEED_FILE, "w"))
    return s


def all_cells():
    sd, out = seeds(), []
    odir = f"{OUT}/bench38_ln_{ARM}_{D // 1000}k_c{F}"
    for j in range(20):
        out.append(
            dict(
                name=f"bc10_s{j}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=odir,
                env={},
                needs=[f"{NEST}/genomic_d{D}.npz"],
                cmd=f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py --D {D} --cfg {ARM} --seeds {j}={sd[j]} --ctx {F} && touch {R}/{odir}/COMPLETE_s{j}",
                done=dict(type="file", path=f"{odir}/COMPLETE_s{j}"),
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
        if DRY:
            print(new[0]["cmd"][:330] if new else "nothing new")
            return
        shutil.copy(path, path + ".bak_nov4a")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
