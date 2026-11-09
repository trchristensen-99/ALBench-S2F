"""Nov-8: is the 1M rung still short of epochs?

The per-size recipe gives 1M its own entry, `e200_wd05`, and the bracket run beside it says that entry
may be on the wrong side of its own knee. At 1,000,000 sequences the same corpus trained for 200 epochs
beats the 100-epoch bracket on every axis -- genomic 0.9724 against 0.9695, designed 0.9168 against
0.9083, random 0.9780 against 0.9736 -- so the curve was still rising where the recipe stops. Every
other rung was tuned until it stopped rising; 1M never was, because until this week there was no 1M
rung to tune.

So: the same corpus and the same five seeds at `e300_wd05`, the recipe the 300k rung settled on. If it
beats e200 the per-size table needs a new 1M entry and every 1M number in the study is a floor rather
than a measurement; if it does not, e200 is the knee and the table is right.

One corpus only -- motif_shared_core, which wins the designed axis at every size including 1M -- because
the question is about the recipe, not the corpus, and a second corpus would answer it twice. About 22
GPU-hours and no oracle labelling.

    python scripts/program_oct22/register_nov8h.py [--dry]
"""

import fcntl
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
POOL = "outputs/grid_pools/k562_nested/motif_shared_core_d1000000.npz"
RES = "motif_shared_core"
CFG = "e300_wd05"
D = 1_000_000
SEEDS = (0, 1, 2, 3, 4)


def seed_for(s):
    """Distinct from the e200 draws, so the two recipes are compared over independent inits."""
    return random.Random(f"sc52h|{RES}|{D}|{CFG}|{s}").randrange(10_000_000, 99_999_999)


def all_cells():
    odir = f"{OUT}/bench38_ln_{CFG}_{D // 1000}k_c{F}_{RES}"
    return [
        dict(
            name=f"a1m_sc52_{RES}_{CFG}_s{s}",
            gpus=["h100"],
            ext=[],
            out_dir=odir,
            env={},
            needs=[POOL],
            cmd=(
                f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                f"--D {D} --cfg {CFG} --pool {R}/{POOL} "
                f"--seeds {s}={seed_for(s)} --ctx {F} "
                f"&& touch {R}/{odir}/COMPLETE_s{s}"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_s{s}"),
        )
        for s in SEEDS
    ]


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        for c in new:
            print(" ", c["name"])
        if DRY:
            print()
            print(new[0]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov8h")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
