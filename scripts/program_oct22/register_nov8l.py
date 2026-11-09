"""Nov-8: sample the from-scratch curve at the rungs the acquisition curves actually land on.

Three curves are now in flight for motif_shared_core: from scratch (the sc52 ladder), from a 30k pilot
(al54) and from a 300k pilot (al53). They are only comparable where they are measured at the same
sizes, and today they are not -- the from-scratch ladder has 10k / 30k / 100k / 300k / 1M while the
acquisition arms train at 30k, 55k, 80k, 105k and at 300k, 400k, 500k, 600k. Reading a from-scratch
number at 105k off a curve drawn through 100k and 300k is interpolation dressed as measurement, and the
whole point of the comparison is a gap of a few thousandths.

Since the pools are nested, the missing rungs are cuts rather than new draws: a 600k prefix of the 1M
pool is 600,000 sequences of the same strategy from the same draw. So this costs training time only.

Six rungs x five seeds, matching the acquisition arms' five-seed stacks so the two sides stack the same
way. The small three are V100-eligible; 400k and up stay on H100. About 29 GPU-hours.

    python scripts/program_oct22/register_nov8l.py [--dry]
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
RES = "motif_shared_core"
NEST = "outputs/grid_pools/k562_nested"
# the per-size recipe exactly as acq_ladder.recipe_for() applies it, so a from-scratch rung and an
# acquisition rung of the same size are trained the same way and the curves differ only in the corpus
RUNGS = {
    55_000: ("e500_wd15", ["h100", "v100"]),
    80_000: ("e500", ["h100", "v100"]),
    105_000: ("e500", ["h100", "v100"]),
    400_000: ("e300_wd05", ["h100"]),
    500_000: ("e300_wd05", ["h100"]),
    600_000: ("e300_wd05", ["h100"]),
}
SEEDS = (0, 1, 2, 3, 4)
MARK = f"{NEST}/_sliced_{RES}_curve"


def seed_for(D, s):
    """Deterministic, and distinct from the sc52 ladder's draws at the sizes it already covers."""
    return random.Random(f"sc55|{RES}|{D}|{s}").randrange(10_000_000, 99_999_999)


def all_cells():
    sizes = ",".join(str(d) for d in RUNGS)
    out = [
        dict(
            name=f"pool55_{RES}_curve",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=MARK,
            env={},
            needs=[f"{NEST}/{RES}_d1000000.npz"],
            cmd=(
                f"uv run --no-sync python scripts/slice_nested_prefix.py "
                f"--strategy {RES} --sizes {sizes} "
                f"&& mkdir -p {R}/{MARK} && touch {R}/{MARK}/COMPLETE"
            ),
            done=dict(type="file", path=f"{MARK}/COMPLETE"),
        )
    ]
    for D, (cfg, gpus) in RUNGS.items():
        pool = f"{NEST}/{RES}_d{D}.npz"
        odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{RES}"
        for s in SEEDS:
            out.append(
                dict(
                    name=f"sc55_{RES}_{D // 1000}k_s{s}",
                    gpus=gpus,
                    ext=[],
                    out_dir=odir,
                    env={},
                    needs=[f"{MARK}/COMPLETE"],
                    cmd=(
                        f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                        f"--D {D} --cfg {cfg} --pool {R}/{pool} "
                        f"--seeds {s}={seed_for(D, s)} --ctx {F} "
                        f"&& touch {R}/{odir}/COMPLETE_s{s}"
                    ),
                    done=dict(type="file", path=f"{odir}/COMPLETE_s{s}"),
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
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        for c in new:
            print(" ", c["name"])
        if DRY:
            print()
            print(new[0]["cmd"])
            print(new[-1]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov8l")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
