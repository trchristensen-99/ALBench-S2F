"""Nov-9: give the from-scratch curve the error bar the acquisition curves already have.

The three-curve comparison (from scratch / from a 30k pilot / from a 300k pilot) turns on gaps of a few
thousandths, and right now the two sides are not measured to the same precision. al52 and al54 each run
two independent replicates -- a different candidate draw and a different pilot per replicate -- so the
acquisition side reports a spread. The from-scratch side is one draw (canonical, seed 42) with five
seeds stacked on it, so it reports seed noise only. Comparing a number with a band against a number
without one is how a corpus gets recommended on a difference that was never outside the noise.

So: the same from-scratch rungs on two independent draws. `k562_reps` holds them, and two properties
were checked before registering rather than assumed (scratchpad/check_reps_nov9.py) -- each replicate is
nested (its d100000 is an exact prefix of its d300000, so a 55k and a 105k cut come from one consistent
ordering), and a replicate shares *zero* sequences with the canonical pool, so this measures draw noise
rather than seed noise a second time.

30k is included because it is where all three curves start: the question "did selecting 75k more beat
simply drawing 75k more?" is anchored on that point, and it is cheap.

Five rungs x two replicates x five seeds, every cell V100-eligible -- which is the point, since the
H100s are saturated and roughly two dozen V100s have been idle. About 45 GPU-hours there.

300k earns its place twice: it is a rung of the from-scratch ladder and it is also where al53's arms
start, so it is the one size at which the 300k-start comparison can be given a from-scratch band from
pools that already exist. Above it the replicate pools stop, so al53's 400k/500k/600k rungs still have
no band -- that needs new draws built and oracle-labelled, which is a separate decision.

    python scripts/program_oct22/register_nov9a.py [--dry]
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
REPS_DIR = "outputs/grid_pools/k562_reps"
REPS = (1, 2)
# acq_ladder.recipe_for() exactly, so a from-scratch rung and an acquisition rung of the same size are
# trained identically and the curves differ only in how the corpus was chosen
RUNGS = {
    30_000: "e500_wd2",
    55_000: "e500_wd15",
    80_000: "e500",
    105_000: "e500",
    300_000: "e300_wd05",
}
SLICE = (55_000, 80_000, 105_000)  # 30k and 300k already exist as replicate pools
SEEDS = (0, 1, 2, 3, 4)


def seed_for(rep, D, s):
    """Deterministic, and distinct from the canonical curve's draws at the same sizes."""
    return random.Random(f"sc56|{RES}|r{rep}|{D}|{s}").randrange(10_000_000, 99_999_999)


def mark(rep):
    return f"{REPS_DIR}/_sliced_{RES}_r{rep}_curve"


def all_cells():
    sizes = ",".join(str(d) for d in SLICE)
    out = []
    for rep in REPS:
        out.append(
            dict(
                name=f"pool56_{RES}_r{rep}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=mark(rep),
                env={},
                needs=[f"{REPS_DIR}/{RES}_r{rep}_d300000.npz"],
                cmd=(
                    f"uv run --no-sync python scripts/slice_nested_prefix.py "
                    f"--dir {REPS_DIR} --strategy {RES}_r{rep} --sizes {sizes} "
                    f"&& mkdir -p {R}/{mark(rep)} && touch {R}/{mark(rep)}/COMPLETE"
                ),
                done=dict(type="file", path=f"{mark(rep)}/COMPLETE"),
            )
        )
        for D, cfg in RUNGS.items():
            pool = f"{REPS_DIR}/{RES}_r{rep}_d{D}.npz"
            # bench33_legnet takes the reservoir name from the pool stem, so the `_r{rep}` keeps these
            # models out of the canonical pool's directory. Verified at line 168 of that script.
            odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{RES}_r{rep}"
            needs = [] if D not in SLICE else [f"{mark(rep)}/COMPLETE"]
            for s in SEEDS:
                out.append(
                    dict(
                        name=f"sc56_{RES}_r{rep}_{D // 1000}k_s{s}",
                        gpus=["h100", "v100"],
                        ext=[],
                        out_dir=odir,
                        env={},
                        needs=needs + ([pool] if not needs else []),
                        cmd=(
                            f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                            f"--D {D} --cfg {cfg} --pool {R}/{pool} "
                            f"--seeds {s}={seed_for(rep, D, s)} --ctx {F} "
                            f"&& touch {R}/{odir}/COMPLETE_s{s}"
                        ),
                        done=dict(type="file", path=f"{odir}/COMPLETE_s{s}"),
                    )
                )
    return out


def main():
    cells = all_cells()
    # a replicate that quietly reused the canonical pool would measure nothing
    for c in cells:
        assert "k562_nested" not in c["cmd"], f"{c['name']} reaches into the canonical pools"
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
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
        shutil.copy(path, path + ".bak_nov9a")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
