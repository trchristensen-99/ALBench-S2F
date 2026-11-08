"""Nov-8: the ms:mc ratio at the two small rungs, and a second draw for the activity-chasing bound.

mx53 brackets the 50/50 ms:mc ratio at 100k and 300k, the sizes mx51 reported. But the ratio is exactly
the kind of thing that can move with size: at 10k you can only afford a few thousand of anything, so the
share you give each reservoir matters more than it does at 300k where both sources are past their own
knees. The bracket pools at 10k and 30k were already written by the same build (every size is a prefix of
the 300k file), and the 50/50 anchor and the single-reservoir parents are already measured at both rungs,
so the comparison is complete the moment these finish. Small cells, so they can take the idle V100s.

The second piece is one cell. al52's `oracle_top` arm -- select the 75,000 candidates with the highest
hidden oracle label -- is a bound rather than a method, and it came back with the most surprising number
in the experiment: it loses to plain epistemic uncertainty on all three axes, and collapses on the random
held-out set (0.796 against 0.952), having bought a corpus with a mean label of 3.69 against 0.61 for a
random draw. That is a result worth quoting -- an activity-chaser buys a corpus that cannot predict
ordinary sequences -- and it currently rests on a single replicate. The rep-2 candidate pool already
exists, so a second draw costs one training cell and takes the claim off n=1.

A third replicate of al52 is NOT registered: `evoaug_pape` has no r3 draw in k562_reps, so a rep-3
candidate pool would be mixed over six reservoirs where rep 1 and 2 used seven, and the composition
readout -- the part of al52 worth the most -- would not be comparable across replicates.

    python scripts/program_oct22/register_nov8g.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402
import register_nov8b as b  # noqa: E402
import register_nov8f as f  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
PD = "outputs/grid_pools/k562_mixtures"
RUNGS = {10_000: "e800_wd4", 30_000: "e500_wd2"}  # the per-size recipe, as mx51 used at these rungs


def all_cells():
    out = []
    for mix in f.MIXES:
        for D, cfg in RUNGS.items():
            for rep in f.REPS:
                pool = f"{PD}/{mix}_r{rep}_d{D}.npz"
                odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{mix}_r{rep}"
                for s in f.SEEDS:
                    out.append(
                        dict(
                            # 10k and 30k fit a V100 comfortably, and an idle V100 is the whole
                            # reason these are worth registering now rather than after mx53 reports
                            name=f"mx53_{mix}_{D // 1000}k_r{rep}_s{s}",
                            gpus=["h100", "v100"],
                            ext=[],
                            out_dir=odir,
                            env={},
                            needs=[pool],
                            cmd=(
                                f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                                f"--D {D} --cfg {cfg} --pool {R}/{pool} "
                                f"--seeds {s}={f.seed_for(mix, D, rep, s)} --ctx {F} "
                                f"&& touch {R}/{odir}/COMPLETE_s{s}"
                            ),
                            done=dict(type="file", path=f"{odir}/COMPLETE_s{s}"),
                        )
                    )
    acq, schedule, _ = b.ARMS["os_oracle"]
    out.append(
        b.ladder(
            "os_oracle_s2",
            acq,
            schedule,
            2,
            [f"{b.PD}/cand_r2.npz", f"{b.DONE}/pilot_s2.done"],
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
        shutil.copy(path, path + ".bak_nov8g")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
