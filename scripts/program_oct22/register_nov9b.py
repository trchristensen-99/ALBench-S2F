"""Nov-9: rerun al54's arms under their own tags. The first run measured al52 and called it al54.

`acq_ladder.py` names its training output directory from the arm tag (line 268:
`bench38_ln_{cfg}_{d//1000}k_c{ctx}_{res}`, where `res` is the tag), and `bench33_legnet.py` skips any
model whose `_meta.json` already exists (line 242). al54 was registered with al52's tags -- `os_random_s1`,
`os_epi_s1`, `it3_epi_s1` -- at al52's sizes, so every arm resolved to a directory al52 had already
filled, skipped all five seeds, and touched its `.done`. The arms "COMPLETED" in 11 s to 4 min against
al52's hours, and every `_meta.json` in those directories is still stamped Nov-8 15:35-19:59.

That is the worst possible failure for this particular cell: al54 exists to separate "epistemic is a good
selector" from "al52's genomic pilot was a bad start", and a void al54 would have reproduced al52's
numbers exactly, which reads as "the pilot made no difference". A confirmed null, produced by a skip.

The pilots are genuine and are kept: they ran 2 h 38 m each and round 0 takes its name from the seed
pool (`motif_shared_core_s{rep}`), not from the arm tag, so they never collided. Rerunning with a new arm
tag re-enters round 0, finds those pilot models, and skips straight past them -- which is the intended
reuse, the same pilot for every arm.

~21 GPU-hours. The guard below is the point of this file as much as the rerun is.

    python scripts/program_oct22/register_nov9b.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402
import register_nov8i as i  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
PFX = "a54"  # what the first run lacked: a tag that names the experiment, not just the arm
# (arm -> acquisition, schedule, and the (cfg, D) directories that arm will write)
ARMS = {
    "os_random": ("random", "75000", (("e500", 105_000),)),
    "os_epi": ("epistemic", "75000", (("e500", 105_000),)),
    "it3_epi": (
        "epistemic",
        "25000,25000,25000",
        (("e500_wd15", 55_000), ("e500", 80_000), ("e500", 105_000)),
    ),
}


def all_cells():
    out = []
    for arm, (acq, sched, _dirs) in ARMS.items():
        for rep in i.REPS:
            tag = f"{PFX}_{arm}_s{rep}"
            needs = [f"{i.PD}/cand_r{rep}.npz", f"{i.DONE}/pilot_s{rep}.done"]
            c = i.ladder(tag, acq, rep, needs)
            c["name"] = f"a1m_acq54b_{tag}"  # still matches walltime()'s `a1m_acq5` rule
            if sched != "75000":
                c["cmd"] = c["cmd"].replace("--schedule 75000", f"--schedule {sched}")
            assert f"--schedule {sched} " in c["cmd"], "the schedule did not get swapped"
            out.append(c)
    return out


def main():
    cells = all_cells()
    # The guard. An arm whose output directory already holds models does not train -- it skips and
    # reports whatever is in there. Refuse to register such an arm rather than discover it afterwards.
    for arm, (_acq, _s, dirs) in ARMS.items():
        for rep in i.REPS:
            for cfg, D in dirs:
                d = Path(R) / OUT / f"bench38_ln_{cfg}_{D // 1000}k_c{F}_{PFX}_{arm}_s{rep}"
                hit = sorted(d.glob("*_meta.json")) if d.exists() else []
                assert not hit, (
                    f"{d.name} already holds {len(hit)} models; this arm would skip training"
                )
    print(f"guard: {sum(len(v[2]) for v in ARMS.values()) * len(i.REPS)} output dirs are clear")

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
        shutil.copy(path, path + ".bak_nov9b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
