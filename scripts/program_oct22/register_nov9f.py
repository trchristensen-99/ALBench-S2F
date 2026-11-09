"""Nov-9: give the 1M rung a second draw. It is the last point on the from-scratch curve with none.

Every other rung from 10k to 600k now carries two to four independent draws. 1M carries one
(0.9178+-0.0000!), and it is the rung that the "the curve is still climbing at 1M" claim rests on --
the claim that says the corpus should be large before it is clever. A single draw there cannot tell a
climb from a lucky stack.

Same extension route as sc57, one rung higher: r1's 600k pool is kept byte-identical as a prefix and
only the appended 400k go to the oracle, so the 1M point continues the identical draw already measured
at 30k-600k rather than starting a new ordering at the top of the curve. The two build_pool_extend
defaults are re-pinned and asserted as before (seed 43 for r1, not the canonical 42; an explicit `_r1`
output name).

Five cells plus the extension, e200_wd05 per `acq_ladder.RECIPE`, V100-eligible -- which is the point
this hour, with 12 V100s idle and nothing queued. ~25 GPU-h on H100, up to ~55 if it all lands on V100.

    python scripts/program_oct22/register_nov9f.py [--dry]
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
RD = "outputs/grid_pools/k562_reps"
REP, GSEED = 1, 43  # slice_replicate_pools.SEED0[1]
D, CFG = 1_000_000, "e200_wd05"
BASE = f"{RD}/{RES}_r{REP}_d600000.npz"
TOP = f"{RD}/{RES}_r{REP}_d{D}.npz"
SEEDS = (0, 1, 2, 3, 4)


def seed_for(s):
    return random.Random(f"sc58|{RES}|r{REP}|{D}|{s}").randrange(10_000_000, 99_999_999)


def all_cells():
    odir = f"{OUT}/bench38_ln_{CFG}_{D // 1000}k_c{F}_{RES}_r{REP}"
    out = [
        dict(
            name=f"pool58_{RES}_r{REP}_1m",
            gpus=["h100", "v100"],  # the appended 400k go to the oracle
            ext=[],
            out_dir=RD,
            env={},
            needs=[BASE],
            cmd=(
                f"uv run --no-sync python scripts/build_pool_extend.py "
                f"--strategy {RES} --target {D} --base {R}/{BASE} --out {R}/{TOP} "
                f"--generation-seed {GSEED}"
            ),
            done=dict(type="file", path=TOP),
        )
    ]
    for s in SEEDS:
        out.append(
            dict(
                name=f"sc58_{RES}_r{REP}_1m_s{s}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=odir,
                env={},
                needs=[TOP],
                cmd=(
                    f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                    f"--D {D} --cfg {CFG} --pool {R}/{TOP} "
                    f"--seeds {s}={seed_for(s)} --ctx {F} "
                    f"&& touch {R}/{odir}/COMPLETE_s{s}"
                ),
                done=dict(type="file", path=f"{odir}/COMPLETE_s{s}"),
            )
        )
    return out


def main():
    cells = all_cells()
    pool = cells[0]
    assert f"--generation-seed {GSEED}" in pool["cmd"] and GSEED != 42, (
        "the extension would repeat the canonical draw"
    )
    assert f"--out {R}/{TOP}" in pool["cmd"] and f"_r{REP}_d{D}" in pool["cmd"], (
        "the extension would land under the canonical name"
    )
    assert os.path.exists(os.path.join(R, BASE)), f"{BASE} missing: nothing to extend"
    # the al54 lesson: a directory that already holds models is reported, not trained
    odir = os.path.join(R, OUT, f"bench38_ln_{CFG}_{D // 1000}k_c{F}_{RES}_r{REP}")
    assert not os.path.isdir(odir) or not [
        f for f in os.listdir(odir) if f.endswith("_meta.json")
    ], f"{odir} already holds models"
    print(f"guard: extending {BASE} to {D:,} at seed {GSEED}; output dir clear")

    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print([c["name"] for c in new])
            print()
            print(new[0]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov9f")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
