"""Nov-9: give the 400k/500k/600k from-scratch rungs a second draw, so al53's comparison has a band.

`read_curves.py` shows the from-scratch curve measured at three or four independent draws up to 300k
(+-0.008 to +-0.030, which is 2-5x the seed-only spread) and at exactly one draw above it. al53's arms
land at 400k, 500k and 600k in about a day, and comparing them against a single-draw from-scratch number
is the same mistake the replicates were registered to fix one rung lower.

The replicate raws for motif_shared_core stop at 300k (`k562_reps_raw1m` holds only motif_ct_enriched and
random), so this one has to be drawn and labelled rather than sliced. `build_pool_extend.py` is the right
tool and the cheap one: it keeps the existing r1 pool byte-identical as a prefix and sends only the
appended 300k to the oracle, so the r1 rungs at 400k+ continue the very same draw already used at
30k-300k instead of starting a new ordering halfway up the curve. Its docstring also records that
motif_shared_core was the one reservoir measured prefix-stable (Nov 7), which is why this works here.

Two things it would get wrong if left to its defaults, both set explicitly below:
  --generation-seed  defaults to 42, the *canonical* seed. r1 is seed 43 (slice_replicate_pools.SEED0),
                     and drawing the extension from 42 would append canonical sequences to a replicate.
  --out              defaults to `{strategy}_d{target}.npz`, which drops the `_r1` and would land a
                     replicate pool under the canonical name.

One extra draw, not two: that is enough for a band and matches the two draws al52/al54 each carry. r2
can follow if the band turns out ambiguous. ~45 GPU-hours, nearly all of it V100-eligible training.

    python scripts/program_oct22/register_nov9c.py [--dry]
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
BASE = f"{RD}/{RES}_r{REP}_d300000.npz"
TOP = f"{RD}/{RES}_r{REP}_d600000.npz"
RUNGS = {400_000: "e300_wd05", 500_000: "e300_wd05", 600_000: "e300_wd05"}
SEEDS = (0, 1, 2, 3, 4)


def seed_for(D, s):
    return random.Random(f"sc57|{RES}|r{REP}|{D}|{s}").randrange(10_000_000, 99_999_999)


def all_cells():
    slice_sizes = ",".join(str(d) for d in RUNGS if d < 600_000)
    out = [
        dict(
            name=f"pool57_{RES}_r{REP}_600k",
            gpus=["h100", "v100"],  # the appended 300k go to the oracle
            ext=[],
            out_dir=RD,
            env={},
            needs=[BASE],
            cmd=(
                f"uv run --no-sync python scripts/build_pool_extend.py "
                f"--strategy {RES} --target 600000 --base {R}/{BASE} --out {R}/{TOP} "
                f"--generation-seed {GSEED} "
                f"&& uv run --no-sync python scripts/slice_nested_prefix.py "
                f"--dir {RD} --strategy {RES}_r{REP} --sizes {slice_sizes}"
            ),
            done=dict(type="file", path=f"{RD}/{RES}_r{REP}_d400000.npz"),
        )
    ]
    for D, cfg in RUNGS.items():
        odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{RES}_r{REP}"
        for s in SEEDS:
            out.append(
                dict(
                    name=f"sc57_{RES}_r{REP}_{D // 1000}k_s{s}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=odir,
                    env={},
                    needs=[f"{RD}/{RES}_r{REP}_d{D}.npz"],
                    cmd=(
                        f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                        f"--D {D} --cfg {cfg} --pool {R}/{RD}/{RES}_r{REP}_d{D}.npz "
                        f"--seeds {s}={seed_for(D, s)} --ctx {F} "
                        f"&& touch {R}/{odir}/COMPLETE_s{s}"
                    ),
                    done=dict(type="file", path=f"{odir}/COMPLETE_s{s}"),
                )
            )
    return out


def main():
    cells = all_cells()
    assert f"--generation-seed {GSEED}" in cells[0]["cmd"] and GSEED != 42, (
        "replicate drawn at the canonical seed"
    )
    assert f"--out {R}/{TOP}" in cells[0]["cmd"], (
        "the extension would land under the canonical name"
    )
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
            return
        shutil.copy(path, path + ".bak_nov9c")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
