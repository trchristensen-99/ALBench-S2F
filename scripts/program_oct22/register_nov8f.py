"""Nov-8: the ms:mc RATIO, which mx51 left unmeasured.

mx51 settled WHICH reservoirs to mix. Half-and-half motif_shared_core + motif_ct_enriched beat both of its
parents on held-out genomic (0.9582) and on random (0.9572, above the better parent, so superadditive), and
adding real CREs strictly hurt at matched size. But motif_shared_core ALONE still won the designed set by
0.010 over the 50/50 mixture, and that is larger than the spread between replicate draws -- so 50/50 may be
buying genomic and random accuracy with designed-set accuracy rather than dominating. If a 2:1 ms:mc corpus
keeps the genomic and random gains while recovering the designed set, it is the recommendation and 50/50 is
not; if neither bracket beats 50/50 on any axis, the ratio is a flat direction and the question is closed.

Two ratios one step either side of the measured point, at both sizes mx51 reported, on the same three
replicate draws so the error bars are comparable with its. About 56 GPU-hours and no oracle labelling: a
mixture is interleaved from pools that are already labelled, so this costs training time only.

    python scripts/program_oct22/register_nov8f.py [--dry]
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
PD = "outputs/grid_pools/k562_mixtures"
MIXES = ("mix_ms2_mc1", "mix_ms1_mc2")
RUNGS = {100_000: "e500", 300_000: "e300_wd05"}  # the per-size recipe, as mx51 used
REPS = (1, 2, 3)
SEEDS = (0, 1)


def seed_for(mix, D, rep, s):
    """Deterministic but distinct from mx51's draws, so a rerun reproduces and a comparison is paired."""
    return random.Random(f"mx53|{mix}|{D}|{rep}|{s}").randrange(10_000_000, 99_999_999)


def all_cells():
    out = []
    for mix in MIXES:
        mark = f"{PD}/_built_{mix}"
        out.append(
            dict(
                name=f"pool53_{mix}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=mark,
                env={},
                needs=[f"outputs/grid_pools/k562_reps/motif_shared_core_r3_d300000.npz"],
                cmd=(
                    f"uv run --no-sync python scripts/build_mixture_pools.py --only {mix} "
                    f"&& mkdir -p {R}/{mark} && touch {R}/{mark}/COMPLETE"
                ),
                done=dict(type="file", path=f"{mark}/COMPLETE"),
            )
        )
        for D, cfg in RUNGS.items():
            for rep in REPS:
                pool = f"{PD}/{mix}_r{rep}_d{D}.npz"
                odir = f"{OUT}/bench38_ln_{cfg}_{D // 1000}k_c{F}_{mix}_r{rep}"
                for s in SEEDS:
                    out.append(
                        dict(
                            name=f"mx53_{mix}_{D // 1000}k_r{rep}_s{s}",
                            gpus=["h100"],
                            ext=[],
                            out_dir=odir,
                            env={},
                            needs=[pool],
                            cmd=(
                                f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                                f"--D {D} --cfg {cfg} --pool {R}/{pool} "
                                f"--seeds {s}={seed_for(mix, D, rep, s)} --ctx {F} "
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
        shutil.copy(path, path + ".bak_nov8f")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
