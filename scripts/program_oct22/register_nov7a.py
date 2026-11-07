"""Nov-7: regenerate every zoonomia_orthologs pool now that the sampler deduplicates.

A zoonomia draw emits one sequence per callable species per window, and in a conserved window many species carry the IDENTICAL 200-mer. Measured Nov 6: 15.2-15.8% of every draw was exact repeats, at every size from 30k
to 300k, while every other reservoir in the study is at 0.0%. So the arm has been competing at a sixth less distinct data than its nominal size all along, and a real library would not pay to synthesise the same
oligo twice. `ZoonomiaOrthologSampler(dedupe=True)` is now the default and a 5k draw measures 16.5% -> 0.0% with the same 43 species represented.

This rebuilds the four draws the study actually consumes -- the canonical pool (generation seed 42) and the three replicates (43/44/45) -- at 300k each, from which every smaller size is sliced as a prefix. About
4.5 GPU-hours, nearly all of it oracle labelling.

The old pools and the results trained on them are NOT deleted: `scripts/supersede_zoonomia_dup.py` moves both aside, which both keeps the before/after comparison (what does 15% duplication cost?) and makes the
dispatcher re-run the affected cells against the new pools.

    python scripts/program_oct22/register_nov7a.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov2b as nb  # noqa: E402

R, DRY = nb.R, nb.DRY
STRAT = "zoonomia_orthologs"
# seed 42 is the canonical pool every non-replicate cell reads; 43/44/45 are the three replicate draws
SEEDS = (42, 43, 44, 45)
SIZE = 300_000


def all_cells():
    out = []
    for gseed in SEEDS:
        o = f"outputs/grid_pools/k562_zoo_dedupe_raw300k/{STRAT}__g{gseed}"
        out.append(
            dict(
                name=f"zoo7_{STRAT}_g{gseed}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=o,
                env={},
                needs=[],
                cmd=(
                    f"uv run --no-sync python scripts/build_pool.py --strategy {STRAT} "
                    f"--size {SIZE} --generation-seed {gseed} --out-dir {R}/{o} "
                    f"&& touch {R}/{o}/COMPLETE"
                ),
                done=dict(type="file", path=f"{o}/COMPLETE"),
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
        if DRY:
            for c in new:
                print(" ", c["name"], "->", c["out_dir"])
            print(new[0]["cmd"][:220])
            return
        shutil.copy(path, path + ".bak_nov7a")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
