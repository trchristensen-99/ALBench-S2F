"""Nov-11: the 300k replicate draws that replicates 4, 5 and 6 of the acquisition ladder need.

`register_nov11b` and `register_nov11c` registered reps 3-6 of al55/al56/al57 this morning. Rep 3
runs; reps 4-6 do not, and the dispatcher has them PAUSED after three short failures each. The
cause is not the training cells but their candidate pools: `build_acq_candidates.py --reps 5` reads
`outputs/grid_pools/k562_reps/{strategy}_r5_d300000.npz` for each reservoir, every one of those
files is missing, so it skips all of them and aborts on `candidates labelled by different oracles:
set()`.

The guard those cells passed compared each new cell's COMMAND against its rep-1 twin. That was the
wrong invariant: it proves the experiment has the same shape, not that its inputs exist. The
`needs` list would have caught it had it named the replicate pools rather than the candidate file
the cell itself writes.

This builds the missing draws. One cell per (strategy, replicate): a fresh 300k draw at that
replicate's own generation seed, oracle-labelled, which is exactly how r1/r2/r3 were made
(`SEED0 = {1: 43, 2: 44, 3: 45}` in `slice_replicate_pools.py`, continued here as 4/5/6 -> 46/47/48).

Two things that make this simpler than the r1-r3 build was:

  * Reps 4-6 have no smaller pools on disk yet, so there is no prefix invariant to preserve and
    none of `extend_replicate_pools_300k.py`'s append logic is needed -- `install_replicate_pools.py`
    just slices prefixes of the new draw, and nesting holds by construction.
  * Only the five reservoirs that actually have r1-r3 at 300k are built, so reps 4-6 draw from the
    same five as reps 1-3. `evoaug_paper2023` and `genomic` have no 300k replicate at any rep, and
    adding them here would make the later replicates richer than the earlier ones, which is the one
    difference a replicate must not have.

15 cells, about 17 minutes of oracle time each (~300 seq/s), so roughly 4-5 GPU-hours in total.
Run `python scripts/install_replicate_pools.py --reps 4 5 6` once they finish; the paused
acquisition cells then find their inputs and the dispatcher resumes them.

    python scripts/program_oct22/register_nov11e.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402

R, DRY = m.R, m.DRY
RAW = "outputs/grid_pools/k562_reps_raw300k_new"
REPS_DIR = "outputs/grid_pools/k562_reps"
SIZE = 300_000
SEED0 = {1: 43, 2: 44, 3: 45, 4: 46, 5: 47, 6: 48}  # continues slice_replicate_pools.SEED0
NEW_REPS = (4, 5, 6)
# exactly the reservoirs that already have r1-r3 at 300k -- no more, no fewer
STRATS = (
    "motif_shared_core",
    "mutagenesis",
    "random",
    "zoonomia_orthologs",
    "motif_ct_enriched",
)


def main():
    assert set(SEED0[r] for r in NEW_REPS).isdisjoint({43, 44, 45}), (
        "a new replicate must not reuse an existing generation seed"
    )

    # the premise of this script: these reservoirs have r1-r3 at 300k and the new reps have nothing
    for s in STRATS:
        have = [r for r in (1, 2, 3) if os.path.exists(f"{R}/{REPS_DIR}/{s}_r{r}_d{SIZE}.npz")]
        assert have, f"{s}: no existing 300k replicate -- it is not one of the five, check STRATS"
    for s in STRATS:
        for rep in NEW_REPS:
            p = f"{R}/{REPS_DIR}/{s}_r{rep}_d{SIZE}.npz"
            assert not os.path.exists(p), f"{p} already exists; this would overwrite a draw"

    cells = []
    for rep in NEW_REPS:
        g = SEED0[rep]
        for s in STRATS:
            odir = f"{RAW}/{s}__g{g}"
            cells.append(
                dict(
                    name=f"poolrep_{s}_r{rep}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=odir,
                    env=dict(m.ENV),
                    needs=[],
                    cmd=(
                        f"uv run --no-sync python scripts/build_pool.py --strategy {s} "
                        f"--size {SIZE} --generation-seed {g} --out-dir {R}/{RAW} "
                        f"&& touch {R}/{odir}/COMPLETE"
                    ),
                    done=dict(type="file", path=f"{odir}/COMPLETE"),
                )
            )

    assert len({c["name"] for c in cells}) == len(cells)
    assert len({c["done"]["path"] for c in cells}) == len(cells), "two cells share a done marker"
    print(f"guard: {len(cells)} cells = {len(STRATS)} reservoirs x {len(NEW_REPS)} replicates")
    for rep in NEW_REPS:
        print(f"  rep {rep}: generation seed {SEED0[rep]}")

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
            for c in new:
                print(" ", c["name"])
            return
        shutil.copy(path, path + ".bak_nov11e")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
