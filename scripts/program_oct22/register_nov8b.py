"""Nov-8: does buying the oligo budget in ROUNDS beat buying it in one batch?

Every result in this study so far compares corpora of a fixed size chosen in advance. The library decision
has a second axis nobody has measured: a 3.2M-oligo order can be split, with each tranche chosen by a model
trained on the ones before it. Splitting costs real money and calendar time, so it has to earn them.

The design starts from the realistic position -- a 30k pilot of real human CREs already measured -- and
spends 75k more oligos, ending every arm at 105k labelled sequences from the same candidate pool with the
same per-size recipe. The only thing that varies is when the choice was made and what made it:

  one-shot   select all 75,000 from the pilot ensemble
  3 x 25k    select 25,000, retrain, select again, retrain, select again

and, crossed with that, the selector: `epistemic` (top-k across-seed disagreement, which is the library's
`badge_epistemic_only` control -- uncertainty with no diversity term) and `badge` (the same uncertainty used
to weight the embedding, with k-means++ for diversity). `random` is the control and it runs one-shot only:
three random draws without replacement ARE one random draw of the same size, so an iterative random arm
would spend three ensembles reproducing its own one-shot result. `oracle_top` selects on the hidden oracle
label and is a bound, not a method -- it says how much of any gain is available to an activity-chaser.

The candidate pool is 1M sequences mixed equally across seven reservoirs, so the selector also chooses the
corpus composition. That is the part worth watching: rp47 and mx51 found the best composition by brute
force, and an acquisition function that rediscovers it is a much stronger result than one that merely picks
good sequences within a reservoir. Each arm logs what it bought, by reservoir, every round.

Two replicates, each drawing candidates from a different replicate of every reservoir and training its pilot
with different seeds. About 112 GPU-hours and no oracle labelling: every candidate is already labelled, and
the selector is never shown a label.

    python scripts/program_oct22/register_nov8b.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
PD = "outputs/grid_pools/k562_acq52"
SEED_POOL = "outputs/grid_pools/k562_nested/genomic_d30000.npz"
REPS = (1, 2)
DONE = f"{OUT}/acq52_done"

# arm -> (acquisition, schedule, which replicates). The bound runs once: it is not a method and its
# variance across replicates is not a quantity anyone will quote.
ARMS = {
    "os_random": ("random", "75000", REPS),
    "os_epi": ("epistemic", "75000", REPS),
    "os_badge": ("badge", "75000", REPS),
    "it3_epi": ("epistemic", "25000,25000,25000", REPS),
    "it3_badge": ("badge", "25000,25000,25000", REPS),
    "os_oracle": ("oracle_top", "75000", (1,)),
}


def ladder(tag, acq, schedule, rep, needs):
    done = f"{DONE}/{tag}.done"
    return dict(
        name=f"a1m_acq52_{tag}",
        gpus=["h100"],
        ext=[],
        out_dir=DONE,
        env={},
        needs=list(needs),
        cmd=(
            f"uv run --no-sync python scripts/program_oct22/acq_ladder.py "
            f"--seed_pool {SEED_POOL} --candidates {PD}/cand_r{rep}.npz "
            f"--schedule {schedule} --acq {acq} --tag {tag} --seed {rep} "
            f"--n_seeds 5 --ctx {F} --pool_dir {PD} --out_root {OUT} --done {done}"
        ),
        done=dict(type="file", path=done),
    )


def all_cells():
    out = []
    for rep in REPS:
        cand = f"{PD}/cand_r{rep}.npz"
        o = f"{PD}/_built_cand_r{rep}"
        out.append(
            dict(
                name=f"a1m_acqc52_r{rep}",
                gpus=["h100", "v100"],
                ext=[],
                out_dir=o,
                env={},
                needs=[SEED_POOL],
                cmd=(
                    f"uv run --no-sync python scripts/build_acq_candidates.py "
                    f"--seed-pool {SEED_POOL} --n 1000000 --reps {rep} --out {cand} "
                    f"&& mkdir -p {R}/{o} && touch {R}/{o}/COMPLETE"
                ),
                done=dict(type="file", path=f"{o}/COMPLETE"),
            )
        )
        # the pilot ensemble, trained once per replicate; every arm of that replicate waits on it
        out.append(ladder(f"pilot_s{rep}", "random", "0", rep, [cand]))
    for arm, (acq, schedule, reps) in ARMS.items():
        for rep in reps:
            out.append(
                ladder(
                    f"{arm}_s{rep}",
                    acq,
                    schedule,
                    rep,
                    [f"{PD}/cand_r{rep}.npz", f"{DONE}/pilot_s{rep}.done"],
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
            print(new[-1]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov8b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
