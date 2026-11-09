"""Nov-8: is al52's acquisition gain real, or an escape from a bad pilot?

al52 starts every arm from the same 30,000-sequence genomic pilot, and genomic is the worst corpus in
the study on the designed axis -- at 100k it scores 0.604 where motif_shared_core scores 0.779. The
random control spends its 75,000 oligos at the candidate pool's equal 1/7 mix and so stays close to
where it started; the epistemic arm moves hard toward motif_shared_core (26%) and away from genomic
(8%). So the +0.102 designed gain has two readings, and al52 cannot tell them apart:

  the selector is good            -- it finds informative sequences wherever they are, or
  the pilot was bad              -- any move away from an all-genomic start would have helped this much.

The control is the same experiment from a pilot that is already the best corpus we know. If epistemic
still beats random by a wide margin from a motif_shared_core pilot, the gain is the selector. If the
margin mostly disappears, al52 measured the distance from genomic to motif_shared_core and the library
recommendation is "start from the right corpus", not "use an acquisition function".

Two arms only -- random and epistemic -- because badge already tracks epistemic at a constant offset and
the bound answers a different question. Two replicates each, matching al52 exactly in every other
respect: 30k pilot, 75,000 selected one-shot, ending at 105,000, same recipe, five seeds, same stack.
About 20 GPU-hours and no oracle labelling.

    python scripts/program_oct22/register_nov8i.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402
import register_nov8b as b  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
PD = "outputs/grid_pools/k562_acq54"
SEED_POOL = "outputs/grid_pools/k562_nested/motif_shared_core_d30000.npz"
REPS = (1, 2)
DONE = f"{OUT}/acq54_done"
ARMS = {"os_random": "random", "os_epi": "epistemic"}


def ladder(tag, acq, rep, needs):
    """al52's cell, with its candidate pool and its pilot swapped for the motif_shared_core ones."""
    done = f"{DONE}/{tag}.done"
    return dict(
        name=f"a1m_acq54_{tag}",
        gpus=["h100"],
        ext=[],
        out_dir=DONE,
        env={},
        needs=list(needs),
        cmd=(
            f"uv run --no-sync python scripts/program_oct22/acq_ladder.py "
            f"--seed_pool {SEED_POOL} --candidates {PD}/cand_r{rep}.npz "
            f"--schedule 75000 --acq {acq} --tag {tag} --seed {rep} "
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
                name=f"a1m_acqc54_r{rep}",
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
        out.append(ladder(f"pilot_s{rep}", "random", rep, [cand]))
    for arm, acq in ARMS.items():
        for rep in REPS:
            # the replicate's own candidate pool, not whichever one the loop above happened to end on
            needs = [f"{PD}/cand_r{rep}.npz", f"{DONE}/pilot_s{rep}.done"]
            out.append(ladder(f"{arm}_s{rep}", acq, rep, needs))
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
        assert b.SEED_POOL != SEED_POOL, "the control must not reuse al52's pilot"
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        for c in new:
            print(" ", c["name"])
        if DRY:
            print()
            print(new[0]["cmd"])
            print(new[-1]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov8i")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
