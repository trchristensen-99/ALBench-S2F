"""Nov-8: al53 -- does buying in rounds still pay when you start from 300k?

al52 answered the rounds question at one starting point: from a 30k pilot, three tranches of 25k beat a
single 75k batch by +0.0085 on the designed set, real but an order of magnitude smaller than the gain
from choosing the selector at all. The library decision is not at 30k, though. It is at the scale where
splitting an order costs real money and calendar time, and where the pilot model is already good enough
that there may be little left for a second look to find.

So: the same experiment one decade up. A 300,000-sequence pilot, 300,000 more oligos, every arm ending
at 600,000, with the only difference being when the choice was made.

  os_random    300,000 at the candidate pool's own mix          -- the control
  os_epi       300,000 chosen once from the pilot ensemble      -- 1 x 300k
  it3_epi      100,000, retrain, 100,000, retrain, 100,000      -- 3 x 100k

Epistemic only. al52 measured badge below epistemic on all three axes at both schedules and `oracle_top`
is a bound that al52 has already pinned at n=2, so spending four more 600k ensembles to re-measure
either would buy nothing this experiment needs.

**The pilot is motif_shared_core, not genomic.** al52 started from genomic because that is the realistic
pilot at 30k, but a 300k genomic seed cannot work here: genomic draws are ~95% identical to each other
at 300k (the pool ceiling is ~315k), so excluding the seed would leave almost no genomic candidates and
the candidate pool would silently stop being an equal seven-way mix. motif_shared_core has a 0.0%
duplicate rate across independent draws, scales past 1M, and wins the designed axis at every size -- and
it is the same pilot al54 uses at 30k, so al53 and al54 together give a 30k-start and a 300k-start curve
on one corpus, which is the comparison worth having.

Every rung takes `e300_wd05` from the per-size table: 400k, 500k and 600k all fall to the 300k rung, and
the Nov-8 bracket showed e300 is at or past the knee even at 1M. About 100 GPU-hours, no oracle
labelling -- every candidate is already labelled and the selector is never shown a label.

    python scripts/program_oct22/register_nov8j.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
PD = "outputs/grid_pools/k562_acq53"
SEED_POOL = "outputs/grid_pools/k562_nested/motif_shared_core_d300000.npz"
REPS = (1, 2)
DONE = f"{OUT}/acq53_done"
ARMS = {
    "os_random": ("random", "300000"),
    "os_epi": ("epistemic", "300000"),
    "it3_epi": ("epistemic", "100000,100000,100000"),
}


def ladder(tag, acq, schedule, rep, needs):
    done = f"{DONE}/{tag}.done"
    return dict(
        name=f"a1m_acq53_{tag}",
        gpus=["h100"],  # 600k x 5 seeds: a V100 would take two and a half times as long
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
                name=f"a1m_acqc53_r{rep}",
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
        # the pilot ensemble at 300k, trained once per replicate; every arm of that replicate waits on it
        out.append(ladder(f"pilot_s{rep}", "random", "0", rep, [cand]))
    for arm, (acq, schedule) in ARMS.items():
        for rep in REPS:
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
            print(new[0]["cmd"])
            print(new[-1]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov8j")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
