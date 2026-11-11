"""Nov-11: al58 -- the acquisition ladder seeded from GENOMIC, at the 30k and 300k starts.

Every rounds-vs-one-shot rung in the program so far seeds from `motif_shared_core`: al55, al56,
al57 and al52/al54 all start from that reservoir, and only the BADGE / oracle-top / k-means++ arms
were ever seeded from genomic -- which is why the method comparison had to be split by seed pool
and why the genomic panel has no random control.

That leaves the question a reader asks first unanswered: if you already hold N genomic sequences,
what does selection buy? Genomic is the corpus anyone would actually start from, and it is the one
seed pool for which we cannot currently say. This runs the same three arms -- random control,
one-shot ensemble-disagreement, three-round ensemble-disagreement -- from genomic at two starts:

    30k  + 30k   -> 60,000     3 x 10k against 1 x 30k, two replicates
    300k + 300k  -> 600,000    3 x 100k against 1 x 300k, ONE replicate

The geometry matches al56 and al53 exactly, so a genomic start can be read against the
motif_shared_core start at the same rung with nothing else varying. The 300k rung takes one
replicate rather than two because a 600k ensemble is roughly fifteen GPU-hours per arm; a second
replicate would put this over the sign-off threshold on its own.

100k is deliberately left out: it would add about 50 GPU-h and the 30k and 300k ends bracket it.

About 80 GPU-hours.

    python scripts/program_oct22/register_nov11g.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
PD = "outputs/grid_pools/k562_acq58"
DONE = f"{OUT}/acq58_done"
NEST = "outputs/grid_pools/k562_nested"

# start size -> (seed pool, tranches, replicates, which GPUs the arm may use)
RUNGS = {
    30_000: (f"{NEST}/genomic_d30000.npz", (10_000, 10_000, 10_000), (1, 2), ["h100", "v100"]),
    300_000: (f"{NEST}/genomic_d300000.npz", (100_000, 100_000, 100_000), (1,), ["h100"]),
}
ARMS = {
    "os_random": "random",
    "os_epi": "epistemic",
    "it3_epi": "epistemic",
}


def schedule_for(arm: str, tranches: tuple[int, ...]) -> str:
    return ",".join(str(t) for t in tranches) if arm.startswith("it3") else str(sum(tranches))


def ladder(tag, acq, schedule, rep, needs, seed_pool, gpus):
    done = f"{DONE}/{tag}.done"
    return dict(
        name=f"a1m_acq58_{tag}",
        gpus=list(gpus),
        ext=[],
        out_dir=DONE,
        env={},
        needs=list(needs),
        cmd=(
            f"uv run --no-sync python scripts/program_oct22/acq_ladder.py "
            f"--seed_pool {seed_pool} --candidates {PD}/cand_r{rep}.npz "
            f"--schedule {schedule} --acq {acq} --tag {tag} --seed {rep} "
            f"--n_seeds 5 --ctx {F} --pool_dir {PD} --out_root {OUT} --done {done}"
        ),
        done=dict(type="file", path=done),
    )


def all_cells():
    out, built = [], set()
    for D0, (seed_pool, tranches, reps, gpus) in RUNGS.items():
        k = D0 // 1000
        for rep in reps:
            cand = f"{PD}/cand_r{rep}.npz"
            o = f"{PD}/_built_cand_r{rep}"
            if rep not in built:
                built.add(rep)
                out.append(
                    dict(
                        name=f"a1m_acqc58_r{rep}",
                        gpus=["h100", "v100"],
                        ext=[],
                        out_dir=o,
                        env={},
                        needs=[],
                        cmd=(
                            f"uv run --no-sync python scripts/build_acq_candidates.py "
                            f"--seed-pool {seed_pool} --n 1000000 --reps {rep} --out {cand} "
                            f"&& mkdir -p {R}/{o} && touch {R}/{o}/COMPLETE"
                        ),
                        done=dict(type="file", path=f"{o}/COMPLETE"),
                    )
                )
            # the pilot ensemble at D0, once per (rung, replicate); every arm of it waits here
            out.append(ladder(f"g{k}_pilot_s{rep}", "random", "0", rep, [cand], seed_pool, gpus))
        for arm, acq in ARMS.items():
            for rep in reps:
                out.append(
                    ladder(
                        f"g{k}_{arm}_s{rep}",
                        acq,
                        schedule_for(arm, tranches),
                        rep,
                        [f"{PD}/cand_r{rep}.npz", f"{DONE}/g{k}_pilot_s{rep}.done"],
                        seed_pool,
                        gpus,
                    )
                )
    return out


def main():
    cells = all_cells()
    for D0, (seed_pool, tranches, reps, gpus) in RUNGS.items():
        assert os.path.exists(os.path.join(R, seed_pool)), f"missing {seed_pool}"
        assert sum(tranches) == D0, f"{D0}: tranches must sum to the start size, al53/al56 geometry"
    for c in cells:
        assert "--acq " in c["cmd"] and "--seed_pool " in c["cmd"], c["name"]
        # a 600k ensemble on a V100 is past what the wall will grant; it would be killed, not slow
        if "g300_" in c["name"]:
            assert c["gpus"] == ["h100"], c["name"]
    assert len({c["name"] for c in cells}) == len(cells), "duplicate cell name"
    assert len({c["done"]["path"] for c in cells}) == len(cells), "two cells share a done marker"
    n30 = sum(1 for c in cells if "_g30_" in c["name"])
    n300 = sum(1 for c in cells if "_g300_" in c["name"])
    print(
        f"guard: {len(cells)} cells -- {n30} at the 30k start, {n300} at 300k, "
        f"{len(cells) - n30 - n300} candidate builds"
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
        if DRY:
            for c in new:
                print(" ", c["name"])
            return
        shutil.copy(path, path + ".bak_nov11g")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
