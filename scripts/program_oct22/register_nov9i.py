"""Nov-9: al55 -- the 100k-start curve, so the fall-off can be attributed to scale rather than geometry.

Today's two acquisition curves disagree about how much selection is worth, but they differ in two things
at once. al52/al54 start from a 30k pilot and buy 3.5x it (+0.041 designed at 105k); al53 starts from
300k and buys 1x it (+0.014 at the first increment, level by 500k). Start size and multiple-of-pilot are
confounded, so "selection stops paying above ~100k" is not yet a claim the data supports.

al55 copies al53's geometry exactly -- pilot, then the same number again, three equal tranches against
one -- and changes only the starting point:

  os_random    100,000 at the candidate pool's own mix           -- the control
  os_epi       100,000 chosen once from the pilot ensemble       -- 1 x 100k
  it3_epi      34,000, retrain, 33,000, retrain, 33,000          -- 3 x ~33k

Rungs land at 134k / 167k / 200k; the one-shot arms land only at 200k. Every rung takes `e500` from the
per-size table (`recipe_for` sends anything up to 230k to the 100k rung), so the recipe is constant along
the ladder and the only thing varying is when the choice was made.

Same pilot as al53 and al54 -- motif_shared_core, 0.0% duplicates across independent draws, scales past
1M -- so the three curves sit on one corpus and are directly comparable.

**V100-eligible, unlike al53.** al53 pinned H100 because a 600k x 5-seed ensemble is two and a half times
slower on a V100. At a 200k endpoint a V100 model is ~2 h, which slow_nice's 30-day wall absorbs easily,
and the free capacity this arm was registered to fill is V100 capacity (16 idle, 0 idle H100).

About 41 H100-hours, or ~98 V100-hours if it all lands there. No oracle labelling: every candidate is
already labelled and the selector is never shown a label.

    python scripts/program_oct22/register_nov9i.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acq_ladder as al  # noqa: E402
import register_ln100 as ln  # noqa: E402

R, OUT, F, DRY = ln.R, ln.OUT, ln.F, ln.DRY
PD = "outputs/grid_pools/k562_acq55"
SEED_POOL = "outputs/grid_pools/k562_nested/motif_shared_core_d100000.npz"
REPS = (1, 2)
DONE = f"{OUT}/acq55_done"
D0, TRANCHES = 100_000, (34_000, 33_000, 33_000)
ARMS = {
    "os_random": ("random", str(sum(TRANCHES))),
    "os_epi": ("epistemic", str(sum(TRANCHES))),
    "it3_epi": ("epistemic", ",".join(str(t) for t in TRANCHES)),
}


def ladder(tag, acq, schedule, rep, needs):
    done = f"{DONE}/{tag}.done"
    return dict(
        name=f"a1m_acq55_{tag}",
        gpus=["h100", "v100"],  # 200k x 5 seeds is ~2 h per model on a V100; see the docstring
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
                name=f"a1m_acqc55_r{rep}",
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
        # the pilot ensemble at 100k, trained once per replicate; every arm of that replicate waits on it
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
    cells = all_cells()
    # the ladder is only a scale comparison if every rung trains under the same recipe
    rungs = [D0 + sum(TRANCHES[: i + 1]) for i in range(len(TRANCHES))]
    recipes = {d: al.recipe_for(d) for d in [D0, *rungs]}
    assert len(set(recipes.values())) == 1, f"the recipe changes along the ladder: {recipes}"
    assert rungs[-1] == 2 * D0, f"al53's geometry is pilot then the same again, not {rungs[-1]}"
    assert os.path.exists(os.path.join(R, SEED_POOL)), (
        f"{SEED_POOL} missing: no pilot to start from"
    )
    for c in cells:
        assert c["gpus"] == ["h100", "v100"], c["gpus"]
        assert f"{PD}/" in c["cmd"], "a cell would write into another experiment's pool dir"
    print(
        f"guard: {len(cells)} cells, h100+v100, rungs {rungs} all on {set(recipes.values()).pop()}"
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
            print(new[-1]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov9i")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
