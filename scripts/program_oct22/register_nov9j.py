"""Nov-9: al56 -- al53's geometry at the 30k start, so scale and multiple-of-pilot come apart.

al52/al54 buy 3.5x their pilot and gain +0.041 designed; al53 buys 1x its pilot and gains +0.014 at the
first increment. Two things differ at once, and the cheap reading is that the multiple explains it. The
existing intermediate rungs already argue against that -- al54 at 55k has bought 1.83x its pilot and is
+0.0353 designed over the same-size scratch curve, against +0.0135 for al53 at 400k (1.33x) -- but that
contrast is selected-vs-drawn across two experiments, not os-vs-random inside one, so it is suggestive
rather than decisive.

al56 settles it with the within-experiment contrast at the matched multiple: a 30k pilot, 30k more
oligos, every arm ending at 60,000.

  os_random    30,000 at the candidate pool's own mix       -- the control
  os_epi       30,000 chosen once from the pilot ensemble   -- 1 x 30k
  it3_epi      10,000, retrain, 10,000, retrain, 10,000     -- 3 x 10k

Together with al55 (100k -> 200k) and al53 (300k -> 600k) this gives three starts at one fixed geometry,
which is the axis the curves were missing. Pilot motif_shared_core, the same as al53/al54/al55.

Note on the recipe, since al55's guard asserted the opposite: this ladder *does* cross a rung boundary.
`recipe_for` sends 40k to the 30k rung (`e500_wd2`) and 50k/60k to the 55k rung (`e500_wd15`). That is
the study's per-size standard and is not overridden here -- every arm at the top rung still trains under
one recipe, so the os-vs-it3 comparison is clean; only it3's two intermediate curve points sit either
side of the boundary, and they are read as curve shape rather than as an arm comparison.

About 12 GPU-hours, the cheapest arm in the program. No oracle labelling: every candidate is already
labelled and the selector is never shown a label.

    python scripts/program_oct22/register_nov9j.py [--dry]
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
PD = "outputs/grid_pools/k562_acq56"
SEED_POOL = "outputs/grid_pools/k562_nested/motif_shared_core_d30000.npz"
REPS = (1, 2)
DONE = f"{OUT}/acq56_done"
D0, TRANCHES = 30_000, (10_000, 10_000, 10_000)
ARMS = {
    "os_random": ("random", str(sum(TRANCHES))),
    "os_epi": ("epistemic", str(sum(TRANCHES))),
    "it3_epi": ("epistemic", ",".join(str(t) for t in TRANCHES)),
}


def ladder(tag, acq, schedule, rep, needs):
    done = f"{DONE}/{tag}.done"
    return dict(
        name=f"a1m_acq56_{tag}",
        gpus=["h100", "v100"],  # 60k x 5 seeds is well under an hour even on a V100
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
                name=f"a1m_acqc56_r{rep}",
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
        # the pilot ensemble at 30k, trained once per replicate; every arm of that replicate waits on it
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
    top = D0 + sum(TRANCHES)
    assert top == 2 * D0, f"al53's geometry is pilot then the same again, not {top}"
    # the arms are only comparable with each other if they all land under one recipe at the top rung;
    # the intermediate rungs may differ (this ladder crosses the 55k boundary -- see the docstring)
    ladder_recipes = {d: al.recipe_for(d) for d in (D0, 40_000, 50_000, top)}
    print(f"recipe along the ladder: {ladder_recipes}")
    assert os.path.exists(os.path.join(R, SEED_POOL)), (
        f"{SEED_POOL} missing: no pilot to start from"
    )
    for c in cells:
        assert c["gpus"] == ["h100", "v100"], c["gpus"]
        assert f"{PD}/" in c["cmd"], "a cell would write into another experiment's pool dir"
    print(f"guard: {len(cells)} cells, h100+v100, top rung {top:,} under {al.recipe_for(top)}")

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
        shutil.copy(path, path + ".bak_nov9j")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
