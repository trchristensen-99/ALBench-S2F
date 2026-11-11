"""Nov-11: al57 -- the fixed-geometry acquisition ladder at a 10k start, the rung below everything.

al56/al55/al53 run one geometry (pilot, then the same number again) from three starts: 30k -> 60k,
100k -> 200k, 300k -> 600k. The designed-set gain of selection over the control falls across them
(+0.0529, +0.0539, +0.0177), and the shape of that fall is the claim the scaling section makes.
Three points cannot separate "flat, then a collapse at 300k" from a smooth decline, and there is no
point at all below 30k.

10k is the rung worth adding, for two reasons beyond arithmetic:

  - it is the only size where the architecture-search loop BEATS the tuned LegNet bank (rr40 0.8029
    against 0.7801), so it is where the study's conclusions are least uniform;
  - it is the regime a new assay actually starts in. "Selection pays most when data is scarce" is
    the practical claim this program makes, and it currently has no measurement in the scarce half.

The honest alternative outcome is just as useful: a 10k pilot ensemble may be too weak to rank a
million candidates, in which case the gain turns over rather than continuing to rise, and the answer
to "how small can the pilot be before selection stops paying?" is a number rather than a guess.

The ladder stays under ONE recipe, which al56's could not do. `recipe_for` sends every point from
10,000 to 20,000 to the 10k rung (`e800_wd4`), so the os-vs-it3 contrast and the intermediate curve
points are all trained the same way; the guard asserts it rather than trusting it.

The tranches are 3334/3333/3333 because a 10,000-oligo increment does not divide into three equal
parts. They sum to exactly the increment, which is what the geometry requires; the unevenness is in
the intermediate curve points only and not in any arm comparison.

About 8 GPU-hours, below al56's 12 at twice the size. No oracle labelling: every candidate is
already labelled and the selector is never shown a label.

    python scripts/program_oct22/register_nov11a.py [--dry]
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
PD = "outputs/grid_pools/k562_acq57"
SEED_POOL = "outputs/grid_pools/k562_nested/motif_shared_core_d10000.npz"
REPS = (1, 2)
DONE = f"{OUT}/acq57_done"
D0, TRANCHES = 10_000, (3334, 3333, 3333)
ARMS = {
    "os_random": ("random", str(sum(TRANCHES))),
    "os_epi": ("epistemic", str(sum(TRANCHES))),
    "it3_epi": ("epistemic", ",".join(str(t) for t in TRANCHES)),
}


def ladder(tag, acq, schedule, rep, needs):
    done = f"{DONE}/{tag}.done"
    return dict(
        name=f"a1m_acq57_{tag}",
        gpus=["h100", "v100"],  # 20k x 5 seeds is minutes, even on a V100
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
                name=f"a1m_acqc57_r{rep}",
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
        # the pilot ensemble at 10k, trained once per replicate; every arm of that replicate waits on it
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
    assert top == 2 * D0, f"the geometry is pilot then the same again, not {top}"
    assert sum(TRANCHES) == D0, f"tranches must sum to the increment, not {sum(TRANCHES)}"
    # unlike al56, this ladder never crosses a rung boundary: every point trains under one recipe,
    # so the intermediate curve points are comparable with the arms rather than only with each other
    pts = (D0, D0 + TRANCHES[0], D0 + TRANCHES[0] + TRANCHES[1], top)
    recipes = {d: al.recipe_for(d) for d in pts}
    print(f"recipe along the ladder: {recipes}")
    assert len(set(recipes.values())) == 1, f"the ladder crosses a rung boundary: {recipes}"
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
        shutil.copy(path, path + ".bak_nov11a")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
