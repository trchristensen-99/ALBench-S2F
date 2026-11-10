"""Nov-10: the BADGE control AGENTS.md requires, at the geometry where BADGE was actually measured.

AGENTS.md is explicit: "If you add an acquisition method, add its controls at the same time."
BADGE has been run in the scaling curves since al52 (`os_badge`, `it3_badge`) and its controls
never have -- `acq_ladder.select()` only names `badge_kmeanspp_only` in a docstring, so the arm
could not have run even if it had been registered. That is now implemented and this registers it.

The control matters beyond compliance, because the measured numbers are strange. At +75k on a 30k
genomic pilot the ranking is epistemic .8104 designed, badge .7790, random .7082. BADGE is the
fusion of epistemic weighting with k-means++ coverage, yet it LOSES to epistemic alone by .031.
Two readings fit that: either the k-means++ half actively dilutes a good uncertainty signal, or the
embedding geometry is doing the work and the epistemic weighting is incidental. They imply opposite
things about what to buy, and only the unweighted-coverage arm separates them:

    epistemic             uncertainty, no coverage        measured, .8104
    badge                 uncertainty x coverage          measured, .7790
    badge_kmeanspp_only   coverage, no uncertainty        THIS ARM
    random                neither                         measured, .7082

If coverage-only lands near BADGE, the k-means++ half is what BADGE is actually buying and the
epistemic weighting is noise. If it lands near random, BADGE's fusion is destroying a signal that
epistemic alone keeps. If it lands between random and epistemic, both halves contribute and the
fusion is simply mis-weighted.

Same pool, same pilot, same seeds and the same 5-member tuned LegNet ensemble as every other al52
arm, so the four numbers are directly comparable. One-shot only: the rounds axis is already
measured and adds +0.009 designed, far less than the spread being resolved here.

    python scripts/program_oct22/register_nov10c.py [--dry]
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
PD = "outputs/grid_pools/k562_acq52"
SEED_POOL = "outputs/grid_pools/k562_nested/genomic_d30000.npz"
REPS = (1, 2)
DONE = f"{OUT}/acq52_done"
D0, BUDGET = 30_000, 75_000
ARMS = {"os_kmeanspp": ("badge_kmeanspp_only", str(BUDGET))}


def ladder(tag, acq, schedule, rep):
    done = f"{DONE}/{tag}_s{rep}.done"
    return dict(
        name=f"a1m_acq52_{tag}_s{rep}",
        gpus=["h100", "v100"],
        ext=[],
        out_dir=DONE,
        env={},
        needs=[f"{PD}/cand_r{rep}.npz", SEED_POOL],
        cmd=(
            f"uv run --no-sync python scripts/program_oct22/acq_ladder.py "
            f"--seed_pool {SEED_POOL} --candidates {PD}/cand_r{rep}.npz "
            f"--schedule {schedule} --acq {acq} --tag {tag} --seed {rep} "
            f"--n_seeds 5 --ctx {F} --pool_dir {PD} --out_root {OUT} --done {done}"
        ),
        done=dict(type="file", path=done),
    )


def main():
    # the control must be a method select() can actually dispatch, which is what was missing
    import inspect

    src = inspect.getsource(al.select)
    for method, _ in ARMS.values():
        assert f'method == "{method}"' in src, f"acq_ladder.select() cannot dispatch {method!r}"
    assert 'want_emb=method in ("badge", "badge_kmeanspp_only")' in src, (
        "the control needs embeddings; select() must request them for it too"
    )
    # ...and the CLI has to accept it. Checking only select() passed a first registration whose
    # jobs then died at argparse: "invalid choice: 'badge_kmeanspp_only'". A dispatchable method
    # that no command line can reach is not a runnable arm.
    cli = inspect.getsource(al.main)
    for method, _ in ARMS.values():
        assert f'"{method}"' in cli, f"--acq does not accept {method!r}"
        # ...and main() has to build the pilot ensemble for it. `needs_model` was a list of the
        # methods that DO consult the model, so this one was omitted and `ens` stayed None until
        # select() dereferenced it. Asserting the inverse spelling keeps that from recurring.
        assert 'needs_model = a.acq not in ("random", "oracle_top")' in cli, (
            "needs_model must list the methods that do NOT consult the model"
        )
    # it has to sit beside the arms it explains, so geometry and recipe must match al52's
    assert al.recipe_for(D0 + BUDGET) == al.recipe_for(105_000), "not al52's rung"
    print(
        f"guard: select() dispatches {[m for m, _ in ARMS.values()]}, "
        f"{D0 + BUDGET:,} on {al.recipe_for(D0 + BUDGET)}"
    )

    cells = [ladder(tag, acq, sched, rep) for tag, (acq, sched) in ARMS.items() for rep in REPS]
    for c in cells:
        assert os.path.exists(os.path.join(R, c["needs"][0])), f"missing {c['needs'][0]}"
        assert not os.path.exists(os.path.join(R, c["done"]["path"])), f"{c['name']} already done"
    print(f"guard: {len(cells)} cells, candidate pools present")

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
            print([c["name"] for c in new])
            print()
            print(new[0]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov10c")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
