"""Register the Wednesday programme: tuned-LegNet ensembles, weights saved, on every reservoir.

Phase 1 of the plan, and the part that fits inside the standing budget.

ARM A -- "start from 0". Random acquisition from a reservoir IS the reservoir's nested pool
at size D, so this arm is simply the tuned per-size LegNet recipe trained on each
reservoir's own 30k pool. Five seeds each, because 3 seeds recover only 65-75% of the
20-seed ensemble gain and 5 recover 84-89%, and the between-reservoir gaps we are trying to
resolve are smaller than the single-seed spread.

ARM B -- "start from 30k, then acquire". One shared seed set at 30k (so every acquisition
method starts from the SAME model and the comparison is of the selection rule alone),
then +30k chosen by each rule and retrained at 60k.

The controls are not optional here and are registered at the same time as the methods they
control: `random` is the floor every acquisition rule must beat, and `badge_epistemic_only`
and `badge_kmeanspp_only` exist so a BADGE gain can be attributed to the fusion rather than
to either half. BADGE and BatchBALD both have failure modes that return a full, plausible
batch while actually selecting at random, with nothing in the output to say so.

Every cell passes --save-weights: the point of this run is a bank of trained students that
can be scored against new eval sets over the following days without retraining.

    python3 scratchpad/register_nov22.py --dry-run
    python3 scratchpad/register_nov22.py
"""

from __future__ import annotations

import argparse
import json
import os

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = f"{R}/scratchpad/dispatch_registry.json"
POOLS = f"{R}/outputs/grid_pools/k562_nested"
# Its own output root: the genomic 30k directory under menu_retrain_v0 already holds
# COMPLETE_s0..s2 from an earlier run trained WITHOUT --save-weights, so sharing it would
# mark these cells done on sight and leave the bank without the weights this run exists for.
OUT_ROOT = "outputs/wed41"

# The per-size recipe measured on Nov 1 (docs/TUNED_LEGNET.md). Weight decay roughly halves
# at each size step; the published 100-epoch schedule is badly under-trained at our sizes.
CFG_FOR_SIZE = {10_000: "e800_wd4", 30_000: "e500_wd2", 100_000: "e500", 300_000: "e300_wd05"}

# Arm A: every reservoir with a 30k pool, minus the ones that cannot carry the comparison.
# `zoonomia_hx` is excluded for the reason it was held back from the shared folder -- no
# params provenance stamp and 423 duplicate rows in 100k -- and the `accessible_*` tiles
# are a different kind of object (cell-type-specific subsets, not a general reservoir).
ARM_A = [
    "genomic",
    "random",
    "mutagenesis",
    "mutagenesis_mr10",
    "dinuc_shuffle",
    "gc_matched",
    "random_gc_sweep",
    "motif_shared_core",
    "motif_ct_enriched",
    "motif_syntax_core",
    "zoonomia",
    "zoonomia_orthologs",
    "hg38_random",
    "evoaug",
    "evoaug_ours_default",
    "evoaug_paper2023",
    "evo2",
    "d3_steps64",
    "d3_steps256",
]
SEEDS_A = {0: 10236470, 1: 10236471, 2: 10236472, 3: 10236473, 4: 10236474}


def cells_arm_a(d: int) -> list[dict]:
    cfg = CFG_FOR_SIZE[d]
    out = []
    for res in ARM_A:
        pool = f"{POOLS}/{res}_d{d}.npz"
        if not os.path.exists(pool):
            print(f"  skip {res} d{d}: no pool")
            continue
        tag = f"bench38_ln_{cfg}_{d // 1000}k_c20" + ("" if res == "genomic" else f"_{res}")
        out_dir = f"{OUT_ROOT}/{tag}"
        for i, s in SEEDS_A.items():
            done = f"{out_dir}/COMPLETE_s{i}"
            cmd = (
                f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                f"--D {d} --cfg {cfg} --seeds {i}={s} --ctx 20 --save-weights "
                f"--out-root {R}/{OUT_ROOT}"
            )
            if res != "genomic":
                cmd += f" --pool {pool}"
            out.append(
                {
                    "name": f"wed_a{d // 1000}k_{res}_s{i}",
                    "gpus": ["h100", "v100"],
                    "ext": [],
                    "out_dir": out_dir,
                    "env": {},
                    "needs": [pool],
                    "cmd": cmd + f" && touch {R}/{done}",
                    "done": {"type": "file", "path": done},
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sizes", default="30000")
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="3 recovers 65-75%% of the 20-seed ensemble gain, 5 recovers 84-89%%",
    )
    a = ap.parse_args()

    global SEEDS_A
    SEEDS_A = {i: s for i, s in list(SEEDS_A.items())[: a.n_seeds]}
    new = []
    for d in [int(x) for x in a.sizes.split(",")]:
        new += cells_arm_a(d)

    reg = json.load(open(REG))
    have = {c["name"] for c in reg}
    fresh = [c for c in new if c["name"] not in have]
    print(f"\n{len(new)} cells built, {len(new) - len(fresh)} already registered, {len(fresh)} new")
    by_res: dict[str, int] = {}
    for c in fresh:
        by_res[c["name"].split("_", 2)[2].rsplit("_s", 1)[0]] = (
            by_res.get(c["name"].split("_", 2)[2].rsplit("_s", 1)[0], 0) + 1
        )
    for k, v in sorted(by_res.items()):
        print(f"   {k:24s} {v} seeds")
    mins = {30_000: 24, 10_000: 13, 100_000: 48, 300_000: 145}
    est = sum(mins[int(c["cmd"].split("--D ")[1].split()[0])] for c in fresh) / 60
    print(f"\nestimated {est:.0f} GPU-h (measured: 24 min per seed at 30k)")
    if a.dry_run:
        print("\n--dry-run: registry not written")
        print(json.dumps(fresh[0], indent=1) if fresh else "")
        return

    os.replace(REG, REG + ".bak_nov22")
    with open(REG, "w") as fh:
        json.dump(reg + fresh, fh, indent=1)
    print(f"\nregistry: {len(reg)} -> {len(reg) + len(fresh)} cells (backup .bak_nov22)")


if __name__ == "__main__":
    main()
