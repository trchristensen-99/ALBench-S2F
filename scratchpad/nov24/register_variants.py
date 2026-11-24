"""Register the reservoir-PARAMETER students: Arm A2, 25 variants x 30k x 3 seeds.

Arm A ranked 19 reservoir families. This ranks the settings inside four of them, plus the
four model-guided objectives registered on Nov 24, because a library order is a set of
settings and not a family name. scratchpad/nov24/variants.py is the table and says what
each row is for.

THREE SEEDS, NOT FIVE. Arm A used five, on the measured grounds that three recover only
65-75% of the twenty-seed ensemble gain. That argument applies to comparing one family
with another, where the gaps are ~0.01. Here each variant is read against its OWN family
baseline from Arm A, which has five seeds, and the within-family effects we care about
(does site density matter at all?) are either large or not worth acting on. Three seeds
at 25 variants costs 30 GPU-h; five would cost 50 and buy resolution we would not use.

Every cell saves weights, so a variant that wins can be scored against the eval panel
without retraining -- which is what Arm A's bank made possible and is the reason the
Wednesday arms were worth their GPU time twice over.

    python3 scratchpad/nov24/register_variants.py --dry-run
    python3 scratchpad/nov24/register_variants.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = f"{R}/scratchpad/dispatch_registry.json"
POOLS = f"{R}/outputs/grid_pools/k562_variants"
OUT_ROOT = "outputs/var41"

# The Nov 1 per-size recipe (docs/TUNED_LEGNET.md). 30k only: the point of this arm is
# a wide parameter comparison at one size, not a scaling curve per parameter.
D = 30_000
CFG = "e500_wd2"
SEEDS = {0: 10236470, 1: 10236471, 2: 10236472}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants import VARIANTS  # noqa: E402


def cells() -> list[dict]:
    out = []
    for name, strategy, params, _why in VARIANTS:
        pool = f"{POOLS}/{name}_d30000.npz"
        tag = f"var41_ln_{CFG}_{D // 1000}k_c20_{name}"
        out_dir = f"{OUT_ROOT}/{tag}"
        for i, s in SEEDS.items():
            done = f"{out_dir}/COMPLETE_s{i}"
            cmd = (
                f"uv run --no-sync python scripts/program_oct22/bench33_legnet.py "
                f"--D {D} --cfg {CFG} --seeds {i}={s} --ctx 20 --save-weights "
                f"--pool {pool} --out-root {R}/{OUT_ROOT}"
            )
            out.append(
                {
                    "name": f"var_a30k_{name}_s{i}",
                    "gpus": ["h100", "v100"],
                    "ext": [],
                    "out_dir": out_dir,
                    "env": {},
                    # The pool is a `needs` rather than a prerequisite job: the build
                    # array and this registration go in together, and the dispatcher
                    # simply will not start a cell whose pool is not on disk yet.
                    "needs": [pool],
                    "cmd": cmd + f" && touch {R}/{done}",
                    "done": {"type": "file", "path": done},
                    "strategy": strategy,
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    new = cells()
    reg = json.load(open(REG))
    have = {c["name"] for c in reg}
    fresh = [c for c in new if c["name"] not in have]
    print(f"{len(new)} cells built, {len(new) - len(fresh)} already registered, {len(fresh)} new")
    by_strat: dict[str, int] = {}
    for c in fresh:
        by_strat[c["strategy"]] = by_strat.get(c["strategy"], 0) + 1
    for k, v in sorted(by_strat.items()):
        print(f"   {k:22s} {v} cells")
    print(f"\nestimated {len(fresh) * 24 / 60:.0f} GPU-h (measured: 24 min per seed at 30k)")

    missing = sorted({c["needs"][0] for c in fresh if not os.path.exists(c["needs"][0])})
    if missing:
        print(f"\n{len(missing)} of {len(VARIANTS)} pools not built yet; the dispatcher")
        print("will hold those cells until build_variants.sbatch writes them.")

    if a.dry_run:
        print("\n--dry-run: registry not written")
        print(json.dumps(fresh[0], indent=1) if fresh else "")
        return

    os.replace(REG, REG + ".bak_nov24")
    with open(REG, "w") as fh:
        json.dump(reg + fresh, fh, indent=1)
    print(f"\nregistry: {len(reg)} -> {len(reg) + len(fresh)} cells (backup .bak_nov24)")


if __name__ == "__main__":
    main()
