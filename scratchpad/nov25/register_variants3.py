"""Register the density-ladder students: 4 variants x 30k x 3 seeds.

Same recipe, size, context and out-root as Arm A2/A3, so these rows join that one table
and are read against the existing density rungs without adjustment. Seeds differ from
A3's 10236480-2 for the same reason A3's differed from A2's: these pools did not exist
when those seeds were drawn, and reusing them would imply a pairing that is not there.

    python3 scratchpad/nov25/register_variants3.py --dry-run
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

D = 30_000
CFG = "e500_wd2"
SEEDS = {0: 10236490, 1: 10236491, 2: 10236492}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants3 import VARIANTS  # noqa: E402


def cells() -> list[dict]:
    out = []
    for name, strategy, _params, _why in VARIANTS:
        pool = f"{POOLS}/{name}_d30000.npz"
        # `bench38_`, matching what the trainer names its own directory. A guessed path is
        # the Nov 15 failure: the cell looks registered, the dispatcher marks it done on a
        # COMPLETE file in a directory holding no weights, and the reader returns models=0.
        tag = f"bench38_ln_{CFG}_{D // 1000}k_c20_{name}"
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
    print(f"estimated {len(fresh) * 47 / 60:.0f} GPU-h (measured: 47 min per seed at 30k)")

    missing = sorted({c["needs"][0] for c in fresh if not os.path.exists(c["needs"][0])})
    if missing:
        print(f"{len(missing)} of {len(VARIANTS)} pools not built yet; held by the needs gate")

    if a.dry_run:
        print("\n--dry-run: registry not written")
        return

    os.replace(REG, REG + ".bak_a3d")
    with open(REG, "w") as fh:
        json.dump(reg + fresh, fh, indent=1)
    print(f"\nregistry: {len(reg)} -> {len(reg) + len(fresh)} cells (backup .bak_a3d)")


if __name__ == "__main__":
    main()
