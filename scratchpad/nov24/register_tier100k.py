"""Register the 100k tier students so the tier arms enter the RESERVOIR table.

Recipe, context, out-root and GPU list are copied from the Arm A 100k motif cells
unchanged (cfg e500, outputs/menu_retrain_v0, h100|v100), so these rows are read against
motif_shared_core and motif_ct_enriched at 100k without any adjustment. Three seeds, which
is what the 30k tier arms use.

    python3 scratchpad/nov24/register_tier100k.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = f"{R}/scratchpad/dispatch_registry.json"
POOLS = f"{R}/outputs/grid_pools/k562_variants"
OUT_ROOT = "outputs/menu_retrain_v0"

D = 100_000
CFG = "e500"
SEEDS = {0: 10236490, 1: 10236491, 2: 10236492}
NAMES = ["tier_balanced", "tseq_pure"]


def cells() -> list[dict]:
    out = []
    for name in NAMES:
        pool = f"{POOLS}/{name}_d{D}.npz"
        # The trainer derives its own directory as stem.split("_d")[0] on the pool name;
        # neither of these contains "_d", checked.
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
                    "name": f"tier100k_{name}_s{i}",
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
    a = ap.parse_args()

    bad = [n for n in NAMES if "_d" in n]
    if bad:
        raise SystemExit(f"names collapse under stem.split('_d')[0]: {bad}")

    new = cells()
    reg = json.load(open(REG))
    have = {c["name"] for c in reg}
    fresh = [c for c in new if c["name"] not in have]
    print(f"{len(new)} cells built, {len(fresh)} new")
    print(f"estimated {len(fresh) * 48 / 60:.1f} GPU-h (145 min/seed at 300k scaled to 100k)")
    for m in sorted({c["needs"][0] for c in fresh if not os.path.exists(c["needs"][0])}):
        print(f"  pool not on disk yet, cell held: {os.path.basename(m)}")
    if a.dry_run:
        print("\n--dry-run: registry not written")
        return
    os.replace(REG, REG + ".bak_tier100k")
    with open(REG, "w") as fh:
        json.dump(reg + fresh, fh, indent=1)
    print(f"\nregistry: {len(reg)} -> {len(reg) + len(fresh)} cells")


if __name__ == "__main__":
    main()
